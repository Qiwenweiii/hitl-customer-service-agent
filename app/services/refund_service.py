"""退款业务能力：判定某笔订单能否受理退款，并在人工确认后将付款状态推进为 refunding。

与「取消订单」分开建模：取消只改订单状态，退款只改付款状态。
退款判定依赖系统时间（签收时效），因此与时钟注入强绑定在本 Service；
订单状态、付款状态的枚举定义在 app/models.py，时效天数配置在 app/config.py。
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from app.config import REFUND_WINDOW_DAYS
from app.data_loader import update_order_payment_status
from app.models import Order, OrderStatus
from app.services.order_service import OrderService

# 没有签收时间也允许受理退款的订单状态：尚未发货、或已取消但付过款（款项还卡在已付状态）。
# 已发货未签收不在此列——按政策第四章应走拒收，由 NEED_REJECT_DELIVERY 单独提示。
REFUNDABLE_WITHOUT_DELIVERY_STATUSES: frozenset[OrderStatus] = frozenset(
    {"pending", "paid", "cancelled"}
)

# 退款判定结论：只有 CAN_REFUND 会进入人工确认，其余都属于不该打扰人工的直接拒绝分支。
RefundOutcome = Literal[
    "CAN_REFUND",  # 满足退款条件（仅判定，尚未受理）
    "REFUNDING_SUBMITTED",  # 本次已受理退款，付款状态改为 refunding
    "REFUND_EXPIRED",  # 已超过签收次日起的退款时效
    "NEED_REJECT_DELIVERY",  # 已发货未签收：按政策走拒收，不在本入口退款
    "NO_DELIVERY_TIME",  # 状态已签收但缺签收时间，无法核定时效（数据异常）
    "NOT_PAID",  # 从未支付，没有可退的款项
    "ALREADY_REFUNDING",  # 退款已在处理中，不重复受理
    "ALREADY_REFUNDED",  # 款项已退回，不重复受理
    "NOT_FOUND",  # 订单不存在或订单号为空
]


class RefundResult(BaseModel):
    """退款判定结果，check_refund 与 apply_refund 共用。

    ``days_remaining`` 是距时效截止日的剩余自然日：可退时为非负数（0 表示最后一天），
    超期时为负数（绝对值即已超期天数），不受签收时效约束（未发货／已取消）时为 ``None``。
    """

    outcome: RefundOutcome
    order: Order | None = None
    days_remaining: int | None = None


class RefundService:
    """退款判定与受理的业务逻辑。

    对上层（Tool / Agent）屏蔽数据来源细节：本类不直接读写文件，
    所有数据存取一律经 app/data_loader.py；订单读取复用 OrderService，
    避免订单号归一化口径在两处各自漂移。

    Args:
        now: 当前时间的取值函数，默认 ``datetime.now``。退款时效判定依赖系统时间，
            注入后测试才能固定时钟、稳定覆盖期内与超期边界。
    """

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or datetime.now
        self._order_service = OrderService()

    def check_refund(self, order_id: str) -> RefundResult:
        """只判定该订单能否受理退款，不执行受理、不改动任何数据。

        规则（自上而下匹配）：
        - 从未支付（``unpaid``）：无款可退，``NOT_PAID``。
        - 已在退款处理中 / 已退回（``refunding`` / ``refunded``）：不重复受理。
        - 有签收时间：按「签收次日起 :data:`REFUND_WINDOW_DAYS` 个自然日」核时效，
          期内 ``CAN_REFUND``（含截止日当天），超期 ``REFUND_EXPIRED``。
        - 无签收时间且订单为未发货／已取消：款项还卡在已付状态，``CAN_REFUND`` 且不受签收时效约束。
        - 无签收时间但订单已发货（在途）：按政策应走拒收，``NEED_REJECT_DELIVERY``。
        - 订单状态已是 delivered 却没有签收时间：数据异常，无法核时效，``NO_DELIVERY_TIME``。

        Args:
            order_id: 订单号，例如 ``ORD-2026-0004``；首尾空格与大小写会被归一化。

        Returns:
            :class:`RefundResult`：除订单不存在外均带订单快照与 ``days_remaining``；
            本方法不会返回 ``REFUNDING_SUBMITTED``。
        """
        order = self._order_service.get_order(order_id)
        if order is None:
            return RefundResult(outcome="NOT_FOUND")
        if order.payment_status == "unpaid":
            return RefundResult(outcome="NOT_PAID", order=order)
        if order.payment_status == "refunding":
            return RefundResult(outcome="ALREADY_REFUNDING", order=order)
        if order.payment_status == "refunded":
            return RefundResult(outcome="ALREADY_REFUNDED", order=order)

        if order.delivered_at is not None:
            # 签收当日不计入：截止日 = 签收日 + N 天，剩余天数按自然日向下取整到日期差。
            deadline = order.delivered_at.date() + timedelta(days=REFUND_WINDOW_DAYS)
            days_remaining = (deadline - self._now().date()).days
            if days_remaining >= 0:
                return RefundResult(
                    outcome="CAN_REFUND", order=order, days_remaining=days_remaining
                )
            return RefundResult(
                outcome="REFUND_EXPIRED", order=order, days_remaining=days_remaining
            )

        if order.status in REFUNDABLE_WITHOUT_DELIVERY_STATUSES:
            return RefundResult(outcome="CAN_REFUND", order=order)
        if order.status == "shipped":
            return RefundResult(outcome="NEED_REJECT_DELIVERY", order=order)
        return RefundResult(outcome="NO_DELIVERY_TIME", order=order)

    def apply_refund(self, order_id: str) -> RefundResult:
        """受理退款：满足条件时把付款状态改为 ``refunding`` 并写回数据层。

        规则：
        - 只有 :meth:`check_refund` 判定为 ``CAN_REFUND`` 的订单才会写入 ``refunding``。
        - 受理只表示「退款已提交人工/财务处理」，款项是否到账由支付渠道异步决定，
          本方法不写 ``refunded``。
        - 其余任何结论都不改动订单数据。

        Args:
            order_id: 订单号，例如 ``ORD-2026-0004``；首尾空格与大小写会被归一化。

        Returns:
            :class:`RefundResult`：真正受理时 ``outcome`` 为 ``REFUNDING_SUBMITTED``
            且订单快照的付款状态已是 ``refunding``，其余情况沿用 check_refund 的结论。
        """
        # 与取消同理：不复用上层在人工确认之前的判定，受理前以此刻数据层状态为准。
        check = self.check_refund(order_id)
        if check.outcome != "CAN_REFUND":
            return check

        update_order_payment_status(check.order.order_id, "refunding")
        return RefundResult(
            outcome="REFUNDING_SUBMITTED",
            order=check.order.model_copy(update={"payment_status": "refunding"}),
            days_remaining=check.days_remaining,
        )
