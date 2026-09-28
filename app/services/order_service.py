from typing import Literal

from pydantic import BaseModel

from app.data_loader import get_order_by_id, update_order_status
from app.models import Order, OrderStatus

# 允许取消的订单状态：发货前都可以直接取消；已发货、已签收需走退货流程，不属于取消范围。
CANCELLABLE_STATUSES: frozenset[OrderStatus] = frozenset({"pending", "paid"})

CancelOutcome = Literal[
    "CAN_CANCEL",  # 满足取消条件（仅判定，尚未执行）
    "CANCELLED",  # 本次已执行取消并写回数据
    "CANCELLED_ALREADY",  # 此前已是 cancelled，不重复执行
    "NOT_ALLOWED",  # 当前状态不允许取消
    "NOT_FOUND",  # 订单不存在或订单号为空
]


class CancelOrderResult(BaseModel):
    """取消订单的判定结果，check_cancel 与 cancel_order 共用。

    只表达业务结论与订单快照，面向用户的措辞由上层（Tool）负责，避免业务规则与话术耦合。
    """

    outcome: CancelOutcome
    order: Order | None = None


class OrderService:
    """订单查询与取消的业务逻辑。

    对上层（Tool / Agent）屏蔽数据来源细节：当前数据来自 Mock JSON，
    后续切换为真实数据库或订单中台接口时，只需修改本类实现。
    本类不直接读写文件，所有数据存取一律经 app/data_loader.py。
    """

    def get_order(self, order_id: str) -> Order | None:
        """按订单号查询订单。

        Args:
            order_id: 订单号，例如 ``ORD-2026-0001``；首尾空格与大小写会被归一化。

        Returns:
            命中的 :class:`Order`；订单不存在或订单号为空时返回 ``None``。
        """
        normalized = order_id.strip().upper()
        if not normalized:
            return None
        return get_order_by_id(normalized)

    def check_cancel(self, order_id: str) -> CancelOrderResult:
        """只判定该订单能否取消，不执行取消、不改动任何数据。

        供上层在拿到人工确认之前先行检查使用（Human-in-the-Loop 的确认前一步）。

        Args:
            order_id: 订单号，例如 ``ORD-2026-0001``；首尾空格与大小写会被归一化。

        Returns:
            :class:`CancelOrderResult`：``outcome`` 取值 ``CAN_CANCEL``（满足取消条件，
            尚未执行）/ ``CANCELLED_ALREADY`` / ``NOT_ALLOWED`` / ``NOT_FOUND``，
            除订单不存在外均带订单快照。本方法不会返回 ``CANCELLED``。
        """
        order = self.get_order(order_id)
        if order is None:
            return CancelOrderResult(outcome="NOT_FOUND")
        if order.status == "cancelled":
            return CancelOrderResult(outcome="CANCELLED_ALREADY", order=order)
        if order.status not in CANCELLABLE_STATUSES:
            return CancelOrderResult(outcome="NOT_ALLOWED", order=order)
        return CancelOrderResult(outcome="CAN_CANCEL", order=order)

    def cancel_order(self, order_id: str) -> CancelOrderResult:
        """按业务规则执行取消；允许取消时把状态改为 cancelled 并写回数据层。

        规则：
        - ``pending`` / ``paid``：允许取消，状态改为 ``cancelled`` 并持久化。
        - ``shipped`` / ``delivered``：不允许取消，不改动订单数据。
        - ``cancelled``：不重复执行取消，不改动订单数据。
        - 订单不存在或订单号为空：不改动任何数据。

        Args:
            order_id: 订单号，例如 ``ORD-2026-0001``；首尾空格与大小写会被归一化。

        Returns:
            :class:`CancelOrderResult`：真正执行了取消时 ``outcome`` 为 ``CANCELLED``，
            其余情况直接沿用 :meth:`check_cancel` 的判定结论；取消成功时订单快照的状态
            已是 ``cancelled``。
        """
        # 不复用上层在人工确认之前拿到的判定：确认期间订单状态可能已被其它流程改写，
        # 执行取消只以此刻数据层里的当前状态为准。
        check = self.check_cancel(order_id)
        if check.outcome != "CAN_CANCEL":
            return check

        update_order_status(check.order.order_id, "cancelled")
        return CancelOrderResult(
            outcome="CANCELLED",
            order=check.order.model_copy(update={"status": "cancelled"}),
        )
