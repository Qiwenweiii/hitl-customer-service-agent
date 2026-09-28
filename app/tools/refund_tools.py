"""退款工具层：面向 LLM 的退款受理入口，判定与写入都在 RefundService，这里只管调用时机与措辞。

与订单查询/取消工具分文件维护：退款是唯一带时效判定、且只推进付款状态（不碰订单状态）的动作，
措辞上还必须始终区分「已受理」与「已到账」。
"""

from langchain.tools import tool
from langgraph.types import interrupt

from app.services.refund_service import RefundResult, RefundService
from app.tools.order_tools import PAYMENT_STATUS_LABELS, is_approved

_refund_service = RefundService()

# interrupt payload 的类型标识：退款确认与取消确认共用一套解析契约，只靠 type 区分渲染。
REFUND_CONFIRMATION_TYPE = "order_refund_confirmation"


def _build_refund_confirmation(result: RefundResult) -> dict:
    """人工受理退款所需的信息快照。

    恢复时 Tool 会从中断点之前重放，本函数因此可能被执行两次，必须保持纯读、无副作用。
    """
    order = result.order
    payment_label = PAYMENT_STATUS_LABELS.get(order.payment_status, order.payment_status)
    delivered_at = (
        f"{order.delivered_at:%Y-%m-%d %H:%M:%S}" if order.delivered_at else "未签收"
    )
    if result.days_remaining is None:
        window_note = "该订单尚未签收，不受签收时效约束"
    elif result.days_remaining == 0:
        window_note = "距退款时效截止日还剩 0 天（今天即最后一天）"
    else:
        window_note = f"距退款时效截止日还剩 {result.days_remaining} 个自然日"
    return {
        "type": REFUND_CONFIRMATION_TYPE,
        "order_id": order.order_id,
        "product_name": order.product_name,
        "amount": order.amount,
        "payment_status": payment_label,
        "delivered_at": delivered_at,
        "days_remaining": result.days_remaining,
        "question": (
            f"订单 {order.order_id}（{order.product_name}，{order.amount:.2f} 元，"
            f"付款状态：{payment_label}，签收时间：{delivered_at}）符合退款条件，"
            f"{window_note}。是否确认受理退款？"
            "注意：确认仅代表受理，款项到账由支付渠道异步完成。"
        ),
    }


def _format_refund_check(result: RefundResult, order_id: str) -> str:
    """把无需人工确认的退款判定结论翻译成中文；业务判断已在 Service 层完成，这里只负责措辞。

    刻意不复述具体时效天数：期限数字以知识库里的退款政策为准，写进工具文案会造成两处口径各自漂移。
    """
    if result.outcome == "NOT_FOUND":
        return (
            f"未找到订单号为 {order_id} 的订单，无法受理退款。请向用户确认订单号是否正确，"
            "本次没有改动任何订单。"
        )
    order = result.order
    label = PAYMENT_STATUS_LABELS.get(order.payment_status, order.payment_status)
    order_desc = f"{order.order_id}（{order.product_name}，{order.amount:.2f} 元，付款状态：{label}）"

    if result.outcome == "NOT_PAID":
        return (
            f"订单 {order_desc} 尚未付款，没有需要退回的款项。本次没有改动任何订单。"
            "如用户想终止这笔订单，应走取消订单（cancel_order），不要说退款已受理。"
        )
    if result.outcome == "ALREADY_REFUNDING":
        return (
            f"订单 {order_desc} 的退款此前已受理，款项正在处理中，无需重复申请。"
            "本次没有改动任何订单。到账时间以支付渠道为准，不得另报一个时间。"
        )
    if result.outcome == "ALREADY_REFUNDED":
        return (
            f"订单 {order_desc} 的款项此前已退回，无需重复申请。本次没有改动任何订单，"
            "也不要再次宣称退款受理成功。"
        )
    if result.outcome == "REFUND_EXPIRED":
        overdue = abs(result.days_remaining or 0)
        return (
            f"订单 {order_desc} 已超过该单的退款时效（自签收次日起算，至今已超期 {overdue} 个自然日），"
            "本入口不受理退款申请。本次没有改动任何订单，请如实告知用户已超期，"
            "并建议按售后政策走退货／换货／维修或转人工客服，不要说退款已受理。"
        )
    if result.outcome == "NEED_REJECT_DELIVERY":
        return (
            f"订单 {order_desc} 已发货但尚未签收，按政策应先拒收包裹，"
            "商品退回仓库后再进入退款流程。本次没有改动任何订单，不要受理在途退款。"
        )
    return (
        f"订单 {order_desc} 已标记为已签收，但系统缺少签收时间，无法核对退款时效，"
        "请转人工客服核实。本次没有改动任何订单，不要告知用户退款已受理。"
    )


def _format_refund_execution(result: RefundResult, order_id: str) -> str:
    """人工确认之后的受理结果文案；款项到账与否不由本工具决定，措辞必须留有余地。"""
    if result.outcome != "REFUNDING_SUBMITTED":
        return f"人工已确认，但退款未受理。{_format_refund_check(result, order_id)}"
    order = result.order
    return (
        f"人工已确认，订单 {order.order_id} 的退款申请已受理，"
        f"付款状态已更新为退款处理中（{order.payment_status}）。"
        f"退款金额按实付 {order.amount:.2f} 元原路退回；款项是否到账、多久到账一律以支付渠道和"
        "退款政策为准。只能告知用户「退款已受理、正在处理」，"
        "不得说款项已到账或退款已完成。"
    )


@tool
def refund_order(order_id: str) -> str:
    """受理订单退款：先检查该订单是否允许退款，允许则暂停等待人工确认，确认通过后才受理。

    什么时候应该调用：
    - 用户要求退款／退钱／把钱退回来，例如"ORD-2026-0004 我要退款""帮我把这笔订单退款"。
    - 用户已取消过订单但款项还没退回时，同样调用本工具受理退款。
    - 只要用户表达了退款意图并给出订单号，就必须先调用本工具核实能否受理，
      不要根据订单状态、签收时间或常识自行判断能不能退。

    什么时候不要调用：
    - 用户尚未提供订单号时，先向用户追问订单号（形如 ORD-2026-0001），不要调用本工具。
    - 用户问的是退款政策、时效天数、到账时间等规则问题（应调用 search_knowledge）。
    - 用户要的是取消订单（应调用 cancel_order）、查订单或物流状态。

    执行与确认：
    - 不满足退款条件（不存在／未付款／已在退款中／已退过／超过退款时效／已发货未签收）时，
      直接返回原因，不会请求人工确认，也不会改动订单数据。
    - 满足条件时本工具会暂停执行、把订单信息交给人工确认；只有人工确认通过，
      付款状态才会变为退款处理中（refunding）。
    - 本工具只完成"受理"，款项是否到账由支付渠道异步决定：只有返回值出现"已受理"
      才能说退款申请已提交，禁止说成"已到账""退款完成"。

    返回值：中文说明，可能是 退款已受理（处理中）／人工未确认（未受理）／
    已超过退款时效／在途请先拒收／未付款／退款已在处理中／款项已退回／缺少签收时间／未找到该订单。

    Args:
        order_id: 订单号，格式形如 ORD-2026-0001。
    """
    check = _refund_service.check_refund(order_id)
    if check.outcome != "CAN_REFUND":
        return _format_refund_check(check, order_id)

    # 中断点之前只有读操作：恢复时本节点会从头重放，写库必须留在 interrupt 之后。
    decision = interrupt(_build_refund_confirmation(check))
    if not is_approved(decision):
        return (
            f"人工未确认，订单 {check.order.order_id} 的退款未受理，付款状态保持原样。"
            "请告知用户这次退款没有通过人工确认，不要说退款已受理。"
        )

    return _format_refund_execution(_refund_service.apply_refund(order_id), order_id)
