"""订单工具层：面向 LLM 的订单查询与订单取消（取消需人工确认）。

受理退款的 refund_order 见 app/tools/refund_tools.py；
本模块的 PAYMENT_STATUS_LABELS 与 is_approved 被它复用，避免两处口径漂移。
"""

from langchain.tools import tool
from langgraph.types import interrupt

from app.models import Order
from app.services.order_service import CancelOrderResult, OrderService

# 订单状态枚举值 -> 面向用户的中文说明（仅用于工具输出，便于 LLM 转述）
ORDER_STATUS_LABELS: dict[str, str] = {
    "pending": "待付款",
    "paid": "已付款",
    "shipped": "已发货",
    "delivered": "已签收",
    "cancelled": "已取消",
}

# 付款状态中文说明：与订单状态分开表达，取消成功不代表款项已退。
PAYMENT_STATUS_LABELS: dict[str, str] = {
    "unpaid": "未付款",
    "paid": "已付款",
    "refunding": "退款处理中",
    "refunded": "已退款",
}

_order_service = OrderService()


def _format_order(order: Order) -> str:
    """把 Order 模型渲染成便于 LLM 阅读与转述的文本。"""
    status_label = ORDER_STATUS_LABELS.get(order.status, order.status)
    payment_label = PAYMENT_STATUS_LABELS.get(order.payment_status, order.payment_status)
    delivered_at = (
        f"{order.delivered_at:%Y-%m-%d %H:%M:%S}" if order.delivered_at else "未签收"
    )
    return (
        f"订单号: {order.order_id}\n"
        f"商品名称: {order.product_name}\n"
        f"订单金额: {order.amount:.2f} 元\n"
        f"订单状态: {status_label}（{order.status}）\n"
        f"付款状态: {payment_label}（{order.payment_status}）\n"
        f"下单用户: {order.user_id}\n"
        f"下单时间: {order.created_at:%Y-%m-%d %H:%M:%S}\n"
        f"签收时间: {delivered_at}"
    )


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询单个订单的详情：商品名称、订单金额、订单状态、下单用户、下单时间。

    什么时候应该调用：
    - 用户询问某个具体订单的信息，例如"帮我查一下订单 ORD-2026-0001"、
      "订单 ORD-2026-0002 付款了吗"、"这个订单花了多少钱"、"我什么时候下的单"。
    - 只要问题涉及订单本身（状态 / 金额 / 商品 / 下单时间），都必须先调用本工具核实，
      不要根据上下文或常识猜测、编造订单信息。
    - 即使你怀疑该订单号不存在或只是占位编号，也必须先调用本工具，
      由本工具的返回值判定订单是否存在，不要仅口头要求用户确认订单号。

    什么时候不要调用：
    - 用户尚未提供订单号时，先向用户追问订单号（形如 ORD-2026-0001），不要调用本工具。
    - 用户问的是发货与配送进展（到哪了 / 签收了吗 / 运单号 / 快递公司），
      应调用 query_logistics 而不是本工具；本工具返回的订单状态只是订单侧状态，
      不等于实际物流状态。
    - 用户问的是退换货政策、配送时效等规则类问题，不属于订单数据查询。

    返回值：命中时返回该订单的可读信息；订单号不存在时返回未找到提示，
    此时不要编造订单信息，应请用户确认订单号是否正确。

    Args:
        order_id: 订单号，格式形如 ORD-2026-0001。
    """
    order = _order_service.get_order(order_id)
    if order is None:
        return f"未找到订单号为 {order_id} 的订单，请向用户确认订单号是否正确。"
    return _format_order(order)


def _describe_order(order: Order) -> str:
    """把订单摘要成一句话，用于取消检查结果里给用户回指是哪一笔。"""
    status_label = ORDER_STATUS_LABELS.get(order.status, order.status)
    return f"{order.order_id}（{order.product_name}，{order.amount:.2f} 元，当前状态：{status_label}）"


def _format_cancel_check(result: CancelOrderResult, order_id: str) -> str:
    """把「无需人工确认」的取消判定结论翻译成中文；业务判断已在 Service 层完成，这里只负责措辞。"""
    if result.outcome == "NOT_FOUND":
        return (
            f"未找到订单号为 {order_id} 的订单，无法受理取消。请向用户确认订单号是否正确，"
            "本次没有改动任何订单。"
        )
    order = _describe_order(result.order)
    if result.outcome == "CANCELLED_ALREADY":
        return f"订单 {order}，此前已经取消，无需重复取消。本次没有改动任何订单。"
    return (
        f"订单 {order} 不支持直接取消：已发货或已签收的订单需要走退货流程。"
        "本次没有改动任何订单，请不要告知用户订单已取消。"
    )


# interrupt payload 的类型标识，便于确认界面按类型渲染不同的确认卡片。
CANCEL_CONFIRMATION_TYPE = "order_cancellation_confirmation"


def _build_cancel_confirmation(order: Order) -> dict:
    """人工确认所需的订单快照。

    恢复时 Tool 会从中断点之前重放，本函数因此可能被执行两次，必须保持纯读、无副作用。
    """
    status_label = ORDER_STATUS_LABELS.get(order.status, order.status)
    return {
        "type": CANCEL_CONFIRMATION_TYPE,
        "order_id": order.order_id,
        "product_name": order.product_name,
        "amount": order.amount,
        "current_status": status_label,
        "question": (
            f"订单 {order.order_id}（{order.product_name}，{order.amount:.2f} 元，"
            f"当前状态：{status_label}）符合取消条件，是否确认执行取消？"
        ),
    }


def is_approved(decision: object) -> bool:
    """人工裁决解析：只接受 ``{"approved": True}``，其余取值一律按拒绝处理（fail-closed）。

    取消与退款共用这套契约，判定必须只有一份实现，否则两个入口的安全边界会各自漂移。
    """
    return isinstance(decision, dict) and decision.get("approved") is True


def _format_cancel_execution(result: CancelOrderResult, order_id: str) -> str:
    """人工确认之后的执行结果文案；只有 Service 真的取消了才能宣布取消成功。"""
    if result.outcome != "CANCELLED":
        return f"人工已确认，但订单未执行取消。{_format_cancel_check(result, order_id)}"
    order = result.order
    message = (
        f"人工已确认，订单 {order.order_id} 已取消，"
        f"状态已更新为{ORDER_STATUS_LABELS[order.status]}（{order.status}）。"
        "可以如实告知用户本次取消已完成。"
    )
    if order.payment_status == "paid":
        # 取消与退款是两个动作：取消不会退钱，否则用户会以为款项自动回来了。
        message += (
            f"注意：该订单已付款 {order.amount:.2f} 元，取消订单本身不会退款，"
            "请如实告知用户款项尚未退回，需要另外发起退款申请（调用 refund_order）。"
        )
    return message


@tool
def cancel_order(order_id: str) -> str:
    """取消订单：先检查该订单是否允许取消，允许则暂停等待人工确认，人工确认通过后才真正取消。

    什么时候应该调用：
    - 用户要求取消某个订单，例如"帮我取消订单 ORD-2026-0001"、"这个订单不要了"、
      "我想把 ORD-2026-0002 退掉/取消掉"。
    - 只要用户表达了取消意图并给出订单号，就必须先调用本工具核实该订单当前是否允许取消，
      不要根据订单号或常识自行判断能不能取消。

    什么时候不要调用：
    - 用户尚未提供订单号时，先向用户追问订单号（形如 ORD-2026-0001），不要调用本工具。
    - 用户问的是订单状态、金额等信息（应调用 query_order），
      或退换货政策等规则问题（应调用 search_knowledge）。

    执行与确认：
    - 订单不满足取消条件（不存在／已发货／已签收／此前已取消）时，直接返回原因，
      不会请求人工确认，也不会改动订单数据。
    - 订单满足取消条件时，本工具会暂停执行、把订单信息交给人工确认；
      只有人工确认通过后才会写回订单状态。
    - 你必须原样依据返回值判断结果：只有返回值出现"已取消""状态已更新"时，
      才能告知用户取消成功；出现"人工未确认"时，取消并未生效，
      不得代替人工作出批准，也不得声称订单已取消。

    返回值：中文说明，可能是 已取消（状态已更新）／人工未确认（订单未取消）／
    此前已取消／当前状态不允许取消／未找到该订单。

    Args:
        order_id: 订单号，格式形如 ORD-2026-0001。
    """
    check = _order_service.check_cancel(order_id)
    if check.outcome != "CAN_CANCEL":
        return _format_cancel_check(check, order_id)

    # 中断点之前只有读操作：恢复时本节点会从头重放，写库必须留在 interrupt 之后。
    decision = interrupt(_build_cancel_confirmation(check.order))
    if not is_approved(decision):
        return (
            f"人工未确认，订单 {check.order.order_id} 未取消，订单状态保持原样。"
            "请告知用户这次取消没有通过人工确认，不要说订单已取消。"
        )

    return _format_cancel_execution(_order_service.cancel_order(order_id), order_id)
