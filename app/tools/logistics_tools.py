from langchain.tools import tool

from app.models import LogisticsInfo
from app.services.logistics_service import LogisticsService

# 物流状态枚举值 -> 面向用户的中文说明（仅用于工具输出，便于 LLM 转述）
LOGISTICS_STATUS_LABELS: dict[str, str] = {
    "pending": "待发货",
    "in_transit": "运输中",
    "delivered": "已签收",
}

_logistics_service = LogisticsService()


def _format_package(info: LogisticsInfo, index: int, total: int) -> str:
    """把单个包裹的物流信息渲染成便于 LLM 阅读与转述的文本。"""
    status_label = LOGISTICS_STATUS_LABELS.get(info.status, info.status)
    title = f"包裹 {index}/{total}" if total > 1 else "物流信息"
    return (
        f"{title}\n"
        f"运单号: {info.tracking_number}\n"
        f"快递公司: {info.carrier}\n"
        f"物流状态: {status_label}（{info.status}）\n"
        f"最新位置: {info.latest_location}\n"
        f"更新时间: {info.latest_update:%Y-%m-%d %H:%M:%S}"
    )


@tool
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流信息：运单号、快递公司、物流状态、最新位置、更新时间。

    什么时候应该调用：
    - 用户询问某个订单的发货或配送进展，例如"订单 ORD-2026-0004 到哪了"、
      "我的快递发货了吗"、"这个包裹签收了吗"、"运单号是多少"。
    - 只要问题涉及物流（是否发货 / 在途位置 / 是否签收 / 运单号 / 快递公司），
      都必须先调用本工具核实，不要根据订单号格式或常识猜测、编造物流信息。
    - 即使你怀疑该订单号不存在或只是占位编号，也必须先调用本工具，
      由本工具的返回值判定订单是否发货、物流是否存在，不要仅口头要求用户确认订单号。

    什么时候不要调用：
    - 用户尚未提供订单号时，先向用户追问订单号（形如 ORD-2026-0001），不要调用本工具。
    - 用户问的是订单本身（商品 / 金额 / 下单时间 / 是否付款），应调用 query_order 而不是本工具；
      本工具不返回商品、金额与下单时间。
    - 用户问的是配送时效政策、运费规则等规则类问题，不属于物流数据查询。

    返回值：命中时返回该订单的物流信息（一个订单可能拆成多个包裹）；
    无物流记录时返回未找到提示，此时不要编造物流信息，
    应如实告知并请用户确认订单号是否正确。

    Args:
        order_id: 订单号，格式形如 ORD-2026-0001。
    """
    packages = _logistics_service.get_logistics(order_id)
    if not packages:
        return (
            f"未找到订单号为 {order_id} 的物流信息，"
            "可能是该订单尚未发货或订单号有误，请向用户确认订单号是否正确。"
        )
    total = len(packages)
    blocks = [_format_package(info, index, total) for index, info in enumerate(packages, start=1)]
    header = f"订单号: {packages[0].order_id}" + (f"，共 {total} 个包裹" if total > 1 else "")
    return "\n\n".join([header, *blocks])
