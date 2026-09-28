import json

from app.config import BASE_DIR
from app.models import LogisticsInfo, Order, OrderStatus, PaymentStatus

ORDERS_FILE = BASE_DIR / "data" / "mock" / "orders.json"
LOGISTICS_FILE = BASE_DIR / "data" / "mock" / "logistics.json"


def load_orders() -> list[Order]:
    with ORDERS_FILE.open(encoding="utf-8") as f:
        return [Order.model_validate(item) for item in json.load(f)]


def load_logistics() -> list[LogisticsInfo]:
    with LOGISTICS_FILE.open(encoding="utf-8") as f:
        return [LogisticsInfo.model_validate(item) for item in json.load(f)]


def get_order_by_id(order_id: str) -> Order | None:
    return next((order for order in load_orders() if order.order_id == order_id), None)


def get_logistics_by_order_id(order_id: str) -> list[LogisticsInfo]:
    return [info for info in load_logistics() if info.order_id == order_id]


def _write_order_field(order_id: str, field: str, value: object) -> bool:
    """按订单号替换单个字段并写回，其余订单与字段（含中文与缩进）保持原样。"""
    with ORDERS_FILE.open(encoding="utf-8") as f:
        raw_orders = json.load(f)

    target = next((item for item in raw_orders if item.get("order_id") == order_id), None)
    if target is None:
        return False
    if target.get(field) != value:
        target[field] = value
        with ORDERS_FILE.open("w", encoding="utf-8") as f:
            json.dump(raw_orders, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return True


def update_order_status(order_id: str, status: OrderStatus) -> bool:
    """更新指定订单的状态并写回 Mock JSON。

    这是修改订单**状态**的唯一入口：业务规则由 Service 层判定，本函数只负责落地。

    Args:
        order_id: 已归一化的订单号，例如 ``ORD-2026-0001``。
        status: 目标订单状态。

    Returns:
        订单存在并已完成写入时返回 ``True``（值原本就相同则不重写文件）；
        订单不存在时返回 ``False``，且不会改动数据文件。
    """
    return _write_order_field(order_id, "status", status)


def update_order_payment_status(order_id: str, payment_status: PaymentStatus) -> bool:
    """更新指定订单的付款状态并写回 Mock JSON。

    与 :func:`update_order_status` 并列，是修改**付款状态**的唯一入口：
    退款受理（``refunding``）只能经此写入，Service 与 Tool 都不直接改文件。

    Args:
        order_id: 已归一化的订单号，例如 ``ORD-2026-0004``。
        payment_status: 目标付款状态。

    Returns:
        订单存在并已完成写入时返回 ``True``；订单不存在时返回 ``False`` 且不改动数据文件。
    """
    return _write_order_field(order_id, "payment_status", payment_status)
