from datetime import datetime

import pytest
from pydantic import ValidationError

from app.data_loader import get_logistics_by_order_id, get_order_by_id, load_logistics, load_orders
from app.models import LogisticsInfo, Order


def test_order_model_can_be_created():
    order = Order(
        order_id="ORD-TEST-0001",
        user_id="U9999",
        product_name="测试商品",
        amount=199.0,
        status="paid",
        payment_status="paid",
        created_at="2026-09-18T10:00:00",
    )
    assert order.order_id == "ORD-TEST-0001"
    assert order.amount == 199.0
    assert order.status == "paid"
    assert isinstance(order.created_at, datetime)
    # 未签收的订单没有签收时间，退款时效走「不受签收约束」分支。
    assert order.delivered_at is None


def test_order_payment_status_is_required():
    """付款状态与订单状态分开建模，两者都不能省。"""
    with pytest.raises(ValidationError):
        Order(
            order_id="ORD-TEST-0003",
            user_id="U9999",
            product_name="测试商品",
            amount=1.0,
            status="paid",
            created_at="2026-09-18T10:00:00",
        )


def test_invalid_payment_status_rejected():
    with pytest.raises(ValidationError):
        Order(
            order_id="ORD-TEST-0004",
            user_id="U9999",
            product_name="测试商品",
            amount=1.0,
            status="paid",
            payment_status="already_refunded",
            created_at="2026-09-18T10:00:00",
        )


def test_logistics_model_can_be_created():
    info = LogisticsInfo(
        order_id="ORD-TEST-0001",
        tracking_number="TEST123456",
        carrier="测试快递",
        status="in_transit",
        latest_location="测试转运中心",
        latest_update="2026-09-18T12:00:00",
    )
    assert info.tracking_number == "TEST123456"
    assert info.status == "in_transit"
    assert isinstance(info.latest_update, datetime)


def test_invalid_order_status_rejected():
    with pytest.raises(ValidationError):
        Order(
            order_id="ORD-TEST-0002",
            user_id="U9999",
            product_name="测试商品",
            amount=1.0,
            status="unknown_status",
            payment_status="paid",
            created_at="2026-09-18T10:00:00",
        )


def test_orders_mock_data_loadable():
    orders = load_orders()
    assert len(orders) >= 5
    assert all(isinstance(order, Order) for order in orders)
    assert {"pending", "paid", "shipped", "delivered", "cancelled"} <= {order.status for order in orders}


def test_delivered_orders_have_delivery_time():
    """退款时效以签收时间为锚点：标记已签收却没有签收时间，判定会落到 NO_DELIVERY_TIME。"""
    delivered = [order for order in load_orders() if order.status == "delivered"]
    assert delivered, "mock 数据里没有已签收订单，退款时效分支测不到"
    assert all(order.delivered_at is not None for order in delivered)


def test_not_delivered_orders_have_no_delivery_time():
    """未签收的订单不该凭空带上签收时间，否则会被误判成已过或仍在时效内。"""
    assert all(
        order.delivered_at is None for order in load_orders() if order.status != "delivered"
    )


def test_logistics_mock_data_loadable():
    records = load_logistics()
    assert len(records) >= 5
    assert all(isinstance(record, LogisticsInfo) for record in records)
    assert {"pending", "in_transit", "delivered"} <= {record.status for record in records}


def test_existing_order_found():
    order = get_order_by_id("ORD-2026-0001")
    assert order is not None
    assert order.user_id == "U1001"


def test_missing_order_returns_none():
    assert get_order_by_id("ORD-NOT-EXIST") is None


def test_get_logistics_by_order_id():
    records = get_logistics_by_order_id("ORD-2026-0004")
    assert len(records) >= 1
    assert all(record.order_id == "ORD-2026-0004" for record in records)
