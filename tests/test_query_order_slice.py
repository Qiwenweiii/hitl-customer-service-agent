"""订单查询垂直切片测试：Tool -> OrderService -> data_loader -> orders.json。"""

import inspect

from app.models import Order
from app.services.order_service import OrderService
from app.tools import order_tools
from app.tools.order_tools import query_order


def test_order_service_returns_order_model():
    order = OrderService().get_order("ORD-2026-0001")
    assert isinstance(order, Order)
    assert order.user_id == "U1001"
    assert order.product_name == "无线降噪耳机"
    assert order.status == "pending"


def test_order_service_normalizes_order_id():
    order = OrderService().get_order("  ord-2026-0001  ")
    assert order is not None
    assert order.order_id == "ORD-2026-0001"


def test_order_service_returns_none_for_unknown_or_blank_order_id():
    service = OrderService()
    assert service.get_order("ORD-NOT-EXIST") is None
    assert service.get_order("   ") is None


def test_query_order_is_langchain_tool_exposed_to_agent():
    assert query_order.name == "query_order"
    assert "order_id" in query_order.args
    description = query_order.description
    assert "订单" in description
    assert "ORD-2026-0001" in description


def test_order_tools_does_not_access_data_loader_directly():
    """架构守卫：Tool 层必须通过 OrderService 访问数据，不能绕过它。"""
    assert "data_loader" not in inspect.getsource(order_tools)


def test_query_order_delegates_to_order_service(monkeypatch):
    calls: list[str] = []
    real_service = OrderService()

    def fake_get_order(order_id: str) -> Order | None:
        calls.append(order_id)
        return real_service.get_order(order_id)

    monkeypatch.setattr(order_tools._order_service, "get_order", fake_get_order)

    output = query_order.invoke({"order_id": "ORD-2026-0003"})

    assert calls == ["ORD-2026-0003"]
    assert "27英寸4K显示器" in output


def test_query_order_tool_returns_readable_order_details():
    output = query_order.invoke({"order_id": "ORD-2026-0002"})
    assert "ORD-2026-0002" in output
    assert "机械键盘" in output
    assert "599.00" in output
    assert "已付款" in output
    assert "2026-09-13 14:20:00" in output


def test_query_order_tool_handles_missing_order():
    output = query_order.invoke({"order_id": "ORD-NOT-EXIST"})
    assert "未找到" in output
    assert "ORD-NOT-EXIST" in output
