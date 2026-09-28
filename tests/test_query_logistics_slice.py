"""物流查询垂直切片测试：Tool -> LogisticsService -> data_loader -> logistics.json。"""

import inspect

from app.models import LogisticsInfo
from app.services.logistics_service import LogisticsService
from app.tools import logistics_tools
from app.tools.logistics_tools import query_logistics


def test_logistics_service_returns_logistics_models():
    records = LogisticsService().get_logistics("ORD-2026-0003")
    assert len(records) == 1
    assert all(isinstance(record, LogisticsInfo) for record in records)
    assert records[0].carrier == "京东物流"
    assert records[0].status == "in_transit"
    assert records[0].latest_location == "南京转运中心"


def test_logistics_service_normalizes_order_id():
    records = LogisticsService().get_logistics("  ord-2026-0004  ")
    assert [record.order_id for record in records] == ["ORD-2026-0004"]


def test_logistics_service_returns_empty_list_for_unknown_or_blank_order_id():
    service = LogisticsService()
    assert service.get_logistics("ORD-NOT-EXIST") == []
    assert service.get_logistics("   ") == []


def test_logistics_service_returns_all_packages_of_one_order(monkeypatch):
    captured: list[str] = []

    def fake_get_logistics_by_order_id(order_id: str) -> list[LogisticsInfo]:
        captured.append(order_id)
        return [
            LogisticsInfo(
                order_id=order_id,
                tracking_number=f"SF000000000{index}",
                carrier="顺丰速运",
                status="in_transit",
                latest_location=f"转运中心 {index}",
                latest_update="2026-09-18T10:00:00",
            )
            for index in range(1, 3)
        ]

    monkeypatch.setattr(
        "app.services.logistics_service.get_logistics_by_order_id", fake_get_logistics_by_order_id
    )

    records = LogisticsService().get_logistics("ord-2026-0099")

    assert captured == ["ORD-2026-0099"]
    assert [record.latest_location for record in records] == ["转运中心 1", "转运中心 2"]


def test_query_logistics_is_langchain_tool_exposed_to_agent():
    assert query_logistics.name == "query_logistics"
    assert "order_id" in query_logistics.args
    description = query_logistics.description
    assert "物流" in description
    assert "ORD-2026-0001" in description


def test_logistics_tools_does_not_access_data_loader_directly():
    """架构守卫：Tool 层必须通过 LogisticsService 访问数据，不能绕过它。"""
    assert "data_loader" not in inspect.getsource(logistics_tools)


def test_query_logistics_delegates_to_logistics_service(monkeypatch):
    calls: list[str] = []
    real_service = LogisticsService()

    def fake_get_logistics(order_id: str) -> list[LogisticsInfo]:
        calls.append(order_id)
        return real_service.get_logistics(order_id)

    monkeypatch.setattr(logistics_tools._logistics_service, "get_logistics", fake_get_logistics)

    output = query_logistics.invoke({"order_id": "ORD-2026-0004"})

    assert calls == ["ORD-2026-0004"]
    assert "SF97530000004" in output


def test_query_logistics_tool_returns_readable_logistics_details():
    output = query_logistics.invoke({"order_id": "ORD-2026-0004"})
    assert "ORD-2026-0004" in output
    assert "SF97530000004" in output
    assert "顺丰速运" in output
    assert "已签收" in output
    assert "北京市朝阳区望京配送站" in output
    assert "2026-09-18 14:15:00" in output


def test_query_logistics_tool_renders_multiple_packages(monkeypatch):
    packages = [
        LogisticsInfo(
            order_id="ORD-2026-0099",
            tracking_number="SF0000000001",
            carrier="顺丰速运",
            status="in_transit",
            latest_location="杭州转运中心",
            latest_update="2026-09-18T10:00:00",
        ),
        LogisticsInfo(
            order_id="ORD-2026-0099",
            tracking_number="SF0000000002",
            carrier="顺丰速运",
            status="pending",
            latest_location="上海发货仓库",
            latest_update="2026-09-18T11:00:00",
        ),
    ]
    monkeypatch.setattr(
        logistics_tools._logistics_service, "get_logistics", lambda order_id: packages
    )

    output = query_logistics.invoke({"order_id": "ORD-2026-0099"})

    assert "共 2 个包裹" in output
    assert "包裹 1/2" in output
    assert "包裹 2/2" in output
    assert "待发货" in output


def test_query_logistics_tool_handles_missing_logistics():
    output = query_logistics.invoke({"order_id": "ORD-NOT-EXIST"})
    assert "未找到" in output
    assert "ORD-NOT-EXIST" in output
