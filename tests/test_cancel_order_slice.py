"""取消订单垂直切片测试：Service 判定/执行 + Tool 检查后经 interrupt 等人工确认 + 数据层写回。

链路：
- cancel_order Tool -> OrderService.check_cancel（只判定）-> interrupt 人工确认 -> OrderService.cancel_order
- OrderService.cancel_order -> data_loader.update_order_status -> orders.json（真正执行）

真实 mock 文件只读不写：所有可能改动状态的测试都把 data_loader.ORDERS_FILE 指向
tmp_path 下的副本，末尾另有哨兵断言真实文件未被污染。
"""

import ast
import inspect
import json
import shutil
from pathlib import Path

import pytest

import app.data_loader as data_loader
from app.services import order_service as order_service_module
from app.services.order_service import CANCELLABLE_STATUSES, CancelOrderResult, OrderService
from app.tools import order_tools
from app.tools.order_tools import CANCEL_CONFIRMATION_TYPE, ORDER_STATUS_LABELS, cancel_order

# 在任何打桩之前记下真实的 mock 数据文件与内容，用于末尾的"测试没有污染真实数据"哨兵。
BASELINE_ORDERS_FILE = data_loader.ORDERS_FILE
BASELINE_ORDERS_TEXT = BASELINE_ORDERS_FILE.read_text(encoding="utf-8")

# 订单状态 -> cancel_order（真正执行）的预期结论。
EXPECTED_OUTCOME: dict[str, str] = {
    "pending": "CANCELLED",
    "paid": "CANCELLED",
    "shipped": "NOT_ALLOWED",
    "delivered": "NOT_ALLOWED",
    "cancelled": "CANCELLED_ALREADY",
}

# check_cancel 只判定不改数据，因此除 pending/paid 得到 CAN_CANCEL 外，
# 数据层里应恒等于判定前的原状态；它的结论集合里不允许出现 CANCELLED。
EXPECTED_CHECK_OUTCOME: dict[str, str] = {
    "pending": "CAN_CANCEL",
    "paid": "CAN_CANCEL",
    "shipped": "NOT_ALLOWED",
    "delivered": "NOT_ALLOWED",
    "cancelled": "CANCELLED_ALREADY",
}

NORMALIZATIONS = {
    "全小写": lambda order_id: order_id.lower(),
    "小写且带首尾空格": lambda order_id: f"  {order_id.lower()}  ",
    "标准大写": lambda order_id: order_id,
}


@pytest.fixture
def orders_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把订单数据文件重定向到 tmp_path 副本，返回副本路径。"""
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    monkeypatch.setattr(data_loader, "ORDERS_FILE", copy_path)
    return copy_path


@pytest.fixture
def forbid_status_write(monkeypatch: pytest.MonkeyPatch):
    """把数据层的写入函数换成抛错版本：任何测试路径试图真正改状态都会立刻失败。"""

    def _forbid(*args, **kwargs):
        raise AssertionError("本阶段不允许真正写入订单状态")

    monkeypatch.setattr(order_service_module, "update_order_status", _forbid)


def read_raw_orders(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_of_stored_order(path: Path, order_id: str) -> str:
    return next(item["status"] for item in read_raw_orders(path) if item["order_id"] == order_id)


def order_id_with_status(status: str) -> str:
    """从当前数据层里取一条指定状态的订单，避免把订单号写死在断言里。"""
    matched = [order.order_id for order in data_loader.load_orders() if order.status == status]
    assert matched, f"mock 数据中没有状态为 {status} 的订单，测试矩阵需要更新"
    return matched[0]


def imported_names(module) -> set[str]:
    """收集模块的导入符号，用于分层守卫（比文本匹配可靠，不会被注释误伤）。"""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


# ---------------------------------------------------------------------------
# Service 层：取消规则（第一阶段）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(EXPECTED_OUTCOME))
def test_cancel_order_outcome_follows_order_status(status: str, orders_file: Path):
    order_id = order_id_with_status(status)

    result = OrderService().cancel_order(order_id)

    assert isinstance(result, CancelOrderResult)
    assert result.outcome == EXPECTED_OUTCOME[status]
    assert result.order is not None and result.order.order_id == order_id
    # 允许取消时状态已改为 cancelled；不允许取消时订单状态保持原样。
    assert result.order.status == status_of_stored_order(orders_file, order_id)


def test_cancel_pending_order_persists_cancelled_status(orders_file: Path):
    order_id = order_id_with_status("pending")

    result = OrderService().cancel_order(order_id)

    assert result.outcome == "CANCELLED"
    assert status_of_stored_order(orders_file, order_id) == "cancelled"


def test_cancel_paid_order_persists_cancelled_status(orders_file: Path):
    order_id = order_id_with_status("paid")

    result = OrderService().cancel_order(order_id)

    assert result.outcome == "CANCELLED"
    assert status_of_stored_order(orders_file, order_id) == "cancelled"


def test_cancelled_status_can_be_read_back_through_query_path(orders_file: Path):
    """取消成功后，沿用现有查询链路必须能读到 cancelled（写回真实生效）。"""
    order_id = order_id_with_status("pending")
    service = OrderService()

    service.cancel_order(order_id)
    order = service.get_order(order_id)

    assert order is not None
    assert order.status == "cancelled"


def test_cancel_shipped_order_keeps_status_unchanged(orders_file: Path):
    order_id = order_id_with_status("shipped")

    result = OrderService().cancel_order(order_id)

    assert result.outcome == "NOT_ALLOWED"
    assert result.order is not None and result.order.status == "shipped"
    assert status_of_stored_order(orders_file, order_id) == "shipped"


def test_cancel_delivered_order_keeps_status_unchanged(orders_file: Path):
    order_id = order_id_with_status("delivered")

    result = OrderService().cancel_order(order_id)

    assert result.outcome == "NOT_ALLOWED"
    assert result.order is not None and result.order.status == "delivered"
    assert status_of_stored_order(orders_file, order_id) == "delivered"


def test_cancel_already_cancelled_order_does_not_repeat(orders_file: Path):
    order_id = order_id_with_status("cancelled")

    result = OrderService().cancel_order(order_id)

    assert result.outcome == "CANCELLED_ALREADY"
    assert result.order is not None and result.order.status == "cancelled"
    assert status_of_stored_order(orders_file, order_id) == "cancelled"


def test_cancel_is_idempotent_across_two_calls(orders_file: Path):
    """同一订单取消一次后再次取消，走「已取消」分支且不重复写数据。"""
    order_id = order_id_with_status("pending")
    service = OrderService()

    first = service.cancel_order(order_id)
    second = service.cancel_order(order_id)

    assert first.outcome == "CANCELLED"
    assert second.outcome == "CANCELLED_ALREADY"
    assert status_of_stored_order(orders_file, order_id) == "cancelled"


def test_cancel_order_rechecks_current_status_before_writing(orders_file: Path):
    """设计要点：执行取消前 Service 自行按数据层当前状态再判一次，不信任上层早先的 CAN_CANCEL。"""
    order_id = order_id_with_status("pending")
    service = OrderService()

    assert service.check_cancel(order_id).outcome == "CAN_CANCEL"
    # 判定之后、执行之前，订单被别的流程推进到已发货（对应 Tool 里等待人工确认的那段时间）。
    data_loader.update_order_status(order_id, "shipped")

    result = service.cancel_order(order_id)

    assert result.outcome == "NOT_ALLOWED"
    assert result.order is not None and result.order.status == "shipped"
    assert status_of_stored_order(orders_file, order_id) == "shipped"


def test_cancel_unknown_order_returns_not_found_without_touching_data(orders_file: Path):
    before = orders_file.read_text(encoding="utf-8")

    result = OrderService().cancel_order("ORD-NOT-EXIST")

    assert result.outcome == "NOT_FOUND"
    assert result.order is None
    assert orders_file.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("case", sorted(NORMALIZATIONS))
def test_cancel_normalizes_order_id_case_and_spaces(orders_file: Path, case: str):
    order_id = order_id_with_status("pending")

    result = OrderService().cancel_order(NORMALIZATIONS[case](order_id))

    assert result.outcome == "CANCELLED"
    assert result.order is not None and result.order.order_id == order_id
    assert status_of_stored_order(orders_file, order_id) == "cancelled"


def test_cancel_blank_order_id_returns_not_found(orders_file: Path):
    before = orders_file.read_text(encoding="utf-8")

    result = OrderService().cancel_order("   ")

    assert result.outcome == "NOT_FOUND"
    assert orders_file.read_text(encoding="utf-8") == before


def test_cancellable_statuses_cover_pending_and_paid():
    """业务口径：只有发货前的订单可以直接取消。"""
    assert CANCELLABLE_STATUSES == frozenset({"pending", "paid"})


# ---------------------------------------------------------------------------
# Service 层：check_cancel 只判定、不改数据（第二阶段 Tool 依赖它）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(EXPECTED_CHECK_OUTCOME))
def test_check_cancel_reports_outcome_without_writing(
    status: str, orders_file: Path, forbid_status_write
):
    order_id = order_id_with_status(status)

    result = OrderService().check_cancel(order_id)

    assert result.outcome == EXPECTED_CHECK_OUTCOME[status]
    assert status_of_stored_order(orders_file, order_id) == status


def test_check_cancel_returns_not_found_for_unknown_order(orders_file: Path, forbid_status_write):
    result = OrderService().check_cancel("ORD-NOT-EXIST")

    assert result.outcome == "NOT_FOUND"
    assert result.order is None


# ---------------------------------------------------------------------------
# 数据层
# ---------------------------------------------------------------------------


def test_update_order_status_returns_true_and_persists(orders_file: Path):
    assert data_loader.update_order_status("ORD-2026-0002", "cancelled") is True
    assert status_of_stored_order(orders_file, "ORD-2026-0002") == "cancelled"


def test_update_order_status_returns_false_for_unknown_order_without_touching_data(
    orders_file: Path,
):
    before = orders_file.read_text(encoding="utf-8")

    assert data_loader.update_order_status("ORD-NOT-EXIST", "cancelled") is False
    assert orders_file.read_text(encoding="utf-8") == before


def test_update_order_status_keeps_other_fields_and_file_format(orders_file: Path):
    """写回只替换目标订单的 status，其它订单与字段（含中文、缩进、末尾换行）保持原样。"""
    original = read_raw_orders(orders_file)

    data_loader.update_order_status("ORD-2026-0003", "cancelled")

    updated = read_raw_orders(orders_file)
    assert [item["order_id"] for item in updated] == [item["order_id"] for item in original]
    untouched = [
        (before, after)
        for before, after in zip(original, updated, strict=True)
        if before["order_id"] != "ORD-2026-0003"
    ]
    assert untouched
    assert all(before == after for before, after in untouched)
    changed = next(item for item in updated if item["order_id"] == "ORD-2026-0003")
    original_changed = next(item for item in original if item["order_id"] == "ORD-2026-0003")
    assert changed["status"] == "cancelled"
    assert {key: value for key, value in changed.items() if key != "status"} == {
        key: value for key, value in original_changed.items() if key != "status"
    }
    text = orders_file.read_text(encoding="utf-8")
    assert text.endswith("]\n")
    assert "无线降噪耳机" in text  # ensure_ascii=False，中文不被转义
    assert '\n  {\n    "order_id"' in text  # 沿用原有 2 空格缩进结构


# ---------------------------------------------------------------------------
# Tool 层：cancel_order 检查 -> interrupt 人工确认 -> 确认通过才执行（第三阶段）
#
# interrupt() 只能在 LangGraph 运行上下文里调用（裸 tool.invoke 会抛
# "Called get_config outside of a runnable context"），所以 Tool 级测试把
# order_tools.interrupt 换成记录器：断言的是「有没有进入中断点、payload 内容、
# 人工裁决值如何被采纳」；真正的暂停与恢复在 tests/test_agent_slice.py 的
# Agent 级闭环里用 InMemorySaver + Command(resume=...) 验证。
# ---------------------------------------------------------------------------


class InterruptRecorder:
    """假 interrupt：记录人工确认 payload，并把预设的裁决值当作人工回应返回。"""

    def __init__(self, decision: object = None) -> None:
        self.payloads: list[dict] = []
        self.decision = decision

    def __call__(self, payload: dict) -> object:
        self.payloads.append(payload)
        return self.decision


def _install_interrupt(monkeypatch: pytest.MonkeyPatch, decision: object) -> InterruptRecorder:
    recorder = InterruptRecorder(decision)
    monkeypatch.setattr(order_tools, "interrupt", recorder)
    return recorder


def _forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("没有人工批准不得执行取消")

    monkeypatch.setattr(order_tools._order_service, "cancel_order", forbidden)


def test_cancel_order_tool_is_langchain_tool():
    assert cancel_order.name == "cancel_order"
    assert "order_id" in cancel_order.args
    description = cancel_order.description
    assert "取消" in description
    assert "ORD-2026-0001" in description
    assert "人工确认" in description


@pytest.mark.parametrize("status", ["pending", "paid"])
def test_cancellable_order_reaches_human_confirmation_with_full_payload(
    status: str, orders_file: Path, forbid_status_write, monkeypatch: pytest.MonkeyPatch
):
    """A：pending / paid 会进入人工确认，payload 带齐确认所需信息，此刻还没写库。"""
    order_id = order_id_with_status(status)
    recorder = _install_interrupt(monkeypatch, {})
    _forbid_execution(monkeypatch)

    output = cancel_order.invoke({"order_id": order_id})

    assert len(recorder.payloads) == 1
    payload = recorder.payloads[0]
    assert set(payload) == {
        "type",
        "order_id",
        "product_name",
        "amount",
        "current_status",
        "question",
    }
    assert payload["type"] == CANCEL_CONFIRMATION_TYPE
    assert payload["order_id"] == order_id
    assert payload["product_name"]
    assert isinstance(payload["amount"], float)
    assert payload["current_status"] == ORDER_STATUS_LABELS[status]
    assert "确认执行取消" in payload["question"]
    assert "人工未确认" in output
    assert status_of_stored_order(orders_file, order_id) == status


@pytest.mark.parametrize("status", ["shipped", "delivered", "cancelled"])
def test_orders_that_cannot_be_cancelled_never_reach_human_confirmation(
    status: str, orders_file: Path, forbid_status_write, monkeypatch: pytest.MonkeyPatch
):
    """B：不允许取消的订单直接返回原因，绝不打扰人工。"""

    def never_reached(payload: dict) -> object:
        raise AssertionError(f"状态为 {status} 的订单不应触发人工确认")

    monkeypatch.setattr(order_tools, "interrupt", never_reached)
    order_id = order_id_with_status(status)

    output = cancel_order.invoke({"order_id": order_id})

    assert order_id in output or status == "cancelled"
    assert status_of_stored_order(orders_file, order_id) == status


def test_unknown_order_never_reaches_human_confirmation(
    orders_file: Path, forbid_status_write, monkeypatch: pytest.MonkeyPatch
):
    def never_reached(payload: dict) -> object:
        raise AssertionError("订单不存在时不应触发人工确认")

    monkeypatch.setattr(order_tools, "interrupt", never_reached)

    output = cancel_order.invoke({"order_id": "ORD-NOT-EXIST"})

    assert "未找到订单号为 ORD-NOT-EXIST 的订单" in output


@pytest.mark.parametrize(
    "decision", [{"approved": False}, {}, "yes", None, {"approved": "true"}, {"approved": 1}]
)
def test_any_non_approving_decision_is_treated_as_rejection(
    decision: object, orders_file: Path, forbid_status_write, monkeypatch: pytest.MonkeyPatch
):
    """C：只有 {"approved": True} 算批准，其余全部 fail-closed 按拒绝处理且不写库。"""
    order_id = order_id_with_status("pending")
    recorder = _install_interrupt(monkeypatch, decision)
    _forbid_execution(monkeypatch)

    output = cancel_order.invoke({"order_id": order_id})

    assert recorder.payloads
    assert "人工未确认" in output and "未取消" in output
    assert "状态已更新" not in output
    assert status_of_stored_order(orders_file, order_id) == "pending"


def test_approved_decision_executes_cancellation(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """D：人工批准后 Service 才真正执行，重读订单状态为 cancelled。"""
    order_id = order_id_with_status("pending")
    recorder = _install_interrupt(monkeypatch, {"approved": True})

    output = cancel_order.invoke({"order_id": order_id})

    assert recorder.payloads[0]["order_id"] == order_id
    assert "已取消" in output and "状态已更新" in output
    assert status_of_stored_order(orders_file, order_id) == "cancelled"
    order = OrderService().get_order(order_id)
    assert order is not None and order.status == "cancelled"


def test_approval_still_reports_truth_if_order_becomes_uncancellable(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """确认期间订单被别的流程改成已发货：Tool 必须如实说没取消，而不是宣称成功。"""
    order_id = order_id_with_status("pending")

    def approve_but_order_shipped(payload: dict) -> object:
        data_loader.update_order_status(order_id, "shipped")
        return {"approved": True}

    monkeypatch.setattr(order_tools, "interrupt", approve_but_order_shipped)

    output = cancel_order.invoke({"order_id": order_id})

    assert "但订单未执行取消" in output
    assert "状态已更新" not in output
    assert status_of_stored_order(orders_file, order_id) == "shipped"


def test_cancel_order_tool_delegates_business_rules_to_service(
    orders_file: Path, forbid_status_write, monkeypatch: pytest.MonkeyPatch
):
    """Tool 只做「检查 + 是否中断 + 措辞」：判定经 check_cancel，写库经 Service。"""
    order_id = order_id_with_status("pending")
    captured: list[str] = []
    service = OrderService()

    def fake_check_cancel(raw_order_id: str) -> CancelOrderResult:
        captured.append(raw_order_id)
        return service.check_cancel(raw_order_id)

    monkeypatch.setattr(order_tools._order_service, "check_cancel", fake_check_cancel)
    _install_interrupt(monkeypatch, {})

    output = cancel_order.invoke({"order_id": NORMALIZATIONS["小写且带首尾空格"](order_id)})

    assert captured == [f"  {order_id.lower()}  "]
    assert "人工未确认" in output


def test_tool_writes_only_after_the_interrupt_point():
    """架构守卫：恢复时 Tool 节点从头重放，所以写库必须严格位于 interrupt 之后。"""
    source = inspect.getsource(order_tools.cancel_order.func)
    assert "_order_service.check_cancel(" in source
    assert "interrupt(" in source
    assert source.index("interrupt(") < source.index("_order_service.cancel_order(")
    assert "CANCELLABLE_STATUSES" not in source
    assert "update_order_status" not in source  # 写入只经 Service


def test_no_order_write_happens_before_the_interrupt_fires(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """上一条只比源码顺序，这条盯真实调用：暂停那一刻数据层一次写入都没有。"""
    order_id = order_id_with_status("pending")
    writes: list[tuple[str, str]] = []

    def recording_update_status(target_id: str, status: str) -> bool:
        writes.append((target_id, status))
        return True

    def confirm_at_interrupt(payload: dict) -> object:
        assert writes == [], "interrupt 之前已经发生订单写操作"
        assert status_of_stored_order(orders_file, order_id) == "pending"
        return {"approved": True}

    monkeypatch.setattr(order_service_module, "update_order_status", recording_update_status)
    monkeypatch.setattr(order_tools, "interrupt", confirm_at_interrupt)

    output = cancel_order.invoke({"order_id": order_id})

    # 恢复时整个节点从头重放，确认点被执行两次；两次都必须还没有任何写入。
    assert writes == [(order_id, "cancelled")]
    assert "已取消" in output


# ---------------------------------------------------------------------------
# 分层守卫
# ---------------------------------------------------------------------------


def test_service_persists_through_data_loader_not_files():
    """架构守卫：取消规则写在 Service 层，落库只能经 data_loader，Service 不碰文件。"""
    names = imported_names(order_service_module)
    assert "update_order_status" in names
    assert "json" not in names

    check_source = inspect.getsource(order_service_module.OrderService.check_cancel)
    cancel_source = inspect.getsource(order_service_module.OrderService.cancel_order)
    # 只判定的方法不得写入；执行取消的方法必须经数据层写入。
    assert "update_order_status(" not in check_source
    assert "update_order_status(" in cancel_source
    assert "open(" not in check_source
    assert "open(" not in cancel_source


def test_check_cancel_is_the_single_source_of_cancel_rules():
    """业务规则只写在 check_cancel 里，cancel_order 复用它，避免两份判定漂移。"""
    cancel_source = inspect.getsource(order_service_module.OrderService.cancel_order)
    assert "check_cancel(" in cancel_source
    assert "CANCELLABLE_STATUSES" not in cancel_source
    # 执行取消前必须先判定：判定调用要出现在写库之前。
    assert cancel_source.index("check_cancel(") < cancel_source.index("update_order_status(")
    check_source = inspect.getsource(order_service_module.OrderService.check_cancel)
    assert "CANCELLABLE_STATUSES" in check_source


def test_query_order_path_still_uses_read_only_get_order():
    """架构守卫：本阶段不改动查询链路，get_order 仍只做归一化与读取。"""
    source = inspect.getsource(OrderService.get_order)
    assert "update_order_status" not in source
    assert data_loader.get_order_by_id("ORD-2026-0001") is not None


# 放在文件末尾：本模块按定义顺序最后执行，用于确认前面的写库测试全部只落在副本上。
def test_real_mock_orders_file_was_never_touched_by_the_suite():
    """哨兵：整个套件跑完，真实 data/mock/orders.json 必须逐字节等于收集时的快照。"""
    assert BASELINE_ORDERS_FILE.read_text(encoding="utf-8") == BASELINE_ORDERS_TEXT
