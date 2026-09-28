"""退款垂直切片测试：Service 判定/受理 + Tool 检查后经 interrupt 等人工确认 + 数据层写回。

链路：
- refund_order Tool -> RefundService.check_refund（只判定）-> interrupt 人工确认 -> RefundService.apply_refund
- RefundService.apply_refund -> data_loader.update_order_payment_status -> orders.json（付款状态改 refunding）

时效判定依赖系统时间，所以本文件的测试一律给 Service 注入固定时钟，
不依赖真实"今天"；真实 mock 文件只读不写，写库测试全部落在 tmp_path 副本上。
"""

import ast
import inspect
import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

import app.data_loader as data_loader
from app.config import KNOWLEDGE_DIR, REFUND_WINDOW_DAYS
from app.services import refund_service as refund_service_module
from app.services.order_service import OrderService
from app.services.refund_service import RefundResult, RefundService
from app.tools import order_tools, refund_tools
from app.tools.refund_tools import PAYMENT_STATUS_LABELS, REFUND_CONFIRMATION_TYPE, refund_order

# 在任何打桩之前记下真实 mock 文件与内容，用于末尾的"测试没有污染真实数据"哨兵。
BASELINE_ORDERS_FILE = data_loader.ORDERS_FILE
BASELINE_ORDERS_TEXT = BASELINE_ORDERS_FILE.read_text(encoding="utf-8")

# 固定时钟：所有时效判定都以这一时刻为准，避免测试随日历漂移。
NOW = datetime(2026, 9, 22, 12, 0)

# 不存在的订单号，用于 NOT_FOUND 分支。
UNKNOWN_ORDER_ID = "ORD-NOT-EXIST"


# ---------------------------------------------------------------------------
# 夹具与工具函数
# ---------------------------------------------------------------------------


@pytest.fixture
def orders_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把订单数据文件重定向到 tmp_path 副本，返回副本路径。"""
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    monkeypatch.setattr(data_loader, "ORDERS_FILE", copy_path)
    return copy_path


@pytest.fixture
def forbid_payment_write(monkeypatch: pytest.MonkeyPatch):
    """把付款状态写入口换成抛错版本：任何测试路径试图受理退款都会立刻失败。"""

    def _forbid(*args, **kwargs):
        raise AssertionError("未经人工确认不得写入付款状态")

    monkeypatch.setattr(refund_service_module, "update_order_payment_status", _forbid)


def set_order_fields(path: Path, order_id: str, **fields: object) -> None:
    """直接改副本里的订单字段，用于构造期内／超期／缺签收时间等场景。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    for item in raw:
        if item["order_id"] == order_id:
            item.update(fields)
    path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_raw_orders(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def stored_field(path: Path, order_id: str, field: str) -> object:
    return next(
        item[field] for item in read_raw_orders(path) if item["order_id"] == order_id
    )


def order_id_with(**fields: object) -> str:
    """按当前数据层里的字段值取一条订单，避免把订单号写死在断言里。"""
    matched = [
        order.order_id
        for order in data_loader.load_orders()
        if all(getattr(order, key) == value for key, value in fields.items())
    ]
    assert matched, f"mock 数据中没有满足 {fields} 的订单，测试矩阵需要更新"
    return matched[0]


def make_delivered_order(path: Path, delivered_at: datetime | None) -> str:
    """把一条订单改成"已签收且付款、签收时间＝delivered_at"，返回订单号。"""
    order_id = order_id_with(status="paid")
    set_order_fields(
        path,
        order_id,
        status="delivered",
        payment_status="paid",
        delivered_at=delivered_at.isoformat() if delivered_at else None,
    )
    return order_id


def service_at(moment: datetime = NOW) -> RefundService:
    return RefundService(now=lambda: moment)


def imported_names(module) -> set[str]:
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
# Service 层：时效边界（判定只读，不写库）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "today,expected_outcome,expected_days",
    [
        # 签收当日不计入窗口：签收 09-18 → 截止 09-25。
        (datetime(2026, 9, 18), "CAN_REFUND", 7),
        (datetime(2026, 9, 19), "CAN_REFUND", 6),
        (datetime(2026, 9, 25), "CAN_REFUND", 0),  # 最后一天仍可受理
        (datetime(2026, 9, 26), "REFUND_EXPIRED", -1),  # 次日起第 8 天超期
        (datetime(2026, 10, 30), "REFUND_EXPIRED", -35),
    ],
)
def test_refund_window_boundary_is_counted_from_the_day_after_delivery(
    orders_file: Path, today: datetime, expected_outcome: str, expected_days: int
):
    """时效口径：自签收次日起 REFUND_WINDOW_DAYS 个自然日，截止日当天仍可退。"""
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 18, 14, 15))

    result = service_at(today).check_refund(order_id)

    assert result.outcome == expected_outcome
    assert result.days_remaining == expected_days
    assert result.order is not None and result.order.order_id == order_id


def test_check_refund_never_writes(orders_file: Path, forbid_payment_write):
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    before = orders_file.read_text(encoding="utf-8")

    result = service_at().check_refund(order_id)

    assert result.outcome == "CAN_REFUND"
    assert orders_file.read_text(encoding="utf-8") == before
    assert stored_field(orders_file, order_id, "payment_status") == "paid"


@pytest.mark.parametrize(
    "fields,expected_outcome",
    [
        ({"payment_status": "unpaid"}, "NOT_PAID"),
        ({"payment_status": "refunding"}, "ALREADY_REFUNDING"),
        ({"payment_status": "refunded"}, "ALREADY_REFUNDED"),
    ],
)
def test_payment_status_decides_the_refusal(
    orders_file: Path, forbid_payment_write, fields: dict, expected_outcome: str
):
    """付款状态优先于时效：没付过钱、已在退款中、已退过的都不重复受理。"""
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    set_order_fields(orders_file, order_id, **fields)

    assert service_at().check_refund(order_id).outcome == expected_outcome


def test_cancelled_but_paid_order_can_still_be_refunded(
    orders_file: Path, forbid_payment_write
):
    """取消与退款分离留下的款项缺口：已取消但付过款的订单必须能退。"""
    order_id = order_id_with(status="cancelled", payment_status="paid")

    result = service_at().check_refund(order_id)

    assert result.outcome == "CAN_REFUND"
    assert result.days_remaining is None  # 从未签收，不受签收时效约束


def test_unshipped_paid_order_can_be_refunded_without_window(
    orders_file: Path, forbid_payment_write
):
    order_id = order_id_with(status="paid", payment_status="paid", delivered_at=None)

    result = service_at().check_refund(order_id)

    assert result.outcome == "CAN_REFUND"
    assert result.days_remaining is None


def test_in_transit_order_must_reject_delivery_instead(
    orders_file: Path, forbid_payment_write
):
    """已发货未签收：按政策走拒收，本入口不受理在途退款。"""
    order_id = order_id_with(status="shipped", payment_status="paid", delivered_at=None)

    assert service_at().check_refund(order_id).outcome == "NEED_REJECT_DELIVERY"


def test_delivered_order_without_delivery_time_is_flagged(orders_file: Path, forbid_payment_write):
    """数据异常兜底：状态已签收但没有签收时间，不能凭空判定超期或在期内。"""
    order_id = make_delivered_order(orders_file, None)

    assert service_at().check_refund(order_id).outcome == "NO_DELIVERY_TIME"


@pytest.mark.parametrize("raw_order_id", [UNKNOWN_ORDER_ID, "   ", ""])
def test_unknown_or_blank_order_id_returns_not_found(
    orders_file: Path, forbid_payment_write, raw_order_id: str
):
    before = orders_file.read_text(encoding="utf-8")

    result = service_at().check_refund(raw_order_id)

    assert result.outcome == "NOT_FOUND"
    assert result.order is None
    assert orders_file.read_text(encoding="utf-8") == before


def test_check_refund_normalizes_order_id_case(orders_file: Path, forbid_payment_write):
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))

    result = service_at().check_refund(f"  {order_id.lower()}  ")

    assert result.outcome == "CAN_REFUND"
    assert result.order is not None and result.order.order_id == order_id


# ---------------------------------------------------------------------------
# Service 层：apply_refund 受理（唯一写库路径）
# ---------------------------------------------------------------------------


def test_apply_refund_persists_refunding(orders_file: Path):
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))

    result = service_at().apply_refund(order_id)

    assert isinstance(result, RefundResult)
    assert result.outcome == "REFUNDING_SUBMITTED"
    assert result.order is not None and result.order.payment_status == "refunding"
    assert stored_field(orders_file, order_id, "payment_status") == "refunding"
    # 订单状态不因退款而改变。
    assert stored_field(orders_file, order_id, "status") == "delivered"


def test_apply_refund_is_idempotent(orders_file: Path):
    """受理一次后再受理，走「已在处理中」分支且不重复写。"""
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    service = service_at()

    first = service.apply_refund(order_id)
    second = service.apply_refund(order_id)

    assert first.outcome == "REFUNDING_SUBMITTED"
    assert second.outcome == "ALREADY_REFUNDING"
    assert stored_field(orders_file, order_id, "payment_status") == "refunding"


@pytest.mark.parametrize(
    "prepare,expected_outcome",
    [
        (
            lambda path, oid: set_order_fields(
                path, oid, delivered_at=datetime(2026, 1, 1).isoformat()
            ),
            "REFUND_EXPIRED",
        ),
        (
            lambda path, oid: set_order_fields(path, oid, payment_status="unpaid"),
            "NOT_PAID",
        ),
        (
            lambda path, oid: set_order_fields(
                path, oid, status="shipped", delivered_at=None
            ),
            "NEED_REJECT_DELIVERY",
        ),
        (
            lambda path, oid: set_order_fields(path, oid, delivered_at=None),
            "NO_DELIVERY_TIME",
        ),
    ],
)
def test_apply_refund_refuses_without_touching_data(
    orders_file: Path, prepare, expected_outcome: str
):
    """任何非 CAN_REFUND 的结论都不能产生写入。"""
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    prepare(orders_file, order_id)
    before = orders_file.read_text(encoding="utf-8")

    result = service_at().apply_refund(order_id)

    assert result.outcome == expected_outcome
    assert orders_file.read_text(encoding="utf-8") == before


def test_apply_refund_rechecks_payment_status_before_writing(orders_file: Path):
    """设计要点：受理前 Service 自行按数据层当前状态再判一次，不信任上层早先的 CAN_REFUND。"""
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    service = service_at()

    assert service.check_refund(order_id).outcome == "CAN_REFUND"
    # 判定之后、受理之前，款项被别的流程退掉了（对应等待人工确认的那段时间）。
    set_order_fields(orders_file, order_id, payment_status="refunded")

    result = service.apply_refund(order_id)

    assert result.outcome == "ALREADY_REFUNDED"
    assert stored_field(orders_file, order_id, "payment_status") == "refunded"


# ---------------------------------------------------------------------------
# Tool 层：refund_order 检查 -> interrupt 人工确认 -> 确认通过才受理
#
# interrupt() 只能在 LangGraph 运行上下文里调用，所以 Tool 级测试把
# refund_tools.interrupt 换成记录器；真正的暂停/恢复在 tests/test_agent_slice.py 验证。
# ---------------------------------------------------------------------------


class InterruptRecorder:
    """假 interrupt：记录人工确认 payload，并把预设裁决值当作人工回应返回。"""

    def __init__(self, decision: object = None) -> None:
        self.payloads: list[dict] = []
        self.decision = decision

    def __call__(self, payload: dict) -> object:
        self.payloads.append(payload)
        return self.decision


def _install_interrupt(monkeypatch: pytest.MonkeyPatch, decision: object) -> InterruptRecorder:
    recorder = InterruptRecorder(decision)
    monkeypatch.setattr(refund_tools, "interrupt", recorder)
    return recorder


def _use_frozen_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool 内部用模块级 Service 单例，测试把它换成注入固定时钟的实例。"""
    monkeypatch.setattr(refund_tools, "_refund_service", RefundService(now=lambda: NOW))


def _forbid_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("没有人工批准不得受理退款")

    monkeypatch.setattr(refund_tools._refund_service, "apply_refund", forbidden)


def test_refund_order_tool_is_langchain_tool():
    assert refund_order.name == "refund_order"
    assert "order_id" in refund_order.args
    description = refund_order.description
    assert "退款" in description
    assert "人工确认" in description
    assert "已到账" in description  # docstring 明确禁止宣称到账


@pytest.mark.parametrize(
    "prepare",
    [
        lambda path: make_delivered_order(path, datetime(2026, 9, 20, 10, 0)),
        lambda path: order_id_with(status="cancelled", payment_status="paid"),
    ],
)
def test_refundable_order_reaches_human_confirmation_with_full_payload(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch, prepare
):
    """可受理的订单会进入人工确认：payload 带齐判断依据，此刻付款状态一个字没改。"""
    _use_frozen_service(monkeypatch)
    order_id = prepare(orders_file)
    recorder = _install_interrupt(monkeypatch, {})
    _forbid_submission(monkeypatch)

    output = refund_order.invoke({"order_id": order_id})

    assert len(recorder.payloads) == 1
    payload = recorder.payloads[0]
    assert set(payload) == {
        "type",
        "order_id",
        "product_name",
        "amount",
        "payment_status",
        "delivered_at",
        "days_remaining",
        "question",
    }
    assert payload["type"] == REFUND_CONFIRMATION_TYPE
    assert payload["order_id"] == order_id
    assert isinstance(payload["amount"], float)
    assert payload["payment_status"] == PAYMENT_STATUS_LABELS["paid"]
    assert "确认受理退款" in payload["question"]
    assert "人工未确认" in output and "未受理" in output
    assert stored_field(orders_file, order_id, "payment_status") == "paid"


@pytest.mark.parametrize(
    "prepare,expected_phrase",
    [
        (
            lambda path: make_delivered_order(path, datetime(2026, 9, 1, 9, 0)),
            "已超过该单的退款时效",
        ),
        (
            lambda path: order_id_with(status="shipped", payment_status="paid"),
            "已发货但尚未签收",
        ),
        (
            lambda path: order_id_with(status="pending", payment_status="unpaid"),
            "尚未付款",
        ),
    ],
)
def test_orders_outside_refund_rules_never_reach_human_confirmation(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch, prepare, expected_phrase: str
):
    """超期／在途／未付款：直接给原因，绝不打扰人工，也不写数据。"""
    _use_frozen_service(monkeypatch)

    def never_reached(payload: dict) -> object:
        raise AssertionError("不满足退款条件的订单不应触发人工确认")

    monkeypatch.setattr(refund_tools, "interrupt", never_reached)
    order_id = prepare(orders_file)
    before = stored_field(orders_file, order_id, "payment_status")

    output = refund_order.invoke({"order_id": order_id})

    assert expected_phrase in output
    assert "人工未确认" not in output  # 根本没进人工确认环节
    assert stored_field(orders_file, order_id, "payment_status") == before


def test_unknown_order_never_reaches_human_confirmation(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_frozen_service(monkeypatch)

    def never_reached(payload: dict) -> object:
        raise AssertionError("订单不存在时不应触发人工确认")

    monkeypatch.setattr(refund_tools, "interrupt", never_reached)

    output = refund_order.invoke({"order_id": UNKNOWN_ORDER_ID})

    assert f"未找到订单号为 {UNKNOWN_ORDER_ID} 的订单" in output


@pytest.mark.parametrize(
    "decision",
    [{"approved": False}, {}, "yes", None, {"approved": "true"}, {"approved": 1}],
)
def test_any_non_approving_decision_is_treated_as_rejection(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch, decision: object
):
    """只有 {"approved": True} 算批准，其余一律 fail-closed 按拒绝处理且不改付款状态。"""
    _use_frozen_service(monkeypatch)
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    recorder = _install_interrupt(monkeypatch, decision)
    _forbid_submission(monkeypatch)

    output = refund_order.invoke({"order_id": order_id})

    assert recorder.payloads
    assert output.startswith("人工未确认")
    assert "未受理" in output
    assert stored_field(orders_file, order_id, "payment_status") == "paid"


def test_approved_decision_submits_refund(orders_file: Path, monkeypatch: pytest.MonkeyPatch):
    """人工批准后 Service 才受理：付款状态写为 refunding，文案只说受理不说到账。"""
    _use_frozen_service(monkeypatch)
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    recorder = _install_interrupt(monkeypatch, {"approved": True})

    output = refund_order.invoke({"order_id": order_id})

    assert recorder.payloads[0]["order_id"] == order_id
    assert output.startswith("人工已确认，订单")
    assert "已受理" in output and "退款处理中" in output
    # 工具必须自带"受理不等于到账"的硬约束文案，否则模型会替用户宣布钱已回来。
    assert "不得说款项已到账或退款已完成" in output
    assert stored_field(orders_file, order_id, "payment_status") == "refunding"
    order = OrderService().get_order(order_id)
    assert order is not None and order.payment_status == "refunding"


def test_no_payment_write_before_the_interrupt_fires(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """运行时证明「interrupt 之前不写库」：在暂停那一刻当场检查写入记录为空。"""
    _use_frozen_service(monkeypatch)
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    writes: list[tuple[str, str]] = []

    def recording_update(target_id: str, payment_status: str) -> bool:
        writes.append((target_id, payment_status))
        return True

    def confirm_at_interrupt(payload: dict) -> object:
        assert writes == [], "interrupt 之前已经发生付款状态写入"
        assert stored_field(orders_file, order_id, "payment_status") == "paid"
        return {"approved": True}

    monkeypatch.setattr(
        refund_service_module, "update_order_payment_status", recording_update
    )
    monkeypatch.setattr(refund_tools, "interrupt", confirm_at_interrupt)

    output = refund_order.invoke({"order_id": order_id})

    # 恢复时整个节点从头重放，确认点执行两次；两次都必须还没有写入。
    assert writes == [(order_id, "refunding")]
    assert "已受理" in output


def test_submission_still_reports_truth_if_order_becomes_unrefundable(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """确认期间款项已被别的流程退掉：Tool 必须如实说没受理，而不是宣称受理成功。"""
    _use_frozen_service(monkeypatch)
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))

    def approve_but_already_refunded(payload: dict) -> object:
        set_order_fields(orders_file, order_id, payment_status="refunded")
        return {"approved": True}

    monkeypatch.setattr(refund_tools, "interrupt", approve_but_already_refunded)

    output = refund_order.invoke({"order_id": order_id})

    assert "但退款未受理" in output
    assert "退款申请已受理" not in output
    assert stored_field(orders_file, order_id, "payment_status") == "refunded"


def test_refund_tool_delegates_business_rules_to_service(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """Tool 只做「检查 + 是否中断 + 措辞」：判定经 check_refund，受理经 Service。"""
    _use_frozen_service(monkeypatch)
    order_id = make_delivered_order(orders_file, datetime(2026, 9, 20, 10, 0))
    captured: list[str] = []
    service = RefundService(now=lambda: NOW)

    def fake_check_refund(raw_order_id: str) -> RefundResult:
        captured.append(raw_order_id)
        return service.check_refund(raw_order_id)

    monkeypatch.setattr(refund_tools._refund_service, "check_refund", fake_check_refund)
    _install_interrupt(monkeypatch, {})
    _forbid_submission(monkeypatch)

    output = refund_order.invoke({"order_id": f"  {order_id.lower()}  "})

    assert captured == [f"  {order_id.lower()}  "]
    assert "人工未确认" in output


def test_refund_tool_writes_only_after_the_interrupt_point():
    """架构守卫：恢复时 Tool 节点从头重放，写库必须严格位于 interrupt 之后。"""
    source = inspect.getsource(refund_tools.refund_order.func)
    assert "_refund_service.check_refund(" in source
    assert "interrupt(" in source
    assert source.index("interrupt(") < source.index("_refund_service.apply_refund(")
    assert "update_order_payment_status" not in source  # 写入只经 Service
    assert "REFUND_WINDOW_DAYS" not in source  # 时效数字不下沉到 Tool


# ---------------------------------------------------------------------------
# 数据层
# ---------------------------------------------------------------------------


def test_update_order_payment_status_returns_true_and_persists(orders_file: Path):
    order_id = order_id_with(status="delivered", payment_status="paid")

    assert data_loader.update_order_payment_status(order_id, "refunding") is True
    assert stored_field(orders_file, order_id, "payment_status") == "refunding"


def test_update_order_payment_status_returns_false_for_unknown_order(orders_file: Path):
    before = orders_file.read_text(encoding="utf-8")

    assert data_loader.update_order_payment_status(UNKNOWN_ORDER_ID, "refunding") is False
    assert orders_file.read_text(encoding="utf-8") == before


def test_update_order_payment_status_keeps_other_fields_and_format(orders_file: Path):
    """写回只替换目标订单的 payment_status，其它订单与字段、格式保持原样。"""
    original = read_raw_orders(orders_file)
    order_id = order_id_with(status="delivered", payment_status="paid")

    data_loader.update_order_payment_status(order_id, "refunding")

    updated = read_raw_orders(orders_file)
    assert [item["order_id"] for item in updated] == [item["order_id"] for item in original]
    untouched = [
        (before, after)
        for before, after in zip(original, updated, strict=True)
        if before["order_id"] != order_id
    ]
    assert untouched and all(before == after for before, after in untouched)
    changed = next(item for item in updated if item["order_id"] == order_id)
    original_changed = next(item for item in original if item["order_id"] == order_id)
    assert changed["payment_status"] == "refunding"
    assert {k: v for k, v in changed.items() if k != "payment_status"} == {
        k: v for k, v in original_changed.items() if k != "payment_status"
    }
    text = orders_file.read_text(encoding="utf-8")
    assert text.endswith("]\n")
    assert "无线降噪耳机" in text  # ensure_ascii=False，中文不被转义


# ---------------------------------------------------------------------------
# 分层与口径守卫
# ---------------------------------------------------------------------------


def test_refund_rules_live_in_service_not_tool_layer():
    """架构守卫：付款状态写入只经 data_loader，Service 不碰文件，Tool 不引用数据层。"""
    names = imported_names(refund_service_module)
    assert "update_order_payment_status" in names
    assert "json" not in names

    check_source = inspect.getsource(RefundService.check_refund)
    apply_source = inspect.getsource(RefundService.apply_refund)
    assert "update_order_payment_status(" not in check_source
    assert "update_order_payment_status(" in apply_source
    assert "open(" not in check_source
    assert "open(" not in apply_source
    # 受理前必须先判定。
    assert apply_source.index("check_refund(") < apply_source.index("update_order_payment_status(")

    tool_names = imported_names(refund_tools)
    assert "data_loader" not in tool_names
    assert "json" not in tool_names


def test_check_refund_is_the_single_source_of_refund_rules():
    """退款规则只写在 check_refund 里，apply_refund 复用它，避免两份判定漂移。"""
    apply_source = inspect.getsource(RefundService.apply_refund)
    assert "check_refund(" in apply_source
    assert "REFUND_WINDOW_DAYS" not in apply_source
    check_source = inspect.getsource(RefundService.check_refund)
    assert "REFUND_WINDOW_DAYS" in check_source


def test_refund_window_matches_the_policy_document():
    """防漂移守卫：代码里的时效天数必须与知识库政策文档写的口径一致。

    两处不一致时，Agent 会一手按工具判定拒绝、一手念政策承诺，正是最伤信任的自相矛盾。
    """
    policy_text = (KNOWLEDGE_DIR / "refund_policy.md").read_text(encoding="utf-8")
    assert f"次日起 {REFUND_WINDOW_DAYS} 个自然日" in policy_text, (
        "refund_policy.md 已不再承诺该时效，须同步修改 REFUND_WINDOW_DAYS 或政策文档"
    )


def test_cancel_execution_hints_the_pending_refund(orders_file: Path, monkeypatch: pytest.MonkeyPatch):
    """取消与退款分离的体验兜底：取消已付款订单必须被告知款项还没退。"""
    order_id = order_id_with(status="paid", payment_status="paid")
    # 走的是取消工具：它的 interrupt 在 order_tools，与退款工具各用各的模块。
    monkeypatch.setattr(order_tools, "interrupt", InterruptRecorder({"approved": True}))

    output = order_tools.cancel_order.invoke({"order_id": order_id})

    assert "已取消" in output
    assert "取消订单本身不会退款" in output
    assert "refund_order" in output


# 放在文件末尾：本模块按定义顺序最后执行，用于确认前面的写库测试全部只落在副本上。
def test_real_mock_orders_file_was_never_touched_by_the_suite():
    """哨兵：整个套件跑完，真实 data/mock/orders.json 必须逐字节等于收集时的快照。"""
    assert BASELINE_ORDERS_FILE.read_text(encoding="utf-8") == BASELINE_ORDERS_TEXT
