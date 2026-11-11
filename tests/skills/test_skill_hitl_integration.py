"""阶段 3：SkillMiddleware 与 HITL / 多轮 / 工具循环的集成钉测。

复用 tests/test_agent_slice.py 的设施做法（GenericFakeChatModel 子类补 bind_tools、
orders.json tmp 副本、RefundService 冻结时钟），不触碰 data/mock 真实数据，不调真实 LLM。
middleware.py / loader.py 逻辑未改动；本文件只钉既有行为。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import data_loader
from app.agent.graph import TOOLS, create_customer_service_agent
import app.tools.refund_tools as refund_tools
from app.services.refund_service import RefundService
from app.skills.loader import load_skill
from app.skills.middleware import SkillMiddleware

REFUND_NOW = datetime(2026, 9, 22, 12, 0)
REFUND_DELIVERED_AT = datetime(2026, 9, 20, 10, 0)
REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_TOOL_NAMES = [tool.name for tool in TOOLS]
SKILL_TOOL_NAMES = ["query_order", "search_knowledge", "refund_order"]


class ScriptedModel(GenericFakeChatModel):
    """GenericFakeChatModel 未实现 bind_tools，而 create_agent 的模型节点会调用它。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _refund_tool_call(order_id: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "refund_order", "args": {"order_id": order_id}, "id": call_id}],
    )


@pytest.fixture
def orders_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    monkeypatch.setattr(data_loader, "ORDERS_FILE", copy_path)
    monkeypatch.setattr(refund_tools, "_refund_service", RefundService(now=lambda: REFUND_NOW))
    return copy_path


def _make_refundable_order(path: Path) -> str:
    """挑一条已付款订单，改成"已签收 + 冻结时钟有效期内"，返回订单号。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    order_id = next(
        item["order_id"]
        for item in raw
        if item["status"] == "paid" and item["payment_status"] == "paid"
    )
    for item in raw:
        if item["order_id"] == order_id:
            item.update(
                status="delivered",
                payment_status="paid",
                delivered_at=REFUND_DELIVERED_AT.isoformat(),
            )
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return order_id


def _payment_status(path: Path, order_id: str) -> str:
    return next(
        item["payment_status"]
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item["order_id"] == order_id
    )


def _capture_model_requests(monkeypatch: pytest.MonkeyPatch) -> list:
    """记录每次真正到达模型节点的 ModelRequest（即注入/过滤之后的形态）。"""
    captured: list = []
    original = SkillMiddleware.wrap_model_call

    def spy(self, request, handler):
        def inner(req):
            captured.append(req)
            return handler(req)

        return original(self, request, inner)

    monkeypatch.setattr(SkillMiddleware, "wrap_model_call", spy)
    return captured


def _refund_constraint_markers() -> list[str]:
    """从真实 SKILL.md 的 ## Constraints 段取前 3 条红线全文，作为注入服从性标记。

    用整行而不是零散关键词：红线行随 SKILL.md 演进自动更新，
    且整行唯一性足以区分"注入了 skill"与"只有基础 system prompt"。
    """
    skill = load_skill(REPO_ROOT / "skills" / "refund-processing" / "SKILL.md")
    section = skill.body.split("## Constraints", 1)[1]
    lines = [
        line.lstrip("- ").strip()
        for line in section.splitlines()
        if line.strip().startswith("-")
    ]
    assert len(lines) >= 3, "SKILL.md 的 Constraints 红线不足 3 条，注入断言无素材"
    return lines[:3]


def test_active_skill_survives_hitl_resume(orders_file: Path, monkeypatch: pytest.MonkeyPatch):
    """用例 A：退款 HITL 暂停→批准后，恢复轮的模型请求仍带 skill 注入。"""
    captured = _capture_model_requests(monkeypatch)
    order_id = _make_refundable_order(orders_file)
    agent = create_customer_service_agent(
        model=ScriptedModel(messages=iter([
            _refund_tool_call(order_id, "call_1"),
            AIMessage(content="已按工具结果如实反馈本次退款申请。"),
        ])),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-skill-hitl-a"}}

    paused = agent.invoke({"messages": [HumanMessage(f"我要退款，订单号 {order_id}")]}, config)

    assert paused["__interrupt__"][0].value["type"] == refund_tools.REFUND_CONFIRMATION_TYPE
    assert len(captured) == 1

    resumed = agent.invoke(Command(resume={"approved": True}), config)

    assert len(captured) == 2
    resume_request = captured[1]
    assert resume_request.state.get("active_skill") == "refund-processing"
    content = resume_request.system_prompt
    assert "Expert SOP" in content
    for marker in _refund_constraint_markers():
        assert marker in content
    assert [tool.name for tool in resume_request.tools] == SKILL_TOOL_NAMES
    assert resumed["messages"][-1].content == "已按工具结果如实反馈本次退款申请。"
    assert _payment_status(orders_file, order_id) == "refunding"


def test_non_trigger_turn_clears_skill_and_restores_all_tools(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """用例 B：钉清除语义 A —— 未命中触发词的下一人类轮显式清掉 active_skill。"""
    captured = _capture_model_requests(monkeypatch)
    agent = create_customer_service_agent(
        model=ScriptedModel(messages=iter([
            AIMessage(content="好的，已了解您的退款诉求。"),
            AIMessage(content="您的包裹正在派送中。"),
        ])),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-skill-hitl-b"}}

    agent.invoke({"messages": [HumanMessage("我要退款")]}, config)

    assert agent.get_state(config).values["active_skill"] == "refund-processing"
    assert captured[0].state.get("active_skill") == "refund-processing"
    assert [tool.name for tool in captured[0].tools] == SKILL_TOOL_NAMES

    agent.invoke({"messages": [HumanMessage("帮我查一下物流")]}, config)

    assert agent.get_state(config).values["active_skill"] is None
    second = captured[1]
    assert second.state.get("active_skill") is None
    assert [tool.name for tool in second.tools] == ALL_TOOL_NAMES
    for marker in _refund_constraint_markers():
        assert marker not in second.system_prompt


def test_tool_loop_keeps_active_skill_across_tool_messages(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """用例 C：单轮内多步工具循环（query_order → refund_order → resume），每次模型调用都不重新路由、不清 skill。

    规格示例含 search_knowledge，那会触发真实 RAG/嵌入依赖，这里用 query_order → refund_order
    两段工具循环等价覆盖"ToolMessage 轮不重新路由"。
    """
    captured = _capture_model_requests(monkeypatch)
    order_id = _make_refundable_order(orders_file)
    agent = create_customer_service_agent(
        model=ScriptedModel(messages=iter([
            AIMessage(
                content="",
                tool_calls=[{"name": "query_order", "args": {"order_id": order_id}, "id": "call_1"}],
            ),
            _refund_tool_call(order_id, "call_2"),
            AIMessage(content="已按工具结果如实反馈本次退款申请。"),
        ])),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-skill-hitl-c"}}

    paused = agent.invoke({"messages": [HumanMessage(f"我要退款，订单号 {order_id}")]}, config)

    # 第 1 次模型调用由 HumanMessage 路由；第 2 次发生在 query_order 的 ToolMessage 之后
    assert len(captured) == 2
    assert isinstance(captured[1].messages[-1], ToolMessage)
    assert paused["__interrupt__"][0].value["type"] == refund_tools.REFUND_CONFIRMATION_TYPE

    agent.invoke(Command(resume={"approved": True}), config)

    assert len(captured) == 3
    assert [req.state.get("active_skill") for req in captured] == ["refund-processing"] * 3
    assert isinstance(captured[2].messages[-1], ToolMessage)
    assert _payment_status(orders_file, order_id) == "refunding"
