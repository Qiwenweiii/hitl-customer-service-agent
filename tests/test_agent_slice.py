"""Agent 层垂直切片测试：用户问题 -> LLM 决策 -> query_order / query_logistics / search_knowledge / cancel_order -> 回答。

不依赖真实 LLM：用 GenericFakeChatModel 按脚本回放模型响应，
验证 create_agent 构建的图的工具调用闭环与终止行为。
知识库工具用替身服务顶掉 RAGService，避免测试加载 bge 模型。
取消订单的人工确认走真实 LangGraph interrupt：文件末尾那组测试注入 InMemorySaver，
用 Command(resume=...) 验证「暂停 → 人工裁决 → 恢复」的完整闭环。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import app.data_loader as data_loader
from app.agent import create_customer_service_agent
from app.agent.graph import SYSTEM_PROMPT, TOOLS
from app.services.order_service import OrderService
from app.services.refund_service import RefundService
from app.tools import order_tools, rag_tools, refund_tools


class ScriptedFakeModel(GenericFakeChatModel):
    """GenericFakeChatModel 未实现 bind_tools，而 create_agent 在模型节点会调用它。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_model(*responses: AIMessage) -> ScriptedFakeModel:
    return ScriptedFakeModel(messages=iter(responses))


def _tool_messages(result: dict) -> list[ToolMessage]:
    return [message for message in result["messages"] if isinstance(message, ToolMessage)]


class StubKnowledgeService:
    """RAGService 替身：记录 query 并返回固定政策片段，避免加载 bge 模型。"""

    def __init__(self, context: str) -> None:
        self.context = context
        self.queries: list[str] = []

    def search(self, query: str) -> str:
        self.queries.append(query)
        return self.context


def _run_policy_question(monkeypatch, question: str, tool_query: str, context: str, answer: str):
    """跑一轮"模型决定调用 search_knowledge"的 Agent 闭环，返回结果与替身服务。"""
    stub = StubKnowledgeService(context)
    monkeypatch.setattr(rag_tools, "_rag_service", stub)
    fake = _fake_model(
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge", "args": {"query": tool_query}, "id": "call_1"}],
        ),
        AIMessage(content=answer),
    )
    agent = create_customer_service_agent(model=fake)

    return agent.invoke({"messages": [HumanMessage(question)]}), stub


def test_agent_calls_query_order_and_returns_final_answer():
    fake = _fake_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "ORD-2026-0001"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="已为您查到订单 ORD-2026-0001 的信息，请问还有什么可以帮您？"),
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0001")]})

    tool_messages = _tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "query_order"
    assert "无线降噪耳机" in tool_messages[0].content
    assert "1299.00" in tool_messages[0].content

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert not final.tool_calls
    assert "已为您查到订单 ORD-2026-0001" in final.content


def test_agent_returns_answer_without_tool_call_when_no_order_id():
    fake = _fake_model(
        AIMessage(content="请问您要查询哪个订单？请提供订单号，形如 ORD-2026-0001。")
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke({"messages": [HumanMessage("我的订单怎么样了？")]})

    assert _tool_messages(result) == []
    assert "订单号" in result["messages"][-1].content


def test_agent_factory_requires_api_key(monkeypatch):
    monkeypatch.setattr("app.agent.graph.LLM_API_KEY", "")
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        create_customer_service_agent()


def test_agent_binds_five_tools():
    assert [tool.name for tool in TOOLS] == [
        "query_order",
        "query_logistics",
        "search_knowledge",
        "cancel_order",
        "refund_order",
    ]


def test_system_prompt_declares_five_capability_groups():
    for phrase in ("订单数据", "物流数据", "企业知识", "订单取消", "退款申请"):
        assert phrase in SYSTEM_PROMPT, f"system_prompt 缺少「{phrase}」这一类能力说明"


def test_system_prompt_covers_every_bound_tool():
    """架构守卫：每个已绑定工具都必须在 system_prompt 里有明确的使用边界。"""
    for tool in TOOLS:
        assert tool.name in SYSTEM_PROMPT


def test_agent_calls_query_logistics_for_shipping_question():
    fake = _fake_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_logistics", "args": {"order_id": "ORD-2026-0004"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="订单 ORD-2026-0004 的快递已签收，签收网点为北京市朝阳区望京配送站。"),
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke({"messages": [HumanMessage("订单 ORD-2026-0004 到哪了？")]})

    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["query_logistics"]
    assert "SF97530000004" in tool_messages[0].content
    assert "已签收" in tool_messages[0].content
    assert "已签收" in result["messages"][-1].content


def test_agent_calls_both_tools_when_question_covers_order_and_logistics():
    fake = _fake_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "ORD-2026-0003"}, "id": "call_1"},
                {
                    "name": "query_logistics",
                    "args": {"order_id": "ORD-2026-0003"},
                    "id": "call_2",
                },
            ],
        ),
        AIMessage(content="订单已付款，包裹正在南京转运中心运输中。"),
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke(
        {"messages": [HumanMessage("订单 ORD-2026-0003 付款了吗，现在到哪了？")]}
    )

    assert sorted(message.name for message in _tool_messages(result)) == [
        "query_logistics",
        "query_order",
    ]


def test_agent_asks_order_id_for_logistics_question_without_order_id():
    fake = _fake_model(
        AIMessage(content="请提供您的订单号，形如 ORD-2026-0001，我来帮您查询物流进展。")
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke({"messages": [HumanMessage("我的快递到哪了？")]})

    assert _tool_messages(result) == []
    assert "订单号" in result["messages"][-1].content


def test_agent_calls_search_knowledge_for_refund_policy_question(monkeypatch):
    result, stub = _run_policy_question(
        monkeypatch,
        question="退款到账要多久？",
        tool_query="退款到账要多久",
        context="平台收到退回商品并验收合格后，1-3 个工作日内原路退回。",
        answer="退款一般在 1-3 个工作日内到账。",
    )

    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["search_knowledge"]
    assert stub.queries == ["退款到账要多久"]
    assert "以下是知识库中与问题相关的内容" in tool_messages[0].content
    assert "1-3 个工作日内原路退回" in tool_messages[0].content
    assert "1-3 个工作日" in result["messages"][-1].content


def test_agent_calls_search_knowledge_for_membership_question(monkeypatch):
    result, stub = _run_policy_question(
        monkeypatch,
        question="会员有哪些权益？",
        tool_query="会员权益",
        context="金卡会员享受全年 12 次免费上门取件与专属客服通道。",
        answer="金卡会员有免费上门取件和专属客服通道。",
    )

    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["search_knowledge"]
    assert stub.queries == ["会员权益"]
    assert "金卡会员" in tool_messages[0].content


def test_search_knowledge_miss_reaches_agent_as_no_result(monkeypatch):
    """知识库未命中时，回填给模型的必须是"未检索到"，不能是硬凑的无关政策。"""
    result, stub = _run_policy_question(
        monkeypatch,
        question="你们停车场免费吗？",
        tool_query="停车收费政策",
        context="",
        answer="抱歉，知识库中暂时没有相关规定，我帮您转人工客服确认。",
    )

    tool_messages = _tool_messages(result)
    assert stub.queries == ["停车收费政策"]
    assert "未检索到" in tool_messages[0].content
    assert "以下是知识库中与问题相关的内容" not in tool_messages[0].content


def test_agent_calls_query_order_for_order_amount_question():
    fake = _fake_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "ORD-2026-0001"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="订单 ORD-2026-0001 金额为 1299.00 元。"),
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke({"messages": [HumanMessage("订单 ORD-2026-0001 多少钱？")]})

    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["query_order"]
    assert "1299.00" in tool_messages[0].content
    assert "1299.00" in result["messages"][-1].content


def test_agent_calls_query_logistics_for_tracking_question():
    fake = _fake_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_logistics", "args": {"order_id": "ORD-2026-0003"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="包裹正在南京转运中心运输中。"),
    )
    agent = create_customer_service_agent(model=fake)

    result = agent.invoke({"messages": [HumanMessage("ORD-2026-0003 到哪里了？")]})

    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["query_logistics"]
    assert "南京转运中心" in tool_messages[0].content
    assert "JD20260000003" in tool_messages[0].content


def test_system_prompt_keeps_policy_questions_off_the_order_id_gate():
    """路由条款守卫：这三条写错就会直接退化成"政策问题也追问订单号"。"""
    assert "政策与规则类问题不需要订单号" in SYSTEM_PROMPT
    assert "不得用常识或模型自身知识自行补充规则" in SYSTEM_PROMPT
    assert "不要把订单实时信息和企业政策混为一谈" in SYSTEM_PROMPT


def test_system_prompt_keeps_cancel_authority_with_the_human_reviewer():
    """路由条款守卫：取消结论只能来自工具返回值，模型既不能代批也不能自己宣布取消成功。"""
    assert "必须调用 cancel_order 受理" in SYSTEM_PROMPT
    assert "不要调用 cancel_order" in SYSTEM_PROMPT
    assert "不得代替人工作出批准" in SYSTEM_PROMPT
    assert '明确返回"已取消""状态已更新"' in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Agent 级 HITL 闭环：真实 interrupt + InMemorySaver + Command(resume=...)
#
# 人工批准会走真实写库路径，因此这组测试同样把 data_loader.ORDERS_FILE 指向
# tmp_path 副本，绝不写 data/mock/orders.json。
# ---------------------------------------------------------------------------


@pytest.fixture
def orders_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    monkeypatch.setattr(data_loader, "ORDERS_FILE", copy_path)
    return copy_path


def _order_id_with_status(status: str) -> str:
    matched = [order.order_id for order in data_loader.load_orders() if order.status == status]
    assert matched, f"mock 数据中没有状态为 {status} 的订单，测试矩阵需要更新"
    return matched[0]


def _stored_status(path: Path, order_id: str) -> str:
    return next(
        item["status"]
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item["order_id"] == order_id
    )


def _invoke_cancel_request(order_id: str):
    """让 Agent 处理一句取消请求，返回 agent / config / invoke 结果（可取消时会停在暂停点）。"""
    agent = create_customer_service_agent(
        model=_fake_model(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "cancel_order", "args": {"order_id": order_id}, "id": "call_1"}
                ],
            ),
            AIMessage(content="已按工具结果如实反馈本次取消申请。"),
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "hitl-cancel-thread"}}

    return agent, config, agent.invoke(
        {"messages": [HumanMessage(f"帮我取消订单 {order_id}")]}, config
    )


@pytest.mark.parametrize("status", ["pending", "paid"])
def test_agent_pauses_for_human_confirmation_before_writing(orders_file: Path, status: str):
    """可取消的订单必须先暂停：payload 带齐确认信息，图停在 tools 节点，数据一个字没改。"""
    order_id = _order_id_with_status(status)

    agent, config, paused = _invoke_cancel_request(order_id)

    interrupts = paused["__interrupt__"]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert set(payload) == {
        "type",
        "order_id",
        "product_name",
        "amount",
        "current_status",
        "question",
    }
    assert payload["type"] == order_tools.CANCEL_CONFIRMATION_TYPE
    assert payload["order_id"] == order_id
    assert isinstance(payload["amount"], float)
    assert payload["current_status"] == order_tools.ORDER_STATUS_LABELS[status]
    assert "确认执行取消" in payload["question"]

    assert agent.get_state(config).next == ("tools",)
    assert _tool_messages(paused) == []
    assert _stored_status(orders_file, order_id) == status


def test_agent_cancels_order_after_human_approval(orders_file: Path):
    """批准链路：同一个 thread_id 恢复 → ToolMessage 表示已取消 → 重读订单为 cancelled。"""
    order_id = _order_id_with_status("paid")
    agent, config, _ = _invoke_cancel_request(order_id)

    resumed = agent.invoke(Command(resume={"approved": True}), config)

    tool_messages = _tool_messages(resumed)
    assert [message.name for message in tool_messages] == ["cancel_order"]
    assert "已取消" in tool_messages[0].content
    assert "状态已更新" in tool_messages[0].content
    assert _stored_status(orders_file, order_id) == "cancelled"
    order = data_loader.get_order_by_id(order_id)
    assert order is not None and order.status == "cancelled"
    assert resumed["messages"][-1].content == "已按工具结果如实反馈本次取消申请。"
    assert not agent.get_state(config).next  # 图已跑完，不再挂起


def test_agent_keeps_order_unchanged_when_human_rejects(orders_file: Path):
    order_id = _order_id_with_status("pending")
    agent, config, _ = _invoke_cancel_request(order_id)

    resumed = agent.invoke(Command(resume={"approved": False}), config)

    tool_content = _tool_messages(resumed)[0].content
    assert "人工未确认" in tool_content
    assert "状态已更新" not in tool_content
    assert _stored_status(orders_file, order_id) == "pending"
    assert not agent.get_state(config).next


@pytest.mark.parametrize(
    "decision", ["yes", "", {"approved": 1}, {"approved": "true"}, {"approve": True}]
)
def test_agent_treats_any_non_approving_resume_as_rejection(
    orders_file: Path, decision: object
):
    """fail-closed：送达工具的裁决值只有 {"approved": True} 算批准。"""
    order_id = _order_id_with_status("pending")
    agent, config, _ = _invoke_cancel_request(order_id)

    resumed = agent.invoke(Command(resume=decision), config)

    tool_content = _tool_messages(resumed)[0].content
    assert "人工未确认" in tool_content and "未取消" in tool_content
    assert _stored_status(orders_file, order_id) == "pending"


def test_empty_dict_resume_never_leaves_the_pause_point(orders_file: Path):
    """当前 LangGraph 版本里 ``Command(resume={})`` 被当成空的「按 interrupt id 恢复」映射，
    图不会继续执行；``resume=None`` 更在库内部就报错。两者都到不了 Tool，
    因此这里只断言更关键的不变量：仍然暂停、订单没被写坏
    （裁决值本身的 fail-closed 解析在 tests/test_cancel_order_slice.py 断言）。
    """
    order_id = _order_id_with_status("pending")
    agent, config, _ = _invoke_cancel_request(order_id)

    resumed = agent.invoke(Command(resume={}), config)

    assert _tool_messages(resumed) == []
    assert agent.get_state(config).next == ("tools",)
    assert _stored_status(orders_file, order_id) == "pending"


def test_resume_replays_the_read_only_cancellation_check(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """恢复时 Tool 节点从头重放：检查会再执行一次，所以写库只能留在 interrupt 之后。"""
    order_id = _order_id_with_status("pending")
    calls: list[str] = []
    real_check_cancel = order_tools._order_service.check_cancel

    def counting_check_cancel(raw_order_id: str):
        calls.append(raw_order_id)
        return real_check_cancel(raw_order_id)

    monkeypatch.setattr(order_tools._order_service, "check_cancel", counting_check_cancel)
    agent, config, _ = _invoke_cancel_request(order_id)
    assert calls == [order_id]

    agent.invoke(Command(resume={"approved": False}), config)

    assert calls == [order_id, order_id]


@pytest.mark.parametrize(
    "status,expected_reason",
    [
        ("shipped", "不支持直接取消"),
        ("delivered", "不支持直接取消"),
        ("cancelled", "此前已经取消"),
    ],
)
def test_agent_never_pauses_for_orders_that_cannot_be_cancelled(
    orders_file: Path, status: str, expected_reason: str
):
    """不该打扰人工的订单：直接拿到原因并结束，不产生中断、不写数据。"""
    order_id = _order_id_with_status(status)

    agent, config, result = _invoke_cancel_request(order_id)

    assert "__interrupt__" not in result
    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["cancel_order"]
    assert expected_reason in tool_messages[0].content
    assert _stored_status(orders_file, order_id) == status
    assert not agent.get_state(config).next


def test_agent_without_checkpointer_dead_ends_at_the_confirmation(orders_file: Path):
    """审计结论：漏传 checkpointer 时图照样抛出中断，但这一轮永远恢复不了。
    危险的不是报错，而是报错前就把订单改掉了 —— 这里钉住「不会写坏」这一半。
    """
    order_id = _order_id_with_status("pending")
    agent = create_customer_service_agent(
        model=_fake_model(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "cancel_order", "args": {"order_id": order_id}, "id": "call_1"}
                ],
            )
        )
    )

    paused = agent.invoke({"messages": [HumanMessage(f"帮我取消订单 {order_id}")]})

    assert paused["__interrupt__"][0].value["order_id"] == order_id
    assert _stored_status(orders_file, order_id) == "pending"
    with pytest.raises(RuntimeError, match="checkpointer"):
        agent.invoke(Command(resume={"approved": True}))
    assert _stored_status(orders_file, order_id) == "pending"


def test_resuming_needs_the_same_thread_id(orders_file: Path):
    """有 checkpointer 但不带 thread_id：LangGraph 直接报错，不会把裁决送到别的线程上。"""
    order_id = _order_id_with_status("pending")
    agent = create_customer_service_agent(
        model=_fake_model(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "cancel_order", "args": {"order_id": order_id}, "id": "call_1"}
                ],
            )
        ),
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="thread_id"):
        agent.invoke({"messages": [HumanMessage(f"帮我取消订单 {order_id}")]})

    assert _stored_status(orders_file, order_id) == "pending"


# ---------------------------------------------------------------------------
# Agent 级退款 HITL 闭环
#
# 退款时效依赖系统时间，这里把 Tool 层的 Service 换成注入固定时钟的实例，
# 并把订单的签收时间设在该时钟的有效期内，测试才不会随日历漂移。
# ---------------------------------------------------------------------------

REFUND_NOW = datetime(2026, 9, 22, 12, 0)
REFUND_DELIVERED_AT = datetime(2026, 9, 20, 10, 0)


def _stored_field(path: Path, order_id: str, field: str) -> object:
    return next(
        item[field]
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item["order_id"] == order_id
    )


def _make_refundable_order(path: Path, delivered_at: datetime | None) -> str:
    """挑一条已付款订单，改成"已签收 + 指定签收时间"，返回订单号。"""
    order_id = next(
        order.order_id
        for order in data_loader.load_orders()
        if order.status == "paid" and order.payment_status == "paid"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    for item in raw:
        if item["order_id"] == order_id:
            item.update(
                status="delivered",
                payment_status="paid",
                delivered_at=delivered_at.isoformat() if delivered_at else None,
            )
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return order_id


def _use_frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        refund_tools, "_refund_service", RefundService(now=lambda: REFUND_NOW)
    )


def _invoke_refund_request(order_id: str):
    """让 Agent 处理一句退款请求，返回 agent / config / invoke 结果。"""
    agent = create_customer_service_agent(
        model=_fake_model(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "refund_order", "args": {"order_id": order_id}, "id": "call_1"}
                ],
            ),
            AIMessage(content="已按工具结果如实反馈本次退款申请。"),
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "hitl-refund-thread"}}

    return agent, config, agent.invoke(
        {"messages": [HumanMessage(f"我要退款，订单号 {order_id}")]}, config
    )


def test_agent_pauses_for_refund_confirmation_before_submitting(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_frozen_clock(monkeypatch)
    order_id = _make_refundable_order(orders_file, REFUND_DELIVERED_AT)

    agent, config, paused = _invoke_refund_request(order_id)

    payload = paused["__interrupt__"][0].value
    assert payload["type"] == refund_tools.REFUND_CONFIRMATION_TYPE
    assert payload["order_id"] == order_id
    assert payload["payment_status"] == refund_tools.PAYMENT_STATUS_LABELS["paid"]
    assert payload["delivered_at"].startswith("2026-09-20")
    assert payload["days_remaining"] == 5

    assert agent.get_state(config).next == ("tools",)
    assert _tool_messages(paused) == []
    assert _stored_field(orders_file, order_id, "payment_status") == "paid"


def test_agent_submits_refund_after_human_approval(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_frozen_clock(monkeypatch)
    order_id = _make_refundable_order(orders_file, REFUND_DELIVERED_AT)
    agent, config, _ = _invoke_refund_request(order_id)

    resumed = agent.invoke(Command(resume={"approved": True}), config)

    tool_content = _tool_messages(resumed)[0].content
    assert tool_content.startswith("人工已确认，订单")
    assert "已受理" in tool_content and "退款处理中" in tool_content
    assert "不得说款项已到账或退款已完成" in tool_content
    assert _stored_field(orders_file, order_id, "payment_status") == "refunding"
    order = OrderService().get_order(order_id)
    assert order is not None and order.payment_status == "refunding"


def test_agent_keeps_payment_status_when_human_rejects_refund(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_frozen_clock(monkeypatch)
    order_id = _make_refundable_order(orders_file, REFUND_DELIVERED_AT)
    agent, config, _ = _invoke_refund_request(order_id)

    resumed = agent.invoke(Command(resume={"approved": False}), config)

    assert "人工未确认" in _tool_messages(resumed)[0].content
    assert _stored_field(orders_file, order_id, "payment_status") == "paid"


def test_agent_never_pauses_for_expired_refund_request(
    orders_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """超期订单在 Agent 层也该直接拿到原因，不产生挂起的确认。"""
    _use_frozen_clock(monkeypatch)
    order_id = _make_refundable_order(orders_file, datetime(2026, 3, 1, 9, 0))

    agent, config, result = _invoke_refund_request(order_id)

    assert "__interrupt__" not in result
    tool_messages = _tool_messages(result)
    assert [message.name for message in tool_messages] == ["refund_order"]
    assert "已超过该单的退款时效" in tool_messages[0].content
    assert _stored_field(orders_file, order_id, "payment_status") == "paid"
    assert not agent.get_state(config).next


def test_system_prompt_keeps_refund_outcome_and_money_claims_ahead_of_the_tool():
    """路由条款守卫：退款结论只能来自工具，且「受理」永远不等于「到账」。"""
    assert "必须调用 refund_order 受理" in SYSTEM_PROMPT
    assert "不要调用 refund_order" in SYSTEM_PROMPT
    assert "不得声称退款已受理" in SYSTEM_PROMPT
    assert "受理不等于钱已到账" in SYSTEM_PROMPT
    assert "不得自行编造到账时间" in SYSTEM_PROMPT
    assert "取消与退款是两个动作" in SYSTEM_PROMPT
