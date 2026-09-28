"""短期 Conversation Memory 切片：同一 thread_id 跨轮继承上下文，不同 thread_id 互不可见。

不调用真实 LLM。这里用的是「回显式假模型」：它只根据自己**实际收到的消息列表**决定下一步
（本轮消息里有没有订单号、上文里出现过哪些订单号），并把每次收到的消息原样记下来。
因此这些断言证明的是「上下文确实跨轮送达了模型」这条链路，不是模型的理解能力；
真实模型指代得对不对由 eval/ 负责。

两个刻意的设计：
- 假模型解析订单号时排除 SystemMessage：``ORD-2026-0001`` 在系统提示里作为示例出现多次，
  只有把它排除在外，第二轮拿到正确订单号才能证明信息来自对话历史。
- 上文的订单号只用于选工具与传参，实时数据一律重新调用工具取，
  因此测试能区分「沿用上一轮的工具结果」和「本轮重新查询」。

写入类测试同样把 data_loader.ORDERS_FILE 指向 tmp_path 副本，绝不改真实订单数据。
"""

import inspect
import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

import app.agent.graph as graph_module
import app.data_loader as data_loader
from app.agent import create_customer_service_agent
from app.agent.graph import SYSTEM_PROMPT
from app.services.order_service import OrderService

ORDER_ID_PATTERN = re.compile(r"ORD-\d{4}-\d{4}")
ASK_ORDER_ID = "请提供您的订单号，形如 ORD-2026-0001。"
# 本轮没写订单号、但明显在接着上文追问的说法
REFERENCE_WORDS = ("它", "这单", "这个订单", "那一单", "刚才")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _history_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """上文 = 最后一条 HumanMessage（本轮提问）之前、且不含系统提示的消息。"""
    current = max(index for index, message in enumerate(messages) if message.type == "human")
    return [message for message in messages[:current] if message.type != "system"]


def _order_ids_in_history(messages: Sequence[BaseMessage]) -> list[str]:
    parts: list[str] = []
    for message in _history_messages(messages):
        parts.append(str(message.content))
        parts.extend(str(call["args"]) for call in getattr(message, "tool_calls", []))
    return ORDER_ID_PATTERN.findall("\n".join(parts))


def _turn_question(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return str(message.content)
    raise AssertionError("模型输入里必须有一条用户提问")


def _tool_for(question: str) -> str:
    if "取消" in question:
        return "cancel_order"
    if "退款" in question:
        return "refund_order"
    if any(word in question for word in ("到哪", "物流", "发货", "签收", "运单", "快递")):
        return "query_logistics"
    return "query_order"


class EchoModel(BaseChatModel):
    """按「本轮提问 + 上文」决定调哪个工具的假模型，并记录每次收到的消息列表。

    它实现的正是系统提示「三、多轮对话与指代」写的口径：本轮有订单号用本轮的；
    没有订单号但存在指代、且上文只指向唯一订单时沿用它；两者都拿不到就索要订单号、不调工具。
    """

    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "conversation-memory-echo"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "EchoModel":
        return self

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        if messages[-1].type == "tool":
            return AIMessage(content="已按本轮工具结果如实回答。")
        question = _turn_question(messages)
        order_id = None
        if in_question := ORDER_ID_PATTERN.findall(question):
            order_id = in_question[0]
        elif any(word in question for word in REFERENCE_WORDS):
            historic = _order_ids_in_history(messages)
            if len(set(historic)) == 1:  # 上文指代不明确时不允许猜，与系统提示同口径
                order_id = historic[0]
        if order_id is None:
            return AIMessage(content=ASK_ORDER_ID)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": _tool_for(question),
                    "args": {"order_id": order_id},
                    "id": f"call_{len(self.calls)}",
                }
            ],
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])


def _tool_messages(result: dict) -> list[ToolMessage]:
    return [message for message in result["messages"] if isinstance(message, ToolMessage)]


def _tool_calls(result: dict) -> list[dict]:
    return [
        {"name": call["name"], "args": call["args"]}
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _current_turn(result: dict) -> list[BaseMessage]:
    """本轮新增的消息 = 最后一条用户提问起（跨轮线程里 result["messages"] 含历史，不能整体判空）。"""
    start = max(index for index, message in enumerate(result["messages"]) if message.type == "human")
    return result["messages"][start:]


def _asked_for_order_id(result: dict) -> bool:
    """本轮没有任何工具调用、只回了一句要订单号 = 模型没能从上文确定用户说的是哪一笔。"""
    turn = _current_turn(result)
    asked = turn[-1].content == ASK_ORDER_ID
    no_calls = [
        call for message in turn if isinstance(message, AIMessage) for call in message.tool_calls
    ] == []
    return asked and no_calls


@pytest.fixture
def orders_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    monkeypatch.setattr(data_loader, "ORDERS_FILE", copy_path)
    return copy_path


def _order_id_with_status(status: str) -> str:
    matched = [order.order_id for order in data_loader.load_orders() if order.status == status]
    assert len(matched) == 1, f"需要恰好一笔 {status} 订单，实际 {matched}"
    return matched[0]


def _stored_status(path: Path, order_id: str) -> str:
    return next(
        item["status"]
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item["order_id"] == order_id
    )


def _agent() -> tuple[EchoModel, Any]:
    """一个带独立 checkpointer 的 Agent 与它的假模型；config 由调用方按线程自取。"""
    model = EchoModel()
    return model, create_customer_service_agent(model=model, checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 跨轮继承：同一个 thread_id
# ---------------------------------------------------------------------------


def test_second_turn_input_contains_the_previous_turn():
    """记忆链路本身：第二轮模型收到的消息里必须有第一轮的提问与答复，而不是只有本轮提问。"""
    model, agent = _agent()
    config = _config("memory-thread")

    agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)
    # 本轮答什么是假模型的事，这条测试只关心历史消息有没有送到
    agent.invoke({"messages": [HumanMessage("谢谢，没别的事了")]}, config)

    # calls[2] = 第二轮的第一次模型调用（前两次属于第一轮：决定调工具 + 看结果作答）
    second_turn_call = model.calls[2]
    contents = [str(message.content) for message in second_turn_call]
    assert "帮我查一下订单 ORD-2026-0004" in contents
    assert "已按本轮工具结果如实回答。" in contents
    assert str(second_turn_call[-1].content) == "谢谢，没别的事了"
    # 系统提示始终只有一份，历史按时间顺序累积在它和本轮提问之间
    assert [message.type for message in second_turn_call] == [
        "system",
        "human",
        "ai",
        "tool",
        "ai",
        "human",
    ]


def test_followup_resolves_order_id_from_previous_turn():
    """验收点 2：本轮消息不含任何订单号，工具却拿到了第一轮那个订单号。"""
    model, agent = _agent()
    config = _config("memory-thread")
    question = "那它现在到哪了？"
    assert not ORDER_ID_PATTERN.search(question)

    agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)
    result = agent.invoke({"messages": [HumanMessage(question)]}, config)

    assert _tool_calls(result)[-1] == {
        "name": "query_logistics",
        "args": {"order_id": "ORD-2026-0004"},
    }
    assert "SF97530000004" in _tool_messages(result)[-1].content
    # 第二轮只需一次工具往返：决定调用 → 拿到结果 → 作答
    assert len(model.calls) == 4


# ---------------------------------------------------------------------------
# 隔离：不同 thread_id / 没有 checkpointer
# ---------------------------------------------------------------------------


def test_other_thread_does_not_inherit_the_first_thread():
    """验收点 3：换一个 thread_id 问同一句指代，模型既拿不到上文订单号、也不会调用工具。"""
    model, agent = _agent()
    config_a = _config("thread-a")
    agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config_a)

    calls_before = len(model.calls)
    result_b = agent.invoke({"messages": [HumanMessage("那它现在到哪了？")]}, _config("thread-b"))

    assert _asked_for_order_id(result_b)
    # B 线程的模型输入里没有任何来自 A 线程的痕迹
    assert _order_ids_in_history(model.calls[calls_before]) == []
    assert [message.type for message in model.calls[calls_before]] == ["system", "human"]
    assert len(agent.get_state(config_a).values["messages"]) == 4
    assert len(agent.get_state(_config("thread-b")).values["messages"]) == 2


def test_without_checkpointer_each_turn_starts_fresh():
    """不注入 checkpointer 不报错，但记忆和 HITL 恢复一起失效：钉住「静默失忆」这个坑。"""
    model = EchoModel()
    agent = create_customer_service_agent(model=model)
    config = _config("no-checkpointer")

    agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)
    result = agent.invoke({"messages": [HumanMessage("那它现在到哪了？")]}, config)

    assert _asked_for_order_id(result)
    assert _order_ids_in_history(model.calls[-1]) == []


def test_ambiguous_history_is_not_resolved_by_guessing():
    """上文出现过两笔订单时，指代不能被随便一个订单号满足 —— 必须回头问用户。"""
    model, agent = _agent()
    config = _config("ambiguous")
    agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)
    agent.invoke({"messages": [HumanMessage("再看看 ORD-2026-0003")]}, config)

    result = agent.invoke({"messages": [HumanMessage("那它到哪了？")]}, config)

    assert len(set(_order_ids_in_history(model.calls[-1]))) == 2
    assert _asked_for_order_id(result)


# ---------------------------------------------------------------------------
# 记忆不得削弱既有边界：HITL 与「以本轮工具结果为准」
# ---------------------------------------------------------------------------


def test_hitl_still_pauses_and_resumes_after_an_earlier_turn(orders_file: Path):
    """同一个线程先查后取消：跨轮上下文不能干扰中断点，裁决仍按 fail-closed 契约走。"""
    order_id = _order_id_with_status("pending")
    _, agent = _agent()
    config = _config("cancel-after-context")

    agent.invoke({"messages": [HumanMessage(f"帮我查一下订单 {order_id}")]}, config)
    paused = agent.invoke({"messages": [HumanMessage("帮我取消它")]}, config)

    assert paused["__interrupt__"][0].value["order_id"] == order_id
    assert _stored_status(orders_file, order_id) == "pending"

    resumed = agent.invoke(Command(resume={"approved": True}), config)

    assert "已取消" in _tool_messages(resumed)[-1].content
    assert _stored_status(orders_file, order_id) == "cancelled"
    assert not agent.get_state(config).next


def test_followup_requeries_instead_of_reusing_the_previous_result(orders_file: Path):
    """沿用只允许沿用订单号：上一轮的结果已过期时，本轮答案仍必须以工具返回值为准。"""
    model, agent = _agent()
    order_id = _order_id_with_status("paid")
    config = _config("stale-result")

    first = agent.invoke({"messages": [HumanMessage(f"帮我查一下订单 {order_id}")]}, config)
    assert "已付款（paid）" in _tool_messages(first)[-1].content

    # 两轮之间该订单被取消，等价于「上一轮的工具结果已经过时」
    assert OrderService().cancel_order(order_id).outcome == "CANCELLED"
    second = agent.invoke({"messages": [HumanMessage("它现在是什么状态？")]}, config)

    assert _tool_calls(second)[-1] == {"name": "query_order", "args": {"order_id": order_id}}
    assert "已取消（cancelled）" in _tool_messages(second)[-1].content
    # 作答时模型手里最后一份工具结果是本轮的新数据，不是上一轮的旧数据
    assert model.calls[-1][-1].type == "tool"
    assert "已取消（cancelled）" in str(model.calls[-1][-1].content)


# ---------------------------------------------------------------------------
# 提示词条款守卫：多轮口径被改坏要当场红
# ---------------------------------------------------------------------------


def test_system_prompt_resolves_references_from_history():
    assert "三、多轮对话与指代" in SYSTEM_PROMPT
    assert "订单号只能取自对话历史中真实出现过的值" in SYSTEM_PROMPT
    assert "不得把本提示里的示例订单号当成上文订单号" in SYSTEM_PROMPT
    assert "把候选订单号列出来，请用户说明是哪一笔" in SYSTEM_PROMPT


def test_system_prompt_keeps_asking_when_history_has_no_order_id():
    """多轮口径不能把「拿不到订单号就追问」这条兜底删掉：三处门槛都要各自保留禁调措辞。"""
    assert SYSTEM_PROMPT.count("也无法从对话历史确定订单号时") == 3
    for phrase in (
        "不要调用 query_order / query_logistics",
        "不要调用 cancel_order",
        "不要调用 refund_order",
        "政策与规则类问题不需要订单号",
    ):
        assert phrase in SYSTEM_PROMPT


def test_system_prompt_keeps_realtime_data_on_the_current_tool_call():
    assert "沿用的只允许是订单号" in SYSTEM_PROMPT
    assert "不得把上一轮的工具结果当作当前情况回答" in SYSTEM_PROMPT
    assert "上一轮人工批准的结论不等于本轮已获批准" in SYSTEM_PROMPT


def test_system_prompt_sections_stay_in_order():
    headers = [
        "一、工具选择",
        "二、订单号",
        "三、多轮对话与指代",
        "四、以工具结果为准，禁止编造",
        "五、取消订单（人工确认）",
        "六、退款申请（人工确认）",
        "七、表达",
    ]
    positions = [SYSTEM_PROMPT.index(header) for header in headers]
    assert positions == sorted(positions)


def test_conversation_state_lives_only_in_the_injected_checkpointer():
    """架构守卫：记忆的载体只能是调用方注入的 checkpointer + State.messages，图内部不自建存储。"""
    source = inspect.getsource(graph_module)
    assert "checkpointer=checkpointer" in source
    assert "langgraph.checkpoint" not in source, "graph 模块不得自己创建 checkpointer"
    assert "_HISTORY" not in source and "messages.append" not in source
