"""CLI Streaming 切片：messages 流里的文本增量上屏、工具调用只映射成状态行。

不调用真实 LLM。这里用的是「脚本化假流式模型」：它按真实流式协议的形状把一次模型输出
拆成 ``tool_call_chunks`` 分片和文本分片再逐条 yield，因此测的是「分片到展示」这条链路，
不是模型的选择能力（选择对不对由 eval/ 负责）。

分片形状来自探针实测（langchain-core 1.6.3 / langgraph 1.2.11）：
- 一次工具调用 = 首个分片带 name+id+参数前缀，后续分片 name=None 只有参数碎片，末尾再一条空分片；
- 工具执行完吐的是 ``ToolMessage``（节点 tools），它没有 ``tool_calls`` 属性；
- 图在人工确认处暂停时，流里只有工具调用分片、没有 ToolMessage，中断信息只在 checkpoint 状态里。

写入类测试同样把 data_loader.ORDERS_FILE 指向 tmp_path 副本，绝不改真实订单数据。
CLI 是 scripts/ 下第一个被测模块：流式循环（读流、去重、什么时候换行）只有从模块 import
进来才能断言，复制到测试里等于测了一份不会跑的副本。
"""

import inspect
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

import app.agent.streaming as streaming_module
import app.data_loader as data_loader
import scripts.hitl_cli as hitl_cli
import scripts.stream_cli as stream_cli
from app.agent import create_customer_service_agent
from app.agent.graph import TOOLS
from app.agent.streaming import (
    TOOL_STATUS_LABELS,
    UNKNOWN_TOOL_STATUS,
    pending_confirmation,
    stream_text,
    tool_status_lines,
)

ORDER_ID_PATTERN = re.compile(r"ORD-\d{4}-\d{4}")
FINAL_ANSWER = "已按本轮工具结果如实回答用户。"
ASK_ORDER_ID = "请提供您的订单号，形如 ORD-2026-0001。"
TEXT_CHUNK_SIZE = 4

# 这些字符串只应存在于内部结构里，出现在屏上就是泄漏给用户。
INTERNAL_MARKERS = (
    "tool_calls",
    "tool_call_chunks",
    "ToolMessage",
    "AIMessageChunk",
    "SystemMessage",
    "langgraph_node",
    "thread_id",
    "response_metadata",
    "call_",
    "args",
)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _split(text: str) -> list[str]:
    return [text[index : index + TEXT_CHUNK_SIZE] for index in range(0, len(text), TEXT_CHUNK_SIZE)]


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


class ScriptedStreamModel(BaseChatModel):
    """假流式模型：先按「本轮提问 + 上文」决定调哪个工具或答什么，再把结果拆成真实分片下发。

    同时实现 ``_generate`` 与 ``_stream`` 是刻意的：现有 ``invoke()`` 入口（hitl_cli、eval）
    走的是前者，流式入口走的是后者，两条路径必须给出同一个决定，
    这样"加了流式没把原路径改坏"才是被断言过的，而不是被假设的。
    """

    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-stream"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedStreamModel":
        return self

    def _historic_order_id(self, messages: Sequence[BaseMessage]) -> str | None:
        """本轮没给订单号时，只允许沿用上文唯一的那笔；系统提示里的示例订单号不算。"""
        found: list[str] = []
        for message in messages[:-1]:
            if message.type == "system":
                continue
            blob = str(message.content) + " ".join(
                str(call["args"]) for call in getattr(message, "tool_calls", [])
            )
            found.extend(ORDER_ID_PATTERN.findall(blob))
        return found[-1] if len(set(found)) == 1 else None

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if messages[-1].type == "tool":
            return AIMessage(content=FINAL_ANSWER)
        question = _turn_question(messages)
        if in_question := ORDER_ID_PATTERN.findall(question):
            order_id = in_question[0]
        else:
            order_id = self._historic_order_id(messages)
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

    def _chunks(self, message: AIMessage) -> list[ChatGenerationChunk]:
        if message.tool_calls:
            call = message.tool_calls[0]
            args = json.dumps(call["args"], ensure_ascii=False)
            half = len(args) // 2
            # 与真实协议同形：工具名只在首个分片出现，后续分片只有参数碎片。
            return [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {"name": call["name"], "args": args[:half], "id": call["id"], "index": 0}
                        ],
                    )
                ),
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {"name": None, "args": args[half:], "id": None, "index": 0}
                        ],
                    )
                ),
                ChatGenerationChunk(message=AIMessageChunk(content="")),
            ]
        return [
            ChatGenerationChunk(message=AIMessageChunk(content=piece))
            for piece in _split(message.content)
        ]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        yield from self._chunks(self._decide(messages))


def _agent() -> tuple[ScriptedStreamModel, Any]:
    model = ScriptedStreamModel()
    return model, create_customer_service_agent(model=model, checkpointer=InMemorySaver())


def _events(agent, payload, config) -> list[Any]:
    """直接跑一条流，拿到原始事件用于断言形状，不经过 CLI 的展示层。"""
    return [message for message, _ in agent.stream(payload, config, stream_mode="messages")]


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


def _stored_field(path: Path, order_id: str, field: str) -> object:
    return next(
        item[field]
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item["order_id"] == order_id
    )


def _stored_status(path: Path, order_id: str) -> str:
    return _stored_field(path, order_id, "status")


# ---------------------------------------------------------------------------
# 展示层纯函数：非 AI 消息、分片去重、未登记工具
# ---------------------------------------------------------------------------


def test_stream_text_only_returns_ai_message_text():
    """约束 10：messages 流里不只有 AIMessageChunk，其他类型必须安静地返回空串而不是报错。"""
    # 分片类的 .type 不是 "ai"，用字符串比较判 AI 消息会把整条流的正文丢掉
    assert AIMessageChunk(content="订单已").type != "ai"
    assert stream_text(AIMessageChunk(content="订单已")) == "订单已"
    assert stream_text(AIMessage(content="发货了")) == "发货了"
    tool_message = ToolMessage(content="订单号: ORD-2026-0004\n商品名称: 蓝牙耳机", tool_call_id="call_1")
    assert stream_text(tool_message) == ""
    assert stream_text(HumanMessage("帮我查订单")) == ""
    assert stream_text(SystemMessage("你是客服")) == ""


def test_stream_text_keeps_only_text_blocks_from_structured_content():
    message = AIMessageChunk(
        content=[
            {"type": "text", "text": "退款"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "reasoning", "reasoning": "内部思考"},
        ]
    )
    assert stream_text(message) == "退款"


def test_tool_status_line_comes_from_the_tool_name():
    """状态文案由工具名查表得到，与模型输出无关（约束 6）。"""
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "query_order", "args": "{", "id": "call_1", "index": 0}],
    )
    assert tool_status_lines(chunk, seen=set()) == ["[正在查询订单...]"]


def test_status_line_is_emitted_once_per_tool_call():
    """一次调用被拆成多条分片：只有带名字的首片出一行，参数碎片与收尾空片不再重复提示。"""
    _, agent = _agent()
    config = _config("dedupe")
    events = _events(agent, {"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)

    seen: set[str] = set()
    lines = [line for message in events for line in tool_status_lines(message, seen=seen)]
    assert lines == ["[正在查询订单...]"]
    # 参数分片确实到了，只是不产生提示
    assert any(
        getattr(message, "tool_call_chunks", None)
        and not message.tool_call_chunks[0]["name"]
        for message in events
    )


def test_complete_message_without_chunks_still_shows_status():
    """模型不支持流式时 messages 模式只吐一条完整 AIMessage，回退读 tool_calls 同一段逻辑。"""
    message = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge", "args": {"query": "退款时效"}, "id": "call_1"}],
    )
    assert tool_status_lines(message, seen=set()) == ["[正在查询相关政策...]"]


def test_tool_message_produces_no_status_and_no_text():
    """工具结果事件在屏上必须完全隐身：它的内容是喂给模型的中间数据。"""
    message = ToolMessage(content="订单号: ORD-2026-0004\n订单状态: 已签收（delivered）", tool_call_id="call_1")
    assert tool_status_lines(message, seen=set()) == []
    assert stream_text(message) == ""


def test_unregistered_tool_falls_back_to_generic_status():
    """以后新增工具忘了登记时，宁可显示一句通用提示，也不把内部工具名抛给用户。"""
    message = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "reissue_invoice", "args": "{", "id": "c1", "index": 0}]
    )
    lines = tool_status_lines(message, seen=set())
    assert lines == [UNKNOWN_TOOL_STATUS]
    assert "reissue_invoice" not in lines[0]


# ---------------------------------------------------------------------------
# CLI 端到端：增量上屏 + 只有正文和状态行
# ---------------------------------------------------------------------------


def test_cli_prints_text_deltas_instead_of_the_whole_reply(monkeypatch: pytest.MonkeyPatch):
    """流式的意义就是边到边打屏：断言 CLI 逐片 print(end="")，而不是攒完再打一次。"""
    writes: list[tuple[str, str]] = []

    def record(*args: object, **kwargs: Any) -> None:
        writes.append(("".join(str(arg) for arg in args), kwargs.get("end", "\n")))

    monkeypatch.setattr("builtins.print", record)
    _, agent = _agent()
    stream_cli.stream_once(
        agent, {"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, _config("delta")
    )

    deltas = [text for text, end in writes if end == ""]
    assert len(deltas) > 1
    assert "".join(deltas) == FINAL_ANSWER
    assert "[正在查询订单...]" in [text for text, end in writes if end == "\n"]


def test_run_turn_shows_only_status_and_reply(capsys: pytest.CaptureFixture[str]):
    """一轮查询：屏上有状态行与最终答复，没有工具结果原文、调用结构和 metadata。"""
    _, agent = _agent()
    stream_cli.run_turn(agent, "帮我查一下订单 ORD-2026-0004", _config("clean-output"))

    out = capsys.readouterr().out
    assert "[正在查询订单...]" in out
    assert FINAL_ANSWER in out
    # 工具返回值里的字段是喂给模型的，不该出现在流式 CLI 上
    assert "商品名称" not in out
    assert "订单状态" not in out
    for marker in INTERNAL_MARKERS:
        assert marker not in out, f"内部结构泄漏到 CLI：{marker}"


def test_run_turn_keeps_status_before_the_reply_lines_up(capsys: pytest.CaptureFixture[str]):
    """状态行必须独占一行，不能和还没收尾的文本粘在同一行上。"""
    _, agent = _agent()
    stream_cli.run_turn(agent, "帮我查一下订单 ORD-2026-0004", _config("line-layout"))

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines == ["[正在查询订单...]", FINAL_ANSWER]


def test_logistics_followup_shows_the_logistics_status(capsys: pytest.CaptureFixture[str]):
    """不同工具必须给出各自那句状态，而不是笼统一句"正在处理"。"""
    _, agent = _agent()
    config = _config("per-tool")
    stream_cli.run_turn(agent, "ORD-2026-0003 到哪了", config)
    out = capsys.readouterr().out
    assert "[正在查询物流...]" in out
    assert "[正在查询订单...]" not in out


# ---------------------------------------------------------------------------
# Memory / thread_id 不受流式改造影响
# ---------------------------------------------------------------------------


def _tool_calls_of(agent, config: dict) -> list[tuple[str, str]]:
    """从线程状态里读本轮实际发生过的工具调用（假模型记录的是输入消息，不含它的决定）。"""
    return [
        (call["name"], str(call["args"]))
        for message in agent.get_state(config).values["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def test_streaming_turns_share_context_on_one_thread():
    """第二轮的模型输入里必须有第一轮的消息，且指代仍然解析到同一笔订单。"""
    model, agent = _agent()
    config = _config("memory")

    stream_cli.run_turn(agent, "帮我查一下订单 ORD-2026-0004", config)
    stream_cli.run_turn(agent, "那它现在到哪了？", config)

    second_turn_input = model.calls[-2]
    assert "帮我查一下订单 ORD-2026-0004" in [str(m.content) for m in second_turn_input]
    assert [m.type for m in second_turn_input] == ["system", "human", "ai", "tool", "ai", "human"]
    # 沿用上文订单号后仍重新调工具（只有订单号可以沿用），且按追问内容换成物流工具
    assert [name for name, _ in _tool_calls_of(agent, config)] == [
        "query_order",
        "query_logistics",
    ]
    assert "ORD-2026-0004" in _tool_calls_of(agent, config)[-1][1]


def test_other_thread_does_not_see_this_threads_history():
    model, agent = _agent()
    stream_cli.run_turn(agent, "帮我查一下订单 ORD-2026-0004", _config("thread-a"))
    stream_cli.run_turn(agent, "那它现在到哪了？", _config("thread-b"))

    other_thread_input = model.calls[-1]
    assert [m.type for m in other_thread_input] == ["system", "human"]
    assert "帮我查一下订单 ORD-2026-0004" not in [str(m.content) for m in other_thread_input]


# ---------------------------------------------------------------------------
# HITL 在流式下仍然可用：暂停、裁决、恢复后再开一条流
# ---------------------------------------------------------------------------


def test_cancel_pauses_streams_status_then_completes_after_approval(
    orders_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """完整链路：状态行 → 暂停打印 payload → 人工批准 → 恢复流给出答复，且只有批准后订单才变。"""
    order_id = _order_id_with_status("pending")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    _, agent = _agent()
    config = _config("cancel-approve")

    stream_cli.run_turn(agent, f"帮我取消订单 {order_id}", config)

    out = capsys.readouterr().out
    assert "[正在检查订单是否可以取消...]" in out
    assert "[暂停] 需要人工确认：" in out
    assert order_id in out  # 确认信息必须给人看明白是哪一笔
    assert FINAL_ANSWER in out
    assert _stored_status(orders_file, order_id) == "cancelled"
    assert pending_confirmation(agent, config) is None
    assert not agent.get_state(config).next


def test_cancel_denied_at_the_prompt_leaves_the_order_untouched(
    orders_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """拒绝路径同样要跑完流：订单保持原样，且没有停在待确认状态。"""
    order_id = _order_id_with_status("pending")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    _, agent = _agent()
    config = _config("cancel-deny")

    stream_cli.run_turn(agent, f"帮我取消订单 {order_id}", config)

    assert "[暂停] 需要人工确认：" in capsys.readouterr().out
    assert _stored_status(orders_file, order_id) == "pending"
    assert pending_confirmation(agent, config) is None


def test_confirmation_question_is_taken_from_the_interrupt_payload(
    orders_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """确认话术只在 Tool 层有一份：CLI 直接问 payload 里的 question，不再复制一张提示表。"""
    order_id = _order_id_with_status("pending")
    asked: list[str] = []

    def record_input(prompt: str = "") -> str:
        asked.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", record_input)
    _, agent = _agent()
    stream_cli.run_turn(agent, f"帮我取消订单 {order_id}", _config("question-from-payload"))

    assert order_id in asked[0]
    assert "是否确认执行取消" in asked[0]


def test_pending_confirmation_reads_the_checkpoint_not_the_stream(orders_file: Path):
    """探针结论：中断不会作为事件出现在 messages 流里，待确认信息只能从状态读。"""
    _, agent = _agent()
    config = _config("interrupt-shape")
    order_id = _order_id_with_status("pending")

    events = _events(agent, {"messages": [HumanMessage(f"帮我取消订单 {order_id}")]}, config)

    assert not any(getattr(message, "type", None) == "tool" for message in events)
    payload = pending_confirmation(agent, config)
    assert payload["type"] == "order_cancellation_confirmation"
    assert payload["order_id"] == order_id


def test_refund_streams_its_own_status_and_asks_for_confirmation(
    orders_file: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """退款走的是同一套展示层：状态句不同、确认 payload 来自 refund 工具，拒绝则不写付款状态。

    签收时间按"昨天"写进副本而不是用 mock 里那两笔 delivered：图用的 RefundService 带真实时钟，
    写死日期会让这条测试随日历漂出退款时效。
    """
    order_id = _order_id_with_status("paid")
    delivered_at = datetime.now() - timedelta(days=1)
    raw = json.loads(orders_file.read_text(encoding="utf-8"))
    for item in raw:
        if item["order_id"] == order_id:
            item["status"] = "delivered"
            item["delivered_at"] = delivered_at.isoformat(sep=" ")
    orders_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda *_: "n")
    _, agent = _agent()
    config = _config("refund-stream")

    stream_cli.run_turn(agent, f"{order_id} 我要退款", config)

    out = capsys.readouterr().out
    assert "[正在处理退款请求...]" in out
    assert "[暂停] 需要人工确认：" in out
    assert "是否确认受理退款" in out
    assert "[正在检查订单是否可以取消...]" not in out
    assert _stored_field(orders_file, order_id, "payment_status") == "paid"
    assert pending_confirmation(agent, config) is None


# ---------------------------------------------------------------------------
# 既有入口不受影响：invoke 路径与 hitl_cli 仍然照常工作
# ---------------------------------------------------------------------------


def test_invoke_path_still_returns_a_complete_reply():
    """同一个假模型的 invoke 分支必须给出与流式一致的结果，钉住"没改坏非流式路径"。"""
    model, agent = _agent()
    config = _config("invoke-untouched")

    result = agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)

    assert result["messages"][-1].content == FINAL_ANSWER
    assert "__interrupt__" not in result
    assert len(model.calls) == 2


def test_hitl_cli_still_drives_the_same_agent(
    orders_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """hitl_cli 的非流式确认入口不受影响：还是 invoke + result['__interrupt__']。"""
    order_id = _order_id_with_status("pending")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    _, agent = _agent()
    config = _config("hitl-cli-regression")

    hitl_cli.run_turn(agent, f"帮我取消订单 {order_id}", config)

    out = capsys.readouterr().out
    assert "[暂停] 需要人工确认：" in out
    assert f"客服: {FINAL_ANSWER}" in out
    assert _stored_status(orders_file, order_id) == "cancelled"


# ---------------------------------------------------------------------------
# 架构守卫：文案只有一份、展示层不做 IO、CLI 不自己拼内部结构
# ---------------------------------------------------------------------------


def test_status_labels_are_the_fixed_wording_for_all_bound_tools():
    """文案与绑定工具一一对应：新增工具没登记就红，登记内容被改坏也红。"""
    assert TOOL_STATUS_LABELS == {
        "query_order": "[正在查询订单...]",
        "query_logistics": "[正在查询物流...]",
        "search_knowledge": "[正在查询相关政策...]",
        "cancel_order": "[正在检查订单是否可以取消...]",
        "refund_order": "[正在处理退款请求...]",
    }
    assert {tool.name for tool in TOOLS} == set(TOOL_STATUS_LABELS)


def test_status_wording_never_leaks_the_tool_identifier():
    for name, label in TOOL_STATUS_LABELS.items():
        assert name not in label
        assert label.startswith("[") and label.endswith("...]")


def test_presenter_is_pure_and_does_not_touch_metadata():
    """展示层只做类型判断与查表：不打印、不读输入、不解析中断，也不碰 metadata。"""
    source = inspect.getsource(streaming_module)
    assert "print(" not in source
    assert "input(" not in source
    assert "interrupt(" not in source
    assert "response_metadata" not in source


def _code_only(source: str) -> str:
    """去掉三引号文档字符串：注释里举例说明"屏上会出现什么"是文档，不该被当成第二份文案。"""
    return re.sub(r'""".*?"""', "", source, flags=re.S)


def test_cli_has_no_second_copy_of_the_status_wording():
    """状态文案与确认话术各自只有一份实现，避免 CLI 与展示层／Tool 层漂移。"""
    code = _code_only(inspect.getsource(stream_cli))
    assert "[正在" not in code
    assert "_CONFIRMATION_PROMPTS" not in code
    assert 'stream_mode="messages"' in code
    assert "stream_text(" in code and "tool_status_lines(" in code
