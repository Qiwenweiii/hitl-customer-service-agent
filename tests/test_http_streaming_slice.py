"""VS3 HTTP Streaming（SSE）切片：``POST /api/chat`` 的 Accept 协商与三类事件的编码。

不调用真实 LLM。分两层测：
1. **假 Agent 层**：脚本化地按真实 ``stream_mode="messages"`` 形状吐 ``(消息, metadata)`` 事件，
   记录 ``stream()`` 收到的入参。这层证的是 SSE 传输本身——分帧、事件顺序、JSON 转义、
   异常只回固定话术、工具原文不外泄、校验仍然挡在 Agent 之前、``Accept`` 没配对时同步行为一字未变。
2. **真图 + 假模型层**：``create_customer_service_agent(model=StreamStubModel(), checkpointer=...)``
   经 ``create_app`` 起完整应用后走流式，证「流式没有把 thread memory 弄坏」：
   同一 ``thread_id`` 第二次请求，模型数得出两轮提问。

事件只有三类（VS3 范围）：``token`` / ``done`` / ``error``。
断言里的「泄漏标记」只查响应**正文**：``X-Thread-Id`` 响应头按设计就该带 thread_id，不是泄漏。
"""

import inspect
import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

import app.agent.streaming as streaming_module
import app.api.chat as chat_module
from app.agent import create_customer_service_agent
from app.api.chat import THREAD_ID_HEADER, create_chat_app
from app.main import create_app

CHAT_URL = "/api/chat"
SSE_HEADERS = {"Accept": "text/event-stream"}
ANSWER = "已按本轮工具结果如实回答用户。"
TEXT_CHUNK_SIZE = 4
# 只存在于假异常里：出现在响应正文就说明把内部错误原文甩给了客户端
INTERNAL_ERROR_DETAIL = "OPENAI_BASE_URL=https://internal.example/v1 RETRIED=3"
# 只应存在于内部结构或工具结果里的字样，出现在 SSE 正文即为泄漏
LEAK_MARKERS = (
    "tool_calls",
    "tool_call_chunks",
    "ToolMessage",
    "AIMessageChunk",
    "langgraph_node",
    "response_metadata",
    "商品名称",
    "订单状态",
    "付款状态",
    "Traceback",
    "RuntimeError",
    INTERNAL_ERROR_DETAIL,
)


class _Interrupt:
    def __init__(self, value: dict) -> None:
        self.value = value


class _Task:
    def __init__(self, interrupts: Sequence[_Interrupt]) -> None:
        self.interrupts = list(interrupts)


class _State:
    """假 ``get_state`` 返回值：只有 ``tasks[*].interrupts`` 是 ``pending_confirmation`` 要读的东西。"""

    def __init__(self, pending: dict | None) -> None:
        self.tasks = () if pending is None else (_Task([_Interrupt(pending)]),)


class ScriptedStreamAgent:
    """假 Agent：按脚本吐 ``(消息, metadata)``，并记录 ``stream()`` 收到的入参。"""

    def __init__(self, messages: Sequence[Any], *, pending: dict | None = None) -> None:
        self._messages = list(messages)
        self._pending = pending
        self.streams: list[tuple[Any, dict, Any]] = []
        self.invoke_calls: list[tuple[Any, Any]] = []
        self.state_reads = 0

    def stream(self, payload: Any, config: dict, **kwargs: Any) -> Iterator[tuple[Any, dict]]:
        self.streams.append((payload, config, kwargs.get("stream_mode")))
        for message in self._messages:
            yield message, {"langgraph_node": "model"}

    def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self.invoke_calls.append((payload, config))
        raise AssertionError("流式请求不该走 invoke")

    def get_state(self, config: dict) -> _State:
        self.state_reads += 1
        return _State(self._pending)


class BrokenStreamAgent(ScriptedStreamAgent):
    """假 Agent：吐出一个文本增量之后模型侧报错（等价于 Key 缺失、上游超时）。"""

    def stream(self, payload: Any, config: dict, **kwargs: Any) -> Iterator[tuple[Any, dict]]:
        self.streams.append((payload, config, kwargs.get("stream_mode")))
        yield AIMessageChunk(content="前半句已经出来了。"), {"langgraph_node": "model"}
        raise RuntimeError(INTERNAL_ERROR_DETAIL)


class RecordingInvokeAgent:
    """假 Agent：只实现 invoke。未带 SSE Accept 的请求必须走它，误开流会被这里的断言抓住。"""

    def __init__(self, answer: str = ANSWER) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self._answer = answer

    def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self.calls.append((payload, config))
        return {"messages": [HumanMessage("订单 ORD-2026-0001 什么状态"), AIMessage(content=self._answer)]}

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("未带 Accept: text/event-stream 的请求不该开流")


class StreamStubModel(BaseChatModel):
    """假流式模型：把「第 N 轮提问」拆成两片下发，同时实现 ``_generate`` 保证两条入口同一个决定。"""

    @property
    def _llm_type(self) -> str:
        return "http-stream-stub"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "StreamStubModel":
        return self

    def _answer_of(self, messages: list[BaseMessage]) -> str:
        if messages[-1].type == "tool":
            return ANSWER
        question = next(
            (str(m.content) for m in reversed(messages) if m.type == "human"),
            "",
        )
        turns = sum(1 for m in messages if m.type == "human")
        return f"第 {turns} 轮提问：{question}"

    def _pieces(self, messages: list[BaseMessage]) -> list[str]:
        text = self._answer_of(messages)
        half = len(text) // 2
        return [text[:half], text[half:]]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._answer_of(messages)))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for piece in self._pieces(messages):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def _client(agent: Any) -> TestClient:
    return TestClient(create_chat_app(agent))


def _chunks(*pieces: str) -> list[AIMessageChunk]:
    """把若干段文本按真实协议形状切成小块。"""
    out: list[AIMessageChunk] = []
    for piece in pieces:
        for start in range(0, len(piece), TEXT_CHUNK_SIZE):
            out.append(AIMessageChunk(content=piece[start : start + TEXT_CHUNK_SIZE]))
    return out


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 正文按空行分帧解析回 ``(事件名, data)``；分帧不完整即失败。"""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block:
            continue
        name: str | None = None
        data: dict | None = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                assert data is None, f"一条事件不该有两个 data 行：{block!r}"
                data = json.loads(line[len("data: ") :])
            else:
                raise AssertionError(f"SSE 出现未知行：{line!r}")
        assert name is not None and data is not None, f"事件缺少 event 或 data 行：{block!r}"
        events.append((name, data))
    return events


def _ask_stream(
    client: TestClient,
    message: str,
    thread_id: str | None = None,
    *,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """发一次流式请求（默认就带上 SSE 的 Accept 头），返回状态码、响应头、原始正文与解析后的事件。"""
    payload: dict[str, Any] = {"message": message}
    if thread_id is not None:
        payload["thread_id"] = thread_id
    with client.stream(
        "POST", CHAT_URL, json=payload, headers=headers or SSE_HEADERS, **kwargs
    ) as response:
        response.read()
        return {
            "status": response.status_code,
            # 直接给 Headers 对象：它按大小写不敏感取值，dict() 会把键小写化而查不到 X-Thread-Id
            "headers": response.headers,
            "body": response.text,
            "events": _parse_sse(response.text),
        }


def _answer_of(outcome: dict[str, Any]) -> str:
    return "".join(data["content"] for name, data in outcome["events"] if name == "token")


def _names(outcome: dict[str, Any]) -> list[str]:
    return [name for name, _ in outcome["events"]]


# ---------------------------------------------------------------- 正常流式

STREAM_TEXT = "您好，订单 ORD-2026-0002 已发货，预计明天送达。"


def test_streaming_request_returns_event_stream() -> None:
    agent = ScriptedStreamAgent(_chunks(STREAM_TEXT))
    outcome = _ask_stream(_client(agent), "我的订单到哪里了")
    assert outcome["status"] == 200
    assert outcome["headers"]["content-type"].startswith("text/event-stream")


def test_events_are_tokens_then_done() -> None:
    agent = ScriptedStreamAgent(_chunks(STREAM_TEXT))
    names = _names(_ask_stream(_client(agent), "我的订单到哪里了"))
    assert set(names) == {"token", "done"}
    assert names[-1] == "done"
    assert names.count("done") == 1


def test_concatenated_tokens_are_the_full_answer() -> None:
    """客户端只会做一件事：把 token.content 按到达顺序拼起来。"""
    agent = ScriptedStreamAgent(_chunks(STREAM_TEXT))
    assert _answer_of(_ask_stream(_client(agent), "我的订单到哪里了")) == STREAM_TEXT


def test_token_event_frame_is_event_line_plus_data_line() -> None:
    """线格式必须是 ``event: token`` + 单行 ``data: {...}`` + 空行，事件名写死不是靠 content-type 区分。"""
    agent = ScriptedStreamAgent([AIMessageChunk(content="你好")])
    body = _ask_stream(_client(agent), "打招呼")["body"]
    assert body == 'event: token\ndata: {"content": "你好"}\n\nevent: done\ndata: {}\n\n'


def test_done_event_carries_an_empty_object() -> None:
    agent = ScriptedStreamAgent(_chunks("答复。"))
    events = _ask_stream(_client(agent), "问")["events"]
    assert events[-1] == ("done", {})


def test_newline_in_answer_stays_inside_one_data_line() -> None:
    """正文含换行时不能被切成第二个 data 行，否则分帧会被客户端读成两条事件。"""
    agent = ScriptedStreamAgent([AIMessageChunk(content="第一行\n第二行")])
    outcome = _ask_stream(_client(agent), "分点回答我")
    assert [line for line in outcome["body"].split("\n") if line.startswith("data: ")] == [
        'data: {"content": "第一行\\n第二行"}',
        "data: {}",
    ]
    assert _answer_of(outcome) == "第一行\n第二行"


def test_chinese_is_not_escaped_into_ascii() -> None:
    agent = ScriptedStreamAgent([AIMessageChunk(content="已发货")])
    assert "已发货" in _ask_stream(_client(agent), "发货了吗")["body"]


def test_stream_uses_messages_mode_and_one_human_message() -> None:
    agent = ScriptedStreamAgent(_chunks("答复。"))
    _ask_stream(_client(agent), "订单 ORD-2026-0002 到哪了")
    payload, _config, stream_mode = agent.streams[0]
    assert stream_mode == "messages"
    assert list(payload) == ["messages"]
    message = payload["messages"][0]
    assert isinstance(message, HumanMessage) and message.content == "订单 ORD-2026-0002 到哪了"


def test_only_the_three_agreed_event_types_are_ever_emitted() -> None:
    """VS3 只允许三类事件：多一个就说明有人顺手加了状态行/工具进度之类的事件。"""
    cases = [
        ScriptedStreamAgent(_chunks("答复。")),
        BrokenStreamAgent(_chunks("答复。")),
        ScriptedStreamAgent([AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}])], pending={"question": "确认？"}),
    ]
    for agent in cases:
        assert set(_names(_ask_stream(_client(agent), "问"))) <= {"token", "done", "error"}


def test_tool_message_and_call_structures_are_not_streamed() -> None:
    """工具原文与内部结构不进 SSE：只吐模型正文那一段。"""
    agent = ScriptedStreamAgent(
        [
            AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": "query_order", "args": "{}", "id": "call_1", "index": 0}],
            ),
            ToolMessage(content="订单号: ORD-2026-0002\n商品名称: 机械键盘\n订单状态: 已付款", tool_call_id="call_1"),
            *_chunks("机械键盘已发货。"),
        ]
    )
    outcome = _ask_stream(_client(agent), "订单 ORD-2026-0002 到哪了")
    assert _answer_of(outcome) == "机械键盘已发货。"
    for marker in LEAK_MARKERS:
        assert marker not in outcome["body"], f"SSE 正文泄漏了内部结构：{marker}"


# ---------------------------------------------------------------- thread_id 接线


def test_thread_id_is_passed_to_the_stream_config() -> None:
    agent = ScriptedStreamAgent(_chunks("答复。"))
    outcome = _ask_stream(_client(agent), "你好", "vip-1")
    assert agent.streams[0][1] == {"configurable": {"thread_id": "vip-1"}}
    assert outcome["headers"][THREAD_ID_HEADER] == "vip-1"


def test_generated_thread_id_is_echoed_in_a_response_header() -> None:
    """SSE 正文里没有承载会话 id 的字段，新建的会话必须从头回显，否则客户端接不上下一轮。"""
    agent = ScriptedStreamAgent(_chunks("答复。"))
    outcome = _ask_stream(_client(agent), "你好")
    thread_id = outcome["headers"][THREAD_ID_HEADER]
    assert thread_id.startswith("chat-") and len(thread_id) > len("chat-")
    assert agent.streams[0][1] == {"configurable": {"thread_id": thread_id}}


def test_two_streamed_turns_on_one_thread_share_memory() -> None:
    """真图 + 假模型：流式没有绕过 checkpointer，同一 thread_id 的第二轮看得到第一轮。"""
    agent = create_customer_service_agent(model=StreamStubModel(), checkpointer=InMemorySaver())
    client = _client(agent)
    first = _ask_stream(client, "你好")
    second = _ask_stream(client, "再问一次", first["headers"][THREAD_ID_HEADER])
    assert _answer_of(first) == "第 1 轮提问：你好"
    assert _answer_of(second) == "第 2 轮提问：再问一次"


def test_whole_app_streams_without_a_real_model() -> None:
    """经 create_app 起整站：VS2 的组装点照样支持流式，且仍是同一个 Agent 实例。"""
    app = create_app(model=StreamStubModel(), checkpointer=InMemorySaver())
    client = TestClient(app)
    outcome = _ask_stream(client, "你好")
    assert _answer_of(outcome) == "第 1 轮提问：你好"
    assert outcome["events"][-1][0] == "done"


# ---------------------------------------------------------------- Agent 异常


def test_agent_failure_becomes_error_event_without_the_traceback() -> None:
    agent = BrokenStreamAgent(_chunks("不会走到这里。"))
    outcome = _ask_stream(_client(agent), "你好")
    assert _names(outcome)[-1] == "error"
    assert outcome["events"][-1][1] == {"message": chat_module.STREAM_ERROR_MESSAGE}
    for marker in LEAK_MARKERS:
        assert marker not in outcome["body"], f"错误响应外泄了内部信息：{marker}"


def test_error_event_is_terminal_and_no_done_follows() -> None:
    """出错后不再补 done：客户端据最后一个事件名分辨成功与失败。"""
    agent = BrokenStreamAgent(_chunks("不会走到这里。"))
    assert _names(_ask_stream(_client(agent), "你好"))[-1] == "error"


def test_tokens_already_sent_survive_the_failure() -> None:
    """异常前已经发出的 token 不回滚：客户端能拼出「已到部分 + 中断」而不是空响应。"""
    agent = BrokenStreamAgent(_chunks("不会走到这里。"))
    outcome = _ask_stream(_client(agent), "你好")
    assert _answer_of(outcome) == "前半句已经出来了。"
    assert _names(outcome) == ["token", "error"]


def test_http_status_stays_200_when_the_stream_fails() -> None:
    """流式已经开始就不能改状态码：失败信息只能走 error 事件，这是 SSE 的既有约束。"""
    agent = BrokenStreamAgent(_chunks("x"))
    assert _ask_stream(_client(agent), "你好")["status"] == 200


# ---------------------------------------------------------------- 停在人工确认


CANCEL_PENDING = {"question": "确认取消订单 ORD-2026-0002？", "order_id": "ORD-2026-0002"}


def test_stream_parked_on_confirmation_reports_error_not_done() -> None:
    """messages 流不吐中断事件：没有任何文本 + 线程确实停在待确认点，必须明说这轮没办理。"""
    agent = ScriptedStreamAgent(
        [AIMessage(content="", tool_calls=[{"name": "cancel_order", "args": {"order_id": "ORD-2026-0002"}, "id": "c"}])],
        pending=CANCEL_PENDING,
    )
    outcome = _ask_stream(_client(agent), "帮我取消订单 ORD-2026-0002")
    assert _names(outcome) == ["error"]
    detail = outcome["events"][0][1]["message"]
    assert "人工确认" in detail and "没有执行" in detail
    assert agent.streams and agent.state_reads == 1


def test_stream_without_text_and_without_pending_still_closes_cleanly() -> None:
    """空流但线程没停在确认点上（假模型/极端情况）：照常 done 收尾，不误报成没办理。"""
    agent = ScriptedStreamAgent([AIMessageChunk(content="")])
    outcome = _ask_stream(_client(agent), "你好")
    assert _names(outcome) == ["done"]
    assert agent.state_reads == 1


def test_pending_confirmation_is_not_read_when_tokens_exist() -> None:
    """正常答复不该多付一次 checkpoint 读：只有零文本时才去问状态。"""
    agent = ScriptedStreamAgent(_chunks("答复。"), pending=CANCEL_PENDING)
    outcome = _ask_stream(_client(agent), "你好")
    assert _names(outcome)[-1] == "done"
    assert agent.state_reads == 0


# ---------------------------------------------------------------- 参数校验仍然挡在前面


INVALID_PAYLOADS: list[dict[str, Any]] = [
    {"message": ""},
    {"message": "   \n\t "},
    {"message": 123},
    {"message": "订单 ORD-2026-0001 什么状态", "thread_id": "   "},
    {"message": "订单 ORD-2026-0001 什么状态", "approved": True},
    {"message": "订单 ORD-2026-0001 什么状态", "stream": True},
]


@pytest.mark.parametrize("payload", INVALID_PAYLOADS, ids=lambda v: json.dumps(v, ensure_ascii=False))
def test_invalid_stream_requests_are_rejected_before_touching_the_agent(payload: dict[str, Any]) -> None:
    """校验与请求体形态都归 ChatRequest 管：流式分支没绕开它，也没往契约里塞新字段。"""
    agent = ScriptedStreamAgent(_chunks("答复。"))
    response = _client(agent).post(CHAT_URL, json=payload, headers=SSE_HEADERS)
    assert response.status_code == 422, response.text
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert agent.streams == [] and agent.invoke_calls == []


def test_empty_body_is_rejected_for_stream_requests_too() -> None:
    agent = ScriptedStreamAgent(_chunks("答复。"))
    assert _client(agent).post(CHAT_URL, json={}, headers=SSE_HEADERS).status_code == 422
    assert agent.streams == []


# ---------------------------------------------------------------- 同步路径不受影响


def test_request_without_accept_header_still_returns_json() -> None:
    agent = RecordingInvokeAgent()
    response = _client(agent).post(CHAT_URL, json={"message": "订单 ORD-2026-0001 什么状态"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "thread_id"}
    assert body["answer"] == ANSWER
    assert body["thread_id"].startswith("chat-")
    assert len(agent.calls) == 1


def test_wildcard_accept_stays_json() -> None:
    """关键守卫：curl 与多数客户端默认发 ``Accept: */*``，这类既有客户端不能因为 VS3 改成读 SSE。"""
    agent = RecordingInvokeAgent()
    response = _client(agent).post(
        CHAT_URL, json={"message": "订单 ORD-2026-0001 什么状态"}, headers={"Accept": "*/*"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["answer"] == ANSWER
    assert len(agent.calls) == 1


def test_json_body_content_type_does_not_imply_streaming() -> None:
    agent = RecordingInvokeAgent()
    response = _client(agent).post(
        CHAT_URL, json={"message": "你好"}, headers={"Content-Type": "application/json"}
    )
    assert isinstance(response.json().get("answer"), str)


# ---------------------------------------------------------------- 架构守卫


def _code_only(source: str) -> str:
    """去掉三引号正文，守卫只看代码，docstring 里合法提到 SSE／事件名不算违规。"""
    return re.sub(r'""".*?"""', "", source, flags=re.S)


def test_streaming_layer_knows_nothing_about_http_or_sse() -> None:
    """分层要求：SSE／FastAPI 的细节只能在 API 层，展示层只产出纯文本增量。"""
    source = _code_only(inspect.getsource(streaming_module))
    for forbidden in ("fastapi", "StreamingResponse", "Request", "event:", "text/event-stream", "json.dumps"):
        assert forbidden not in source, f"展示层不该出现 {forbidden}"


def test_api_reuses_the_shared_stream_driver() -> None:
    """流式取值必须走展示层唯一实现，不在 API 层另写一份 messages 循环。"""
    assert chat_module.stream_answer.__module__ == "app.agent.streaming"
    assert chat_module.pending_confirmation.__module__ == "app.agent.streaming"


def test_stream_driver_yields_only_displayable_text() -> None:
    """``stream_answer`` 自己只吐非空文本：ToolMessage、空分片、非 AI 消息都被它挡掉。"""
    events = [
        HumanMessage("问"),
        AIMessageChunk(content=""),
        ToolMessage(content="订单状态: 已付款", tool_call_id="c1"),
        AIMessageChunk(content="已发货。"),
    ]

    class _Stream:
        def stream(self, payload: Any, config: dict, **kwargs: Any) -> Iterator[tuple[Any, dict]]:
            for message in events:
                yield message, {}

    assert list(streaming_module.stream_answer(_Stream(), {"messages": []}, {})) == ["已发货。"]


def test_streaming_adds_no_endpoint_and_no_request_field() -> None:
    """VS3 不新增端点也不改 ChatRequest：一条 URL、两个请求字段、两个响应字段。"""
    app: FastAPI = create_chat_app(ScriptedStreamAgent(_chunks("答复。")))
    assert [r.path for r in app.routes if r.path.startswith("/api/")] == [CHAT_URL]
    assert list(chat_module.ChatRequest.model_fields) == ["message", "thread_id"]
    assert list(chat_module.ChatResponse.model_fields) == ["answer", "thread_id"]


def test_exactly_three_stream_event_names_reach_the_wire() -> None:
    """三类事件不多不少：把成功、出错、停在确认三种场景的真实事件名取并集来判定。"""
    scenarios = [
        ScriptedStreamAgent(_chunks("答复。")),
        BrokenStreamAgent(_chunks("x")),
        ScriptedStreamAgent(
            [AIMessage(content="", tool_calls=[{"name": "cancel_order", "args": {}, "id": "c"}])],
            pending=CANCEL_PENDING,
        ),
    ]
    emitted: set[str] = set()
    for agent in scenarios:
        emitted.update(_names(_ask_stream(_client(agent), "问")))
    assert emitted == {"token", "done", "error"}
    assert chat_module.MEDIA_TYPE_EVENT_STREAM == "text/event-stream"
