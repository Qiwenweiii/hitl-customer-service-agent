"""Chat API 切片：VS1 的 ``POST /api/chat`` HTTP 契约与 thread_id 接线，VS2 的应用组装与生命周期。

不调用真实 LLM。分三层测：
1. **假 Agent 层**：只记录 invoke 收到的参数、返回预先备好的图结果。
   这层证明的是 HTTP 契约本身——字段校验、thread_id 透传、响应只含 answer／thread_id、
   工具调用与 ToolMessage 不会泄漏到响应里、异常不会被吞成 200。
2. **真图 + 假模型层**：用 ``create_customer_service_agent(model=StubModel(), checkpointer=...)``
   跑完整链路。这层证明的是「thread_id 真的接到了 checkpointer 上」：
   同一个 thread_id 的第二次请求，模型能数出两轮提问；换个 thread_id 就只有一轮。
   以及「需要人工确认的请求不会被 API 悄悄放行」：返回 501 且订单文件字节不变。
3. **应用工厂层**（``app/main.py`` 的 ``create_app``）：monkeypatch 掉建 Agent 的工厂数创建次数，
   证明 Agent 与 Checkpointer 只在启动时创建一次、三个请求共用同一个实例，
   并且组装仍旧只把 Agent 交给 ``create_chat_app``（没有第二套装路由的地方）。

因此这些断言证明的是链路与契约，不是模型的理解能力（真实模型由 eval/ 负责）。
写入类测试一律把 ``data_loader.ORDERS_FILE`` 指向 tmp_path 副本，绝不改真实订单数据。
"""

import inspect
import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

import app.api.chat as chat_module
import app.data_loader as data_loader
import app.main as app_module
from app.agent import create_customer_service_agent
from app.api.chat import ChatRequest, ChatResponse, create_chat_app
from app.session_store import SessionStore

ORDER_ID_PATTERN = re.compile(r"ORD-\d{4}-\d{4}")
CHAT_URL = "/api/chat"
# 假模型拿到工具结果后只回这句与订单内容无关的话：answer 等于它就说明工具跑完了且原文没外泄
TOOL_BASED_ANSWER = "已按本轮工具结果如实回答用户。"
# 假 Agent 返回的答复文本：刻意不含 INTERNAL_MARKERS 里的任何子串，否则泄漏断言会被自己的数据误伤
ANSWER = "这一轮的客服答复。"
# 响应体里绝对不该出现的东西：消息对象、工具调用、内部元数据、以及只存在于工具结果里的订单字段
INTERNAL_MARKERS = (
    "tool_calls",
    "tool_call_chunks",
    "ToolMessage",
    "AIMessage",
    "HumanMessage",
    "SystemMessage",
    "langgraph_node",
    "response_metadata",
    "商品名称",
    "订单状态",
    "付款状态",
)
# 请求体缺字段／类型错/空白等非法取值的组合，每一项都必须被 422 挡下
INVALID_PAYLOADS: list[dict[str, Any]] = [
    {},  # 完全空
    {"thread_id": "vip-1"},  # 缺 message
    {"message": ""},  # 空串
    {"message": "   \n\t "},  # 只有空白：min_length 拦不住，必须靠 strip
    {"message": 123},  # 类型错：不隐式转成字符串
    {"message": ["订单状态"]},  # 类型错
    {"message": None},  # null 不是合法提问
    {"message": "订单 ORD-2026-0001 什么状态", "thread_id": "   "},  # 空白 thread_id
    {"message": "订单 ORD-2026-0001 什么状态", "approved": True},  # 未知字段：不静默忽略
    {"message": "订单 ORD-2026-0001 什么状态", "stream": True},  # 流式尚未实现，不接受开关
]


def _result(answer: str, *, with_tool_message: bool = False) -> dict[str, Any]:
    """伪造一份图返回值：最后一条是 AI 答复，中间可带一条含订单原文的 ToolMessage。"""
    messages: list[BaseMessage] = [HumanMessage("订单 ORD-2026-0001 什么状态")]
    if with_tool_message:
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "query_order", "args": {"order_id": "ORD-2026-0001"}, "id": "call_1"}],
            )
        )
        messages.append(
            ToolMessage(
                content="订单号: ORD-2026-0001\n商品名称: 无线耳机\n订单状态: 待付款（pending）",
                tool_call_id="call_1",
            )
        )
    messages.append(AIMessage(content=answer))
    return {"messages": messages}


class RecordingAgent:
    """假 Agent：原样记录 invoke 收到的入参，返回预设结果。"""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.result = result if result is not None else _result(ANSWER)

    def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self.calls.append((payload, config))
        return self.result


class BrokenAgent:
    """假 Agent：模型侧报错（等价于 API Key 缺失、上游超时）。"""

    def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        raise RuntimeError("LLM 调用失败")


class StubModel(BaseChatModel):
    """假模型：按本轮提问决定调哪个工具，拿到工具结果后回固定话术；问「取消」则停在人工确认。

    两个刻意的取数口径：
    - 「第 N 轮」统计的是自己**收到的消息列表**里的用户轮次，
      这个结论只能来自 checkpointer 里的历史，不可能是客户端传进来的——这正是本切片要证明的那条线。
    - 工具跑完只回一句与订单内容无关的固定话术，
      因此响应里一旦出现商品名、订单状态等字样，就能断定是工具原文泄漏了。
    """

    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "chat-api-stub"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "StubModel":
        return self

    def _question(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if message.type == "human":
                return str(message.content)
        return ""

    def _tool_call(self, name: str, order_id: str) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": name,
                                "args": {"order_id": order_id},
                                "id": f"call_{len(self.calls)}",
                            }
                        ],
                    )
                )
            ]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if messages[-1].type == "tool":
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=TOOL_BASED_ANSWER))]
            )
        question = self._question(messages)
        order_id = (ORDER_ID_PATTERN.findall(question) or [""])[0]
        if "取消" in question:
            return self._tool_call("cancel_order", order_id)
        if order_id:
            return self._tool_call("query_order", order_id)
        turns = sum(1 for message in messages if message.type == "human")
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=f"第 {turns} 轮提问：{question}"))
            ]
        )


def _client(agent: Any, **kwargs: Any) -> TestClient:
    return TestClient(create_chat_app(agent), **kwargs)


@pytest.fixture
def real_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, StubModel, Path]:
    """真图 + 假模型：订单文件换成 tmp_path 副本，任何写库都碰不到 data/mock 里的真实数据。"""
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    monkeypatch.setattr(data_loader, "ORDERS_FILE", copy_path)
    model = StubModel()
    agent = create_customer_service_agent(model=model, checkpointer=InMemorySaver())
    return _client(agent), model, copy_path


def _ask(
    client: TestClient,
    message: str,
    thread_id: str | None = None,
    *,
    session_token: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message}
    if thread_id is not None:
        payload["thread_id"] = thread_id
    if session_token is not None:
        payload["session_token"] = session_token
    response = client.post(CHAT_URL, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _create_session(client: TestClient) -> tuple[str, str]:
    """走分支 A 新建会话，返回 (thread_id, session_token)。

    经 ``create_app`` 起的整站都注入了 SessionStore，续轮必须同时带这两个值，
    这是会话归属契约：裸 thread_id 只会换来 403。
    """
    body = _ask(client, "你好")
    return body["thread_id"], body["session_token"]


def _post(client: TestClient, payload: Any) -> Any:
    return client.post(CHAT_URL, json=payload)


def _stored_order(path: Path, order_id: str) -> dict[str, Any]:
    return next(
        item for item in json.loads(path.read_text(encoding="utf-8")) if item["order_id"] == order_id
    )


def _pending_order_id() -> str:
    matched = [order.order_id for order in data_loader.load_orders() if order.status == "pending"]
    assert len(matched) == 1, f"需要恰好一笔可取消的待付款订单，实际 {matched}"
    return matched[0]


def _code_only(source: str) -> str:
    """去掉三引号文档字符串，只留代码。

    入口脚本的说明文字里合法地提到 InMemorySaver、create_chat_app 这些实现细节，
    拿全文当代码扫会把注释当成调用（与 test_streaming_slice.py 同一手法）。
    """
    return re.sub(r'""".*?"""', "", source, flags=re.S)


# ---------------------------------------------------------------- 正常请求


def test_chat_returns_answer_and_thread_id() -> None:
    agent = RecordingAgent(_result(ANSWER))
    body = _ask(_client(agent), "订单 ORD-2026-0001 什么状态", "vip-1")
    assert body == {"answer": ANSWER, "thread_id": "vip-1"}


def test_response_body_carries_only_the_contract_fields() -> None:
    """工具结果里的订单原文不能出现在响应里：只回 answer／thread_id 两个字段。"""
    agent = RecordingAgent(_result(ANSWER, with_tool_message=True))
    response = _post(_client(agent), {"message": "订单 ORD-2026-0001 什么状态", "thread_id": "vip-1"})
    assert set(response.json()) == {"answer", "thread_id"}
    for marker in INTERNAL_MARKERS:
        assert marker not in response.text, f"响应泄漏了内部结构：{marker}"


def test_answer_is_taken_from_the_last_message_only() -> None:
    """中间那条 AI 消息（带工具调用、正文为空）不能被当成答复。"""
    agent = RecordingAgent(_result("最终答复。", with_tool_message=True))
    assert _ask(_client(agent), "订单 ORD-2026-0001 什么状态")["answer"] == "最终答复。"


def test_content_blocks_are_flattened_to_text() -> None:
    """模型返回内容块形态时，answer 必须是纯文本而不是 blocks 结构。"""
    messages = [AIMessage(content=[{"type": "text", "text": "已找到订单。"}, {"type": "image", "url": "x"}])]
    agent = RecordingAgent({"messages": messages})
    assert _ask(_client(agent), "订单 ORD-2026-0001 什么状态")["answer"] == "已找到订单。"


def test_agent_is_invoked_exactly_once_per_request() -> None:
    """一次请求只跑一轮图：没有重试、更没有自动恢复某个中断。"""
    agent = RecordingAgent()
    _ask(_client(agent), "订单 ORD-2026-0001 什么状态", "vip-1")
    assert len(agent.calls) == 1


# ---------------------------------------------------------------- thread_id 接线


def test_thread_id_is_passed_to_the_checkpointer_config() -> None:
    agent = RecordingAgent()
    _ask(_client(agent), "订单 ORD-2026-0001 什么状态", "vip-1")
    assert agent.calls[0][1] == {"configurable": {"thread_id": "vip-1"}}


def test_message_becomes_a_single_human_message() -> None:
    agent = RecordingAgent()
    _ask(_client(agent), "  订单 ORD-2026-0001 什么状态  ", "vip-1")
    payload = agent.calls[0][0]
    assert list(payload) == ["messages"]
    messages = payload["messages"]
    assert len(messages) == 1 and isinstance(messages[0], HumanMessage)
    assert messages[0].content == "订单 ORD-2026-0001 什么状态"


def test_generated_thread_id_is_echoed_back_and_used_in_config() -> None:
    """没带 thread_id 时新建的会话必须同时回显给客户端并真的用于本轮配置，否则客户端接不上下一轮。"""
    agent = RecordingAgent()
    body = _ask(_client(agent), "订单 ORD-2026-0001 什么状态")
    thread_id = body["thread_id"]
    assert thread_id.startswith("chat-") and len(thread_id) > len("chat-")
    assert agent.calls[0][1] == {"configurable": {"thread_id": thread_id}}


def test_each_request_without_thread_id_gets_its_own_session() -> None:
    client = _client(RecordingAgent())
    first = _ask(client, "订单 ORD-2026-0001 什么状态")["thread_id"]
    second = _ask(client, "订单 ORD-2026-0002 什么状态")["thread_id"]
    assert first != second


def test_same_thread_id_continues_the_conversation(tmp_path: Path) -> None:
    """跨请求记忆的证明：第二次的「第 2 轮」只能来自 checkpointer 里的历史。"""
    copy_path = tmp_path / "orders.json"
    shutil.copyfile(data_loader.ORDERS_FILE, copy_path)
    model = StubModel()
    agent = create_customer_service_agent(model=model, checkpointer=InMemorySaver())
    client = _client(agent)

    first = _ask(client, "你好")
    second = _ask(client, "再问一次", first["thread_id"])
    assert first["answer"] == "第 1 轮提问：你好"
    assert second["answer"] == "第 2 轮提问：再问一次"
    assert second["thread_id"] == first["thread_id"]


def test_other_thread_id_starts_a_new_conversation() -> None:
    model = StubModel()
    agent = create_customer_service_agent(model=model, checkpointer=InMemorySaver())
    client = _client(agent)
    _ask(client, "你好", "vip-1")
    assert _ask(client, "你好", "vip-2")["answer"] == "第 1 轮提问：你好"
    assert _ask(client, "你好", "vip-1")["answer"] == "第 2 轮提问：你好"


def test_without_checkpointer_every_request_is_a_fresh_turn() -> None:
    """反向守卫：thread_id 传了但没配 checkpointer 时记忆静默失效，本切片要求调用方注入。"""
    client = _client(create_customer_service_agent(model=StubModel()))
    first = _ask(client, "你好")
    assert _ask(client, "再问一次", first["thread_id"])["answer"] == "第 1 轮提问：再问一次"


# ---------------------------------------------------------------- 参数校验


@pytest.mark.parametrize("payload", INVALID_PAYLOADS, ids=lambda value: json.dumps(value, ensure_ascii=False))
def test_invalid_requests_are_rejected_before_reaching_the_agent(payload: dict[str, Any]) -> None:
    agent = RecordingAgent()
    response = _post(_client(agent), payload)
    assert response.status_code == 422, response.text
    assert agent.calls == [], "非法请求不该消耗一次模型调用"


def test_whitespace_message_is_rejected_with_a_clear_field() -> None:
    response = _post(_client(RecordingAgent()), {"message": "   "})
    errors = response.json()["detail"]
    assert [error["loc"] for error in errors] == [["body", "message"]]


# ---------------------------------------------------------------- 应用启动与生命周期（VS2）


def test_agent_and_checkpointer_are_created_once_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动期装一次，请求内不再装：三个请求共用同一个 Agent 与同一份 checkpointer。

    续轮必须携带 session_token，这是会话归属契约：create_app 注入的 SessionStore
    会把裸 thread_id 的续轮挡在 403，所以先用不带 thread_id 的请求领一份凭据。
    """
    built: list[tuple[Any, Any]] = []

    def fake_factory(model: Any = None, *, checkpointer: Any = None) -> RecordingAgent:
        built.append((model, checkpointer))
        return RecordingAgent(_result(ANSWER))

    monkeypatch.setattr(app_module, "create_customer_service_agent", fake_factory)
    client = TestClient(app_module.create_app())
    thread_id, session_token = _create_session(client)
    assert thread_id.startswith("chat-") and session_token
    for _ in range(2):
        body = _ask(
            client, "订单 ORD-2026-0001 什么状态", thread_id, session_token=session_token
        )
        assert body["answer"] == ANSWER
        assert body["thread_id"] == thread_id, "续轮复用的必须是同一会话，而非又新建一个"

    assert len(built) == 1, "Agent 必须只在应用启动时创建一次"
    assert built[0][0] is None, "默认不注入模型，构建真实模型的动作留在 graph.py"
    assert isinstance(built[0][1], InMemorySaver)


def test_explicit_checkpointer_is_the_one_every_request_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注入的会话存储原样交给 Agent，且不被每请求替换掉。

    续轮同样必须携带 session_token（会话归属契约），本轮要证明的还是 checkpointer 的唯一性。
    """
    saver = InMemorySaver()
    captured: list[Any] = []

    def fake_factory(model: Any = None, *, checkpointer: Any = None) -> RecordingAgent:
        captured.append(checkpointer)
        return RecordingAgent(_result(ANSWER))

    monkeypatch.setattr(app_module, "create_customer_service_agent", fake_factory)
    client = TestClient(app_module.create_app(checkpointer=saver))
    thread_id, session_token = _create_session(client)
    _ask(client, "再问一次", thread_id, session_token=session_token)

    assert captured == [saver]
    assert captured[0] is saver


def test_create_app_hands_the_agent_to_the_route_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """组装只有一条链路：建 Agent → 连同 SessionStore 交给 create_chat_app，不在别处第二次装路由。"""
    seen: list[tuple[Any, Any]] = []
    monkeypatch.setattr(
        app_module,
        "create_customer_service_agent",
        lambda model=None, *, checkpointer=None: "the-agent",
    )
    monkeypatch.setattr(
        app_module,
        "create_chat_app",
        lambda agent, store: seen.append((agent, store)) or "the-app",
    )
    assert app_module.create_app() == "the-app"
    assert [agent for agent, _store in seen] == ["the-agent"]
    assert isinstance(seen[0][1], SessionStore), "路由工厂必须同时收到会话存储"


def test_create_app_always_injects_a_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产入口守卫：create_app 注入的 session store 恒不为 None，封死「无校验路径」。

    create_chat_app 的 ``session_store=None`` 分支仅供内部测试 seam 使用；
    这里先用真实行为兜底：未签发的 thread_id 必被 403 挡住；再从组装点捕获实参。
    """
    client = TestClient(app_module.create_app(model=StubModel()))
    response = client.post(CHAT_URL, json={"message": "你好", "thread_id": "never-issued"})
    assert response.status_code == 403, "生产路径必须强制会话归属校验，未经签发的 thread 不能进 Agent"

    captured: list[Any] = []
    monkeypatch.setattr(
        app_module,
        "create_customer_service_agent",
        lambda model=None, *, checkpointer=None: RecordingAgent(_result(ANSWER)),
    )
    monkeypatch.setattr(
        app_module,
        "create_chat_app",
        lambda agent, store: captured.append(store) or FastAPI(),
    )
    app_module.create_app()
    assert len(captured) == 1 and isinstance(captured[0], SessionStore), (
        "生产入口不得以 session_store=None 组装路由"
    )


def test_created_app_answers_and_keeps_memory_without_a_real_model() -> None:
    """整站离线跑通：create_app(假模型) 出来的应用行为与 VS1 一致。

    第二轮数得出「第 2 轮」同时否证了每请求新建 Agent：
    那样每轮都配一份全新的 InMemorySaver，轮次永远停在第 1 轮。
    续轮携带第一轮签发的 session_token，这是会话归属契约。
    """
    client = TestClient(app_module.create_app(StubModel()))
    first = _ask(client, "你好")
    second = _ask(
        client, "再问一次", first["thread_id"], session_token=first["session_token"]
    )
    assert first["answer"] == "第 1 轮提问：你好"
    assert second["answer"] == "第 2 轮提问：再问一次"
    assert second["thread_id"] == first["thread_id"]


def test_app_module_builds_nothing_at_import_time() -> None:
    """import app.main 不建 Agent、不留全局应用：否则没配 .env 时连导入都会抛错。"""
    source = _code_only(inspect.getsource(app_module))
    assert source.index("create_customer_service_agent(") > source.index("def create_app(")
    assert not hasattr(app_module, "app"), "模块级不留全局 app，起服务由入口脚本显式组装"


# ---------------------------------------------------------------- 失败路径


def test_pending_human_confirmation_is_refused_not_answered() -> None:
    """图停在待人工确认时不能回一条空 answer：客户端会把「没处理」当成「处理完了」。"""
    agent = RecordingAgent({**_result(""), "__interrupt__": [object()]})
    response = _post(_client(agent), {"message": "帮我取消订单 ORD-2026-0001"})
    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "人工确认" in detail and "没有执行" in detail


def test_upstream_failure_becomes_500_instead_of_an_empty_answer() -> None:
    response = _post(_client(BrokenAgent(), raise_server_exceptions=False), {"message": "订单状态"})
    assert response.status_code == 500


def test_cancel_request_is_refused_and_writes_nothing(real_api) -> None:
    """真图证明：取消请求会走到工具与人工确认，被 API 拒绝后订单文件保持原样。"""
    client, model, orders_file = real_api
    order_id = _pending_order_id()
    before = _stored_order(orders_file, order_id)

    response = _post(client, {"message": f"帮我取消订单 {order_id}"})
    assert response.status_code == 501
    # 只被调用一次：模型刚决定调工具就停在确认点上，没有第二次调用去生成答复
    assert len(model.calls) == 1
    assert _stored_order(orders_file, order_id) == before


def test_refused_cancel_does_not_block_the_session(real_api) -> None:
    """实测：501 只是拒掉这一次办理，同一会话之后照常回答，也不会补出第二次办理。

    代价是该线程历史里留下一条没有工具结果的 AI 消息（所以模型数到的是第 2 轮），
    这个会话不再适合继续办理，见 README 已知缺口。
    """
    client, _model, orders_file = real_api
    order_id = _pending_order_id()
    before = _stored_order(orders_file, order_id)

    assert _post(client, {"message": f"帮我取消订单 {order_id}", "thread_id": "vip-1"}).status_code == 501
    assert _ask(client, "你好", "vip-1")["answer"] == "第 2 轮提问：你好"
    assert _post(
        client, {"message": f"再试一次取消 {order_id}", "thread_id": "vip-1"}
    ).status_code == 501
    assert _stored_order(orders_file, order_id) == before


def test_query_turn_on_real_graph_answers_from_tool_result(real_api) -> None:
    """只读工具跑完整链路：answer 是模型依据工具结果给出的答复，工具原文不进响应。"""
    client, model, _orders_file = real_api
    response = client.post(CHAT_URL, json={"message": "订单 ORD-2026-0002 付款了吗"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == TOOL_BASED_ANSWER
    assert body["thread_id"].startswith("chat-")
    # 两次调用：第一次决定查订单，第二次看到 ToolMessage 后作答
    assert len(model.calls) == 2
    for marker in INTERNAL_MARKERS:
        assert marker not in response.text, f"响应泄漏了内部结构：{marker}"


# ---------------------------------------------------------------- 架构守卫


def test_fake_answer_texts_carry_no_internal_marker() -> None:
    """自检：泄漏断言的前提是假答复本身干净，否则测试数据会让检查失效或误伤。"""
    for text in (ANSWER, TOOL_BASED_ANSWER):
        for marker in INTERNAL_MARKERS:
            assert marker not in text, f"测试数据里不该出现 {marker}"


def _business_endpoint_paths(app: FastAPI) -> list[str]:
    return [route.path for route in app.routes if route.path.startswith("/api/")]


def test_api_exposes_only_the_chat_endpoint() -> None:
    """没有会话列表、没有历史查询：这类端点会把 InMemorySaver 里的所有会话暴露出去。"""
    assert _business_endpoint_paths(create_chat_app(RecordingAgent())) == [CHAT_URL]


def test_api_module_never_builds_a_model_or_a_graph() -> None:
    """API 层只接注入的 Agent：自己建模型会让 API 测试打到真实大模型，也会造出全局图。"""
    source = inspect.getsource(chat_module)
    for forbidden in (
        "create_customer_service_agent(",
        "ChatOpenAI(",
        "build_chat_model(",
        "InMemorySaver",
    ):
        assert forbidden not in source, f"API 层不该出现 {forbidden}"
    assert not hasattr(chat_module, "app"), "模块级不要保存全局 app 或全局 agent"


def test_api_entry_script_only_launches() -> None:
    """启动入口的职责边界（VS2）：只解析 --host/--port 并交给 uvicorn。

    import 阶段不建 Agent、不起服务；组装动作也不在这里——否则 Agent 的创建点会有两份，
    一份在脚本、一份在 app/main.py，换 checkpointer 时必然漏掉一处。
    """
    import scripts.chat_api as entry

    source = _code_only(inspect.getsource(entry))
    assert callable(entry.main)
    entry_point = source.index("def main()")
    assert source.index("create_app(") > entry_point
    assert source.index("uvicorn.run(") > entry_point
    for forbidden in ("create_customer_service_agent(", "create_chat_app(", "InMemorySaver"):
        assert forbidden not in source, f"启动入口不该自己组装：{forbidden}"


def test_data_smoke_no_longer_lives_in_the_app_factory() -> None:
    """app/main.py 只做组装；打订单数据的冒烟挪到了 scripts/smoke_data.py。"""
    import scripts.smoke_data as smoke

    assert callable(smoke.main)
    assert "load_orders" in inspect.getsource(smoke)
    app_source = _code_only(inspect.getsource(app_module))
    assert "print(" not in app_source
    assert "load_orders" not in app_source


def test_api_layer_holds_no_business_rules() -> None:
    """业务判定与数据读写都在 Service / Tool 层；API 不 import 它们，也不自己解析中断。"""
    source = inspect.getsource(chat_module)
    for forbidden in (
        "from app.services",
        "import app.services",
        "from app.data_loader",
        "from app.tools",
        "interrupt(",
        "Command(",
        "REFUND_WINDOW_DAYS",
    ):
        assert forbidden not in source, f"API 层不该出现 {forbidden}"


def test_api_reuses_the_shared_text_extractor() -> None:
    """answer 的取值必须走展示层唯一实现，避免第二份 content 解析与它漂移。"""
    assert chat_module.stream_text.__module__ == "app.agent.streaming"


def test_response_contract_fields_follow_create_and_continue() -> None:
    """契约字段清单 + exclude_none 的向后兼容证明。

    新建会话响应有三个字段（answer/thread_id/session_token）；
    续轮响应只剩两个（answer/thread_id）——session_token 为 None 时被
    ``response_model_exclude_none`` 剔除，老客户端看到的响应体形态与 VS1 一致。
    字段清单断言保证以后不会有人再加第四个静默字段。
    """
    assert list(ChatRequest.model_fields) == ["message", "thread_id", "session_token"]
    assert list(ChatResponse.model_fields) == ["answer", "thread_id", "session_token"]

    client = TestClient(create_chat_app(RecordingAgent(), SessionStore()))
    created = client.post(CHAT_URL, json={"message": "你好"})
    assert created.status_code == 200
    created_body = created.json()
    assert set(created_body) == {"answer", "thread_id", "session_token"}
    assert created_body["session_token"], "新建会话必须回显凭据，否则客户端接不上续轮"

    continued = client.post(
        CHAT_URL,
        json={
            "message": "再问一次",
            "thread_id": created_body["thread_id"],
            "session_token": created_body["session_token"],
        },
    )
    assert continued.status_code == 200
    assert set(continued.json()) == {"answer", "thread_id"}
    assert "session_token" not in continued.json(), (
        "续轮不该再回显凭据（哪怕是 null）：exclude_none 必须生效"
    )
