"""同步 Chat API：把现有客服 Agent 包成一个 HTTP 端点，只做「HTTP 契约 ↔ Agent 契约」的翻译。

分层沿用项目约定：
- 这里只有请求/响应的 Pydantic 契约、``thread_id`` 的组装，以及把 Agent 结果翻译成响应；
- 业务规则仍在 Service / Tool 层：API 不判断能不能取消、该不该退款，也不复述订单字段；
- 正文取值复用展示层唯一的实现 ``app.agent.streaming.stream_text``，不再写第二份 content 解析；
- 与 ``graph.py`` 一样不保存全局 Agent 或全局 checkpointer：图由 ``create_chat_app(agent)`` 注入，
  所以 API 层测试可以塞进假 Agent，不需要真实大模型。

``thread_id`` 就是会话身份，原样透传给 LangGraph 的 ``configurable``：
同一个 ``thread_id`` 的多次请求共享短期记忆（以及待人工确认的状态），不同线程互不可见。
客户端没带 ``thread_id`` 时服务端新建一个并在响应里回显，下一轮带上它即可续上同一会话。

本版本（VS1）刻意没有的能力：流式输出、人工确认（HITL）、认证、前端。
其中人工确认是唯一需要显式拒绝的：用户可能在 ``message`` 里直接要求取消订单，
此时图会停在待确认状态，而本端点没有回答确认问题的能力，因此返回 501 明确拒绝，
而不是回一条空 ``answer`` 让客户端以为处理过了。

拿到 501 之后会怎样（实测，见 ``tests/test_chat_api_slice.py``）：这一次不会写任何数据，
同一会话之后的普通请求照常回答；代价是该线程留下一个待确认中断和一条没有工具结果的 AI 消息。
所以客户端收到 501 应该改用人工确认入口（``scripts/hitl_cli.py``）或换一个全新会话，
不要在同一段对话里反复重试同一笔办理。

VS3 在同一条 ``POST /api/chat`` 上加了 SSE 流式，靠 ``Accept`` 头协商而不是新增字段或新端点：
带 ``Accept: text/event-stream`` 才走流式，其余（含 ``*/*`` 与不带该头的老客户端）照旧返回 JSON。
流式只有三类事件：``token``（``{"content": "增量文本"}``）、``done``（``{}``，正常收尾）、
``error``（``{"message": "..."}``，终止且后面不再有 done）。客户端把依次到达的 token 正文拼起来即为完整答复。
``thread_id`` 生成规则与同步路径一致，并通过响应头 ``X-Thread-Id`` 回显（SSE 正文里没有承载它的字段）。
Agent 侧出错时只回一句固定话术，异常原文与 traceback 留在服务端日志，不进响应。
HITL 仍然不办理：``stream_mode="messages"`` 不会吐中断事件，流只是安静结束，
因此本轮没有任何文本增量且线程确实停在待确认点时，改发 ``error`` 事件说明「这一笔没有办理」，
不回 ``done``，避免客户端把「没处理」读成「答完了」。
"""

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.streaming import pending_confirmation, stream_answer, stream_text

logger = logging.getLogger(__name__)

# 用户没提供会话 id 时由服务端生成；前缀便于在日志与 checkpointer 里分辨来源。
THREAD_ID_PREFIX = "chat-"

# 触发人工确认时的拒绝文案：只能说明"没处理"，不得顺着用户的意图给出办理结果。
HITL_NOT_SUPPORTED_DETAIL = (
    "本次请求触发了需要人工确认的操作（取消订单／受理退款），"
    "当前 API 版本不支持处理人工确认，因此该操作没有执行，订单数据未被改动。"
    "请使用人工确认入口（scripts/hitl_cli.py）办理，或改用查询类问题。"
)

# SSE 的媒体类型：客户端在 Accept 里显式带上它才走流式。
MEDIA_TYPE_EVENT_STREAM = "text/event-stream"
# 流式响应没有承载会话 id 的字段，新建的 thread_id 用这个响应头回显给客户端。
THREAD_ID_HEADER = "X-Thread-Id"
# 流式出错时给客户端的一句话：异常原文可能含模型名、端点地址等内部信息，不进响应。
STREAM_ERROR_MESSAGE = "服务处理本轮请求时出错，本轮答复已中断，请稍后重试。"


class ChatRequest(BaseModel):
    """一轮对话的请求体。

    ``extra="forbid"``：未知字段一律拒绝，不静默忽略。否则客户端可能以为
    自己传的 ``approved=true``／``stream=true`` 之类的字段生效了。
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        ...,
        min_length=1,
        description="用户这轮要问的内容，例如「订单 ORD-2026-0001 现在什么状态」",
    )
    thread_id: str | None = Field(
        default=None,
        description="会话 id（即 LangGraph thread_id）。省略则新建会话，服务端在响应里回显该 id。",
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message 不能为空或只包含空白字符")
        return stripped

    @field_validator("thread_id")
    @classmethod
    def _reject_blank_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("thread_id 不能是空白字符串；需要新会话时请省略该字段")
        return stripped


class ChatResponse(BaseModel):
    """一轮对话的响应体：只有面向用户的答复与会话 id，不含工具调用等内部结构。"""

    answer: str = Field(description="客服答复正文，取自本轮最后一条 AI 消息")
    thread_id: str = Field(description="本轮使用的会话 id，下一轮请求带上它即可延续上下文")


def new_thread_id() -> str:
    """生成一个新的会话 id。"""
    return f"{THREAD_ID_PREFIX}{uuid.uuid4().hex[:12]}"


def build_config(thread_id: str) -> dict:
    """Agent 运行配置：``thread_id`` 决定记忆与待确认状态挂在哪个会话上。"""
    return {"configurable": {"thread_id": thread_id}}


def _answer_of(result: Any) -> str:
    """取本轮最终答复正文。"""
    return stream_text(result["messages"][-1])


def _sse_event(event: str, data: dict) -> str:
    """一条 SSE 事件：``event:`` 行 + ``data:`` 行 + 空行分帧。

    data 走 ``json.dumps(ensure_ascii=False)``：正文里的换行被转义成 ``\\n``，
    一条事件因此永远只占一行 ``data:``，不会把 SSE 的分帧撞断；中文也不变成 ``\\uXXXX``。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _wants_event_stream(request: Request) -> bool:
    """只有显式要求 ``text/event-stream`` 才走流式。

    不能把 ``Accept: */*``（curl 与多数 HTTP 客户端的默认值）也算成流式请求，
    否则 VS3 会悄悄改变所有既有客户端拿到的响应形态。
    """
    return MEDIA_TYPE_EVENT_STREAM in (request.headers.get("accept") or "")


def _stream_events(agent: Any, message: str, config: dict) -> Iterator[str]:
    """把 Agent 的文本增量转成 SSE 事件序列：``token``* 然后 ``done``；出错只发 ``error``。"""
    produced = False
    try:
        for text in stream_answer(agent, {"messages": [HumanMessage(message)]}, config):
            produced = True
            yield _sse_event("token", {"content": text})
        if not produced and pending_confirmation(agent, config) is not None:
            # 图停在待确认点时 messages 流只是结束，不吐中断事件：
            # 回 done 等于告诉客户端「答完了」，必须改口说明这一笔没办理。
            yield _sse_event("error", {"message": HITL_NOT_SUPPORTED_DETAIL})
            return
        yield _sse_event("done", {})
    except Exception:
        # 异常原文与 traceback 只进服务端日志；响应头此时已发出，改不了状态码，只能发 error 事件。
        logger.exception("流式请求处理失败，已向客户端发送 error 事件")
        yield _sse_event("error", {"message": STREAM_ERROR_MESSAGE})


def create_chat_app(agent: Any) -> FastAPI:
    """用注入的 Agent 构建 API 应用。

    Args:
        agent: 客服 Agent（``create_customer_service_agent`` 的返回值）。同步分支用 ``invoke``，
            流式分支用 ``stream(stream_mode="messages")``，只在「本轮没有产出文本」时才读 ``get_state``。
            测试可注入假 Agent，三条路径分别打桩即可，不需要真实大模型。
            调用方负责给它配好 checkpointer，否则 ``thread_id`` 不生效：每轮都是全新上下文。

    Returns:
        暴露 ``POST /api/chat`` 的 FastAPI 应用。
    """
    app = FastAPI(
        title="Customer Service Agent API",
        version="0.1.0",
        description="对话接口：默认返回 JSON；带 Accept: text/event-stream 时返回 SSE 流。"
        "不含人工确认与认证。",
    )

    # 处理函数用普通 def：Agent 的 invoke 是阻塞调用，交给 FastAPI 的线程池跑，
    # 不伪装成 async（图本身没有 await 能力）。
    # 返回 StreamingResponse 时它是个 Response 实例，FastAPI 不会再用 response_model 去校验它，
    # 所以 response_model 保留为同步分支的 JSON 契约。
    @app.post("/api/chat", response_model=ChatResponse, summary="对话（一轮问答，JSON 或 SSE 流）")
    def chat(payload: ChatRequest, request: Request) -> ChatResponse | StreamingResponse:
        thread_id = payload.thread_id or new_thread_id()
        config = build_config(thread_id)
        if _wants_event_stream(request):
            return StreamingResponse(
                _stream_events(agent, payload.message, config),
                media_type=MEDIA_TYPE_EVENT_STREAM,
                headers={THREAD_ID_HEADER: thread_id, "Cache-Control": "no-cache"},
            )
        result = agent.invoke({"messages": [HumanMessage(payload.message)]}, config)
        if result.get("__interrupt__"):
            raise HTTPException(status_code=501, detail=HITL_NOT_SUPPORTED_DETAIL)
        return ChatResponse(answer=_answer_of(result), thread_id=thread_id)

    return app
