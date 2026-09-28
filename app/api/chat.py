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
"""

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.streaming import stream_text

# 用户没提供会话 id 时由服务端生成；前缀便于在日志与 checkpointer 里分辨来源。
THREAD_ID_PREFIX = "chat-"

# 触发人工确认时的拒绝文案：只能说明"没处理"，不得顺着用户的意图给出办理结果。
HITL_NOT_SUPPORTED_DETAIL = (
    "本次请求触发了需要人工确认的操作（取消订单／受理退款），"
    "当前 API 版本不支持处理人工确认，因此该操作没有执行，订单数据未被改动。"
    "请使用人工确认入口（scripts/hitl_cli.py）办理，或改用查询类问题。"
)


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


def create_chat_app(agent: Any) -> FastAPI:
    """用注入的 Agent 构建 API 应用。

    Args:
        agent: 可 ``invoke`` 的客服 Agent（``create_customer_service_agent`` 的返回值），
            测试可注入假 Agent。调用方负责给它配好 checkpointer，
            否则 ``thread_id`` 不生效：每轮都是全新上下文。

    Returns:
        暴露 ``POST /api/chat`` 的 FastAPI 应用。
    """
    app = FastAPI(
        title="Customer Service Agent API",
        version="0.1.0",
        description="同步对话接口（VS1）：不含流式、人工确认与认证。",
    )

    # 处理函数用普通 def：Agent 的 invoke 是阻塞调用，交给 FastAPI 的线程池跑，
    # 不伪装成 async（图本身没有 await 能力）。
    @app.post("/api/chat", response_model=ChatResponse, summary="同步对话（一轮问答）")
    def chat(payload: ChatRequest) -> ChatResponse:
        thread_id = payload.thread_id or new_thread_id()
        result = agent.invoke(
            {"messages": [HumanMessage(payload.message)]},
            build_config(thread_id),
        )
        if result.get("__interrupt__"):
            raise HTTPException(status_code=501, detail=HITL_NOT_SUPPORTED_DETAIL)
        return ChatResponse(answer=_answer_of(result), thread_id=thread_id)

    return app
