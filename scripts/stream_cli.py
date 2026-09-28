#!/usr/bin/env python
"""客服 Agent 的流式命令行入口：模型回复逐字上屏，工具调用只显示一句友好状态。

用法：
  uv run python scripts/stream_cli.py                     # 每次启动就是一个新会话
  uv run python scripts/stream_cli.py --session-id vip-1   # 固定会话 id，便于复现同一个上下文

与 scripts/hitl_cli.py 的差别只有一处：本轮回复走
``agent.stream(..., stream_mode="messages")``，边到边打屏，而不是等 ``invoke`` 拿完整消息。
其余链路完全复用现有实现：同一份 checkpointer + 同一个 thread_id、工具层、
cancel_order / refund_order 的人工确认（interrupt）都不改。

屏上只会出现两类东西：
- 模型正文：``AIMessageChunk`` 的文本增量；
- 状态行：按**工具名**查 ``app.agent.streaming.TOOL_STATUS_LABELS`` 得到的
  ``[正在查询订单...]`` 这类提示，文案不由模型决定。

不会上屏的：``ToolMessage`` 原文（订单详情、政策检索片段等是喂给模型的中间数据）、
``tool_calls`` 结构、``metadata``（节点名、模型名、thread_id 等内部字段）。

HITL 在流式下的读法不一样：图暂停时 messages 流只是结束，不会吐中断事件，
所以待确认信息从 ``agent.get_state(config)`` 读，裁决后用 ``Command(resume=...)`` 再开一条流；
恢复流的第一条事件是工具结果，随后的文本增量就是最终答复。

已知限制同 hitl_cli：checkpointer 是 InMemorySaver，只在当前进程内有效，
退出后待确认请求和历史消息都会丢失。
"""

import argparse
import sys
import uuid
from pathlib import Path

# 直接以脚本方式运行时，项目根目录不在 sys.path 上，需要补上才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.agent import create_customer_service_agent  # noqa: E402
from app.agent.streaming import (  # noqa: E402
    pending_confirmation,
    stream_text,
    tool_status_lines,
)


def build_config(session_id: str | None) -> dict:
    """会话配置：一个进程一个 thread_id，该会话内的所有轮次与暂停中的确认都挂在它上面。"""
    return {"configurable": {"thread_id": session_id or f"stream-cli-{uuid.uuid4().hex[:8]}"}}


def stream_once(agent, payload, config: dict) -> None:
    """开一条流跑到底：文本增量直接上屏，工具调用只打状态行。

    状态行必须单独占一行，所以打之前先把还没换行的文本收尾；
    一条流结束也不能留半行在屏上，否则下一轮的 ``用户:`` 提示会接在后面。
    """
    seen: set[str] = set()
    line_open = False
    for message, _metadata in agent.stream(payload, config, stream_mode="messages"):
        text = stream_text(message)
        if text:
            print(text, end="", flush=True)
            line_open = True
        for line in tool_status_lines(message, seen=seen):
            if line_open:
                print(flush=True)
                line_open = False
            print(line, flush=True)
    if line_open:
        print(flush=True)


def _print_confirmation(payload: dict) -> None:
    """中断 payload 原样列给人看：确认要的就是这些字段，不做筛选也不改写。"""
    print("\n[暂停] 需要人工确认：")
    for key, value in payload.items():
        print(f"  {key}: {value}")


def _ask_approved(payload: dict) -> bool:
    """人工裁决：只有 y／yes 视为批准，其余输入（含回车、EOF）按拒绝处理。

    确认话术直接取 payload 里的 question（Tool 层已按业务写好），这里不另存一份，
    避免同一个订单在两处显示不同的确认问题。
    """
    question = payload.get("question") or "确认执行本次操作？"
    try:
        answer = input(f"{question}(y=批准 / 其他=拒绝): ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def run_turn(agent, question: str, config: dict) -> None:
    """跑一轮对话：流式输出答复；图若暂停在人工确认上，读状态、等裁决、再开一条流。"""
    stream_once(agent, {"messages": [HumanMessage(question)]}, config)
    # 一次确认只解开一个中断点；理论上可能连着多个，循环到线程不再等待为止。
    while (payload := pending_confirmation(agent, config)) is not None:
        _print_confirmation(payload)
        stream_once(agent, Command(resume={"approved": _ask_approved(payload)}), config)


def _ask_question() -> str:
    try:
        return input("\n用户: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="客服 Agent 流式命令行入口（回复实时上屏，取消订单 / 退款仍需人工确认）"
    )
    parser.add_argument(
        "--session-id",
        help="会话 id（即 LangGraph thread_id）；不传则每次启动新建一个会话",
    )
    args = parser.parse_args()

    agent = create_customer_service_agent(checkpointer=InMemorySaver())
    config = build_config(args.session_id)
    print("企业智能客服（流式输出；取消订单 / 退款需人工确认）。输入 exit 退出。")
    print(f"会话: {config['configurable']['thread_id']}（本次运行内的多轮对话共享上下文）")
    while True:
        question = _ask_question()
        if not question or question.lower() in {"exit", "quit"}:
            print("已退出。")
            return
        run_turn(agent, question, config)


if __name__ == "__main__":
    main()
