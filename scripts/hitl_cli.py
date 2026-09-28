#!/usr/bin/env python
"""订单取消 / 退款的人工确认（Human-in-the-Loop）命令行入口，走真实大模型与完整 Agent 链路。

用法：
  uv run python scripts/hitl_cli.py                     # 每次启动就是一个新会话
  uv run python scripts/hitl_cli.py --session-id vip-1   # 固定会话 id，便于复现同一个上下文

一轮完整流程：
  用户输入（例如"帮我取消订单 ORD-2026-0001"或"ORD-2026-0004 我要退款"）
  → Agent 决策调用 cancel_order / refund_order → Service 检查
  → 符合条件时图暂停，打印待人工确认的订单信息
  → 终端输入 y（批准，真正执行取消／受理退款）／其他（拒绝，订单保持原样）
  → 恢复执行并打印客服最终答复

本脚本只做两件事：把 interrupt 的 payload 展示给人看、把人的回答转成
``Command(resume={"approved": ...})``。业务规则与措辞仍在 Service / Tool 层，
这里不放任何业务判断。

thread_id 就是会话身份：同一次运行里的多轮输入共用一个 thread_id，
因此模型能看到之前几轮，"那它到哪了""帮我取消它"这类指代才接得上；
换一个 thread_id 就是另一段全新对话，两边上下文互不可见。

已知限制：checkpointer 是 InMemorySaver，只在当前进程内有效；
退出脚本后待确认请求和历史消息都会丢失，重新运行即从空上下文开始。
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


def build_config(session_id: str | None) -> dict:
    """会话配置：一个进程一个 thread_id，该会话内的所有轮次与暂停中的确认都挂在它上面。"""
    return {"configurable": {"thread_id": session_id or f"hitl-cli-{uuid.uuid4().hex[:8]}"}}


# 确认提示语按 payload 类型区分；CLI 只负责措辞与读输入，不做任何业务判断。
_CONFIRMATION_PROMPTS = {
    "order_cancellation_confirmation": "确认执行取消？",
    "order_refund_confirmation": "确认受理退款？",
}


def _print_confirmation(payload: object) -> None:
    print("\n[暂停] 需要人工确认：")
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {payload!r}")


def _ask_approved(payload: object) -> bool:
    """人工裁决：只有 y／yes 视为批准，其余输入（含回车、EOF）按拒绝处理。"""
    kind = payload.get("type") if isinstance(payload, dict) else None
    question = _CONFIRMATION_PROMPTS.get(kind, "确认执行本次操作？")
    try:
        answer = input(f"{question}(y=批准 / 其他=拒绝): ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _ask_question() -> str:
    try:
        return input("\n用户: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def run_turn(agent, question: str, config: dict) -> None:
    """跑一轮对话：中途遇到人工确认就暂停等待输入，直到拿到最终答复。"""
    result = agent.invoke({"messages": [HumanMessage(question)]}, config)
    while interrupts := result.get("__interrupt__"):
        _print_confirmation(interrupts[0].value)
        result = agent.invoke(
            Command(resume={"approved": _ask_approved(interrupts[0].value)}), config
        )
    print(f"客服: {result['messages'][-1].content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="客服 Agent 命令行入口（取消订单 / 退款需人工确认）")
    parser.add_argument(
        "--session-id",
        help="会话 id（即 LangGraph thread_id）；不传则每次启动新建一个会话",
    )
    args = parser.parse_args()

    agent = create_customer_service_agent(checkpointer=InMemorySaver())
    config = build_config(args.session_id)
    print("企业智能客服（取消订单 / 退款需人工确认）。输入 exit 退出。")
    print(f"会话: {config['configurable']['thread_id']}（本次运行内的多轮对话共享上下文）")
    while True:
        question = _ask_question()
        if not question or question.lower() in {"exit", "quit"}:
            print("已退出。")
            return
        run_turn(agent, question, config)


if __name__ == "__main__":
    main()
