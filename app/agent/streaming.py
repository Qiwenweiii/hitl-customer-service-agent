"""流式事件的展示层：把 ``stream(stream_mode="messages")`` 吐出的原始消息转成可读文本。

这里只有四件事：
1. 从消息里取「可以给人看的正文」（``stream_text``）；
2. 把工具调用翻译成一句状态提示（``tool_status_lines``），文案由**工具名**查表得到；
3. 读出「正在等人工确认」的 payload（``pending_confirmation``）；
4. 把一条 Agent 流跑到底、只产出文本增量（``stream_answer``），供只要正文的消费端使用。

刻意不放的东西：不接模型输出决定展示文案，不含任何业务规则，不做 IO（不 print、不 input），
也不碰 metadata；``stream_answer`` 同样不认识 HTTP／SSE，事件分帧与编码留在 API 层。
业务判定仍在 Service / Tool 层，展示口径只在 ``TOOL_STATUS_LABELS`` 这一份表里。
"""

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage

# 工具名 -> 状态提示。文案必须是程序写死的：模型只负责决定"调哪个工具"，
# 不允许它决定"给用户看什么进度文案"，否则状态提示会变成可被诱导输出的内容。
TOOL_STATUS_LABELS: dict[str, str] = {
    "query_order": "[正在查询订单...]",
    "query_logistics": "[正在查询物流...]",
    "search_knowledge": "[正在查询相关政策...]",
    "cancel_order": "[正在检查订单是否可以取消...]",
    "refund_order": "[正在处理退款请求...]",
}

# 未登记的工具（以后加工具时忘了同步上表）也只能给出这句通用提示，不回显原始名字。
UNKNOWN_TOOL_STATUS = "[正在处理...]"


def stream_text(message: Any) -> str:
    """取这条事件里可展示的正文；没有就返回空串。

    ``stream_mode="messages"`` 不只吐 ``AIMessageChunk``：工具执行完会吐 ``ToolMessage``，
    此外还可能有 HumanMessage / SystemMessage。它们的 content 是喂给模型的中间数据
    （订单详情、政策原文、内部指令），直接打到终端等于把原始工具结果甩给用户，
    因此非 AI 消息一律不展示。

    判定只能靠类型：``AIMessage.type`` 是 ``"ai"``，而流式分片 ``AIMessageChunk.type``
    是 ``"AIMessageChunk"``，按 ``== "ai"`` 比字符串会把整条流的正文全丢掉。
    """
    if not isinstance(message, AIMessage):
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    # 内容块形态（多模态 / v1 输出）：只挑文本块，图片、思考链等其他块不展示。
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _calls_of(message: Any) -> list[dict]:
    """这条事件里的工具调用片段。

    流式分片优先看 ``tool_call_chunks``（参数是残缺 JSON，不能解析）；
    模型不支持流式时 messages 模式只吐一条完整的 ``AIMessage``，那时只有 ``tool_calls``。
    两条路径都走这里，调用方不必区分消息类型。
    """
    chunks = getattr(message, "tool_call_chunks", None) or []
    if chunks:
        return list(chunks)
    return list(getattr(message, "tool_calls", None) or [])


def tool_status_lines(message: Any, *, seen: set[str]) -> list[str]:
    """把这条事件里的工具调用翻译成状态提示行，并就地记录已提示过的调用以去重。

    一次工具调用会被拆成多个分片下发（首个分片带工具名，后续分片只有参数碎片），
    因此只有「第一次见到、且已经拿到工具名」的那个分片会产生提示行。
    ``seen`` 由调用方按每条流单独持有：它是一条流内的去重记忆，不是跨轮状态。
    """
    lines: list[str] = []
    for call in _calls_of(message):
        name = call.get("name")
        if not name:
            continue
        key = call.get("id") or f"index:{call.get('index')}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(TOOL_STATUS_LABELS.get(name, UNKNOWN_TOOL_STATUS))
    return lines


def pending_confirmation(agent: Any, config: dict) -> dict | None:
    """读出该线程当前等待人工确认的 payload；没有暂停中的确认则返回 None。

    为什么不能像 ``invoke`` 那样读返回值里的 ``__interrupt__``：
    messages 模式不会把中断当事件吐出来，图暂停时流只是安静地结束，
    待确认信息只存在于 checkpointer 里的线程状态（``state.tasks[*].interrupts``）。
    本函数只读状态、不做任何裁决，批准与否仍由调用方（人）决定。
    """
    state = agent.get_state(config)
    for task in getattr(state, "tasks", ()) or ():
        for interruption in getattr(task, "interrupts", ()) or ():
            return interruption.value
    return None


def stream_answer(agent: Any, payload: Any, config: dict) -> Iterator[str]:
    """跑一条 Agent 流，只产出可展示的文本增量，空增量一律不发。

    与 ``scripts/stream_cli.py`` 里那个循环不合并：CLI 还要穿插状态行并管理换行，
    而只要正文的消费端（HTTP/SSE）不需要。两边共用同一个 ``stream_text`` 取值口径，
    所以「哪些消息能给人看」这件事仍然只有一份判定。

    这里不认识 SSE／HTTP：不拼事件名、不管分帧与编码，产出的是纯 str 增量。
    """
    for message, _metadata in agent.stream(payload, config, stream_mode="messages"):
        text = stream_text(message)
        if text:
            yield text
