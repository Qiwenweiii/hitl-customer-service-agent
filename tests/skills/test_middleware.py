from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.skills.middleware import SkillMiddleware

SKILL_TEXT = """---
name: refund-processing
description: 测试用退款技能
allowed_tools:
  - query_order
  - refund_order
triggers: [退款, 售后]
priority: 10
---

## Expert SOP

1. 安抚用户情绪。
"""

AFTERSALES_SKILL_TEXT = """---
name: aftersales-basic
description: 低优先级售后技能
allowed_tools:
  - query_order
triggers: [售后]
priority: 1
---

## Expert SOP

基础售后流程。
"""

BROKEN_SKILL_TEXT = """---
name: broken-skill
description: 缺 allowed_tools，应被跳过
---

正文非空但元数据不合法。
"""

SENTINEL = object()


class Recorder:
    def __init__(self):
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest):
        self.requests.append(request)
        return SENTINEL


def _write_skill(root: Path, dir_name: str, text: str) -> Path:
    skill_dir = root / dir_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return root


def _make_skill_root(tmp_path: Path) -> Path:
    return _write_skill(tmp_path / "skills", "refund-processing", SKILL_TEXT)


def _request(
    state: dict,
    tools: list | None = None,
    system_prompt: str | None = "原始系统提示",
) -> ModelRequest:
    return ModelRequest(
        model=None,
        messages=[],
        system_message=SystemMessage(content=system_prompt) if system_prompt is not None else None,
        tools=tools if tools is not None else [],
        state=state,
    )


def test_no_active_skill_passes_request_through(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    tools = [SimpleNamespace(name="query_order")]
    request = _request({"active_skill": None}, tools)
    recorder = Recorder()

    result = middleware.wrap_model_call(request, recorder)

    assert result is SENTINEL
    assert recorder.requests == [request]
    assert recorder.requests[0].system_prompt == "原始系统提示"
    assert recorder.requests[0].tools == tools


def test_active_skill_injects_body_into_system_prompt(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    request = _request({"active_skill": "refund-processing"})
    recorder = Recorder()

    middleware.wrap_model_call(request, recorder)

    merged = recorder.requests[0].system_prompt
    assert merged.startswith("原始系统提示")
    assert "## Expert SOP" in merged
    assert "1. 安抚用户情绪。" in merged
    # 原 request 不被就地修改
    assert request.system_prompt == "原始系统提示"


def test_tools_reordered_to_skill_md_order(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    # request.tools 顺序刻意与 SKILL.md 的 [query_order, refund_order] 不同
    tools = [
        SimpleNamespace(name="refund_order"),
        SimpleNamespace(name="search_knowledge"),
        SimpleNamespace(name="query_order"),
    ]
    request = _request({"active_skill": "refund-processing"}, tools)
    recorder = Recorder()

    middleware.wrap_model_call(request, recorder)

    filtered = recorder.requests[0].tools
    assert [t.name for t in filtered] == ["query_order", "refund_order"]


def test_allowed_tool_missing_from_request_tools_is_skipped(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    # refund_order 在白名单里但不在 request.tools 里，应被跳过且不报错
    tools = [SimpleNamespace(name="query_order")]
    request = _request({"active_skill": "refund-processing"}, tools)
    recorder = Recorder()

    result = middleware.wrap_model_call(request, recorder)

    assert result is SENTINEL
    assert [t.name for t in recorder.requests[0].tools] == ["query_order"]


def test_none_system_prompt_injection_starts_with_non_whitespace(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    request = _request({"active_skill": "refund-processing"}, system_prompt=None)
    recorder = Recorder()

    middleware.wrap_model_call(request, recorder)

    content = recorder.requests[0].system_message.content
    assert not content[0].isspace()
    assert content == SKILL_TEXT.split("---\n\n", 1)[1].strip()


def test_broken_skill_skipped_at_init_others_still_work(tmp_path):
    root = _make_skill_root(tmp_path)
    _write_skill(root, "broken-skill", BROKEN_SKILL_TEXT)

    middleware = SkillMiddleware(skill_root=root)

    assert set(middleware._skills) == {"refund-processing"}
    request = _request({"active_skill": "refund-processing"})
    recorder = Recorder()
    middleware.wrap_model_call(request, recorder)
    assert "## Expert SOP" in recorder.requests[0].system_prompt


def test_missing_skill_dir_falls_back_to_passthrough(tmp_path):
    middleware = SkillMiddleware(skill_root=tmp_path / "no-such-root")
    tools = [SimpleNamespace(name="query_order")]
    request = _request({"active_skill": "not-installed"}, tools)
    recorder = Recorder()

    result = middleware.wrap_model_call(request, recorder)

    assert result is SENTINEL
    assert recorder.requests == [request]


def test_default_skill_root_loads_real_refund_skill():
    middleware = SkillMiddleware()

    assert "refund-processing" in middleware._skills


def test_before_model_trigger_hit_sets_active_skill(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    state = {"messages": [HumanMessage(content="我要申请退款")]}

    update = middleware.before_model(state, runtime=None)

    assert update == {"active_skill": "refund-processing"}


def test_before_model_no_trigger_clears_active_skill(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    state = {"messages": [HumanMessage(content="帮我查一下订单")], "active_skill": "refund-processing"}

    update = middleware.before_model(state, runtime=None)

    assert update == {"active_skill": None}


def test_before_model_non_human_last_message_returns_none(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    ai_state = {"messages": [HumanMessage(content="我要退款"), AIMessage(content="好的，为您办理退款")]}
    tool_state = {"messages": [HumanMessage(content="我要退款"), ToolMessage(content="订单已退款", tool_call_id="t1")]}

    assert middleware.before_model(ai_state, runtime=None) is None
    assert middleware.before_model(tool_state, runtime=None) is None


def test_before_model_empty_messages_returns_none(tmp_path):
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))

    assert middleware.before_model({"messages": []}, runtime=None) is None


def test_before_model_picks_highest_priority_on_conflict(tmp_path):
    root = _make_skill_root(tmp_path)
    _write_skill(root, "aftersales-basic", AFTERSALES_SKILL_TEXT)
    middleware = SkillMiddleware(skill_root=root)
    # "售后" 同时命中两个 skill：refund-processing(priority=10) 应胜过 aftersales-basic(priority=1)
    state = {"messages": [HumanMessage(content="这个售后怎么处理")]}

    update = middleware.before_model(state, runtime=None)

    assert update == {"active_skill": "refund-processing"}


def test_route_then_inject_in_same_chain(tmp_path):
    """钉住链路契约：before_model 写入的 active_skill 在同一次调用链的 wrap_model_call 可见。"""
    middleware = SkillMiddleware(skill_root=_make_skill_root(tmp_path))
    state: dict = {"messages": [HumanMessage(content="我要退款")]}

    update = middleware.before_model(state, runtime=None)
    assert update == {"active_skill": "refund-processing"}
    state.update(update)

    tools = [
        SimpleNamespace(name="refund_order"),
        SimpleNamespace(name="search_knowledge"),
        SimpleNamespace(name="query_order"),
    ]
    request = _request(state, tools, system_prompt="原始系统提示")
    recorder = Recorder()

    middleware.wrap_model_call(request, recorder)

    final_request = recorder.requests[0]
    assert "Expert SOP" in final_request.system_message.content
    assert final_request.system_message.content.startswith("原始系统提示")
    assert [t.name for t in final_request.tools] == ["query_order", "refund_order"]
