"""Skill 注入与触发路由 middleware。

构造期预加载 skill_root 下全部 SKILL.md；运行期不读磁盘、不调 LLM：
- before_model（阶段 2 路由）：仅当最后一条消息是 HumanMessage 时，按 triggers
  规则命中写 active_skill；未命中显式清为 None；其他消息类型不动 state。
- wrap_model_call（阶段 1 注入）：查表注入 system_prompt 并按白名单重排 tools；
  skill.body 不进 messages 历史。
单个文件校验失败只跳过该 skill，不阻断其他 skill，也不抛异常。
**skill 内容或增删需重启进程才能生效，这是有意的设计选择。**
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    Runtime,
)
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .exceptions import SkillValidationError
from .loader import Skill, load_skill

SKILL_FILENAME = "SKILL.md"


class SkillState(AgentState):
    active_skill: str | None


def _tool_name(tool: object) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    return getattr(tool, "__name__", "")


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "".join(parts)


class SkillMiddleware(AgentMiddleware[SkillState]):
    """构造期预加载 skill_root 下全部 SKILL.md（以 frontmatter name 为键）；
    运行期仅查表注入。skill 变更需重启进程生效，这是有意的设计选择。
    """

    state_schema = SkillState

    def __init__(self, skill_root: Path | None = None) -> None:
        super().__init__()
        self.skill_root = (
            skill_root if skill_root is not None else Path(__file__).resolve().parents[2] / "skills"
        )
        self._skills: dict[str, Skill] = {}
        for path in sorted(self.skill_root.glob(f"*/{SKILL_FILENAME}")):
            try:
                skill = load_skill(path)
            except SkillValidationError:
                continue
            self._skills[skill.name] = skill
        self._route_order: list[Skill] = sorted(
            self._skills.values(), key=lambda s: (-s.priority, s.name)
        )

    def before_model(
        self, state: SkillState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None
        text = _message_text(last)
        for skill in self._route_order:
            if any(trigger in text for trigger in skill.triggers):
                return {"active_skill": skill.name}
        return {"active_skill": None}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        skill_name = request.state.get("active_skill")
        if not skill_name:
            return handler(request)
        skill = self._skills.get(skill_name)
        if skill is None:
            return handler(request)
        base = request.system_prompt or ""
        merged = "\n\n".join(part for part in (base, skill.body) if part)
        tool_map = {_tool_name(tool): tool for tool in request.tools}
        filtered = [tool_map[name] for name in skill.allowed_tools if name in tool_map]
        return handler(
            request.override(
                system_message=SystemMessage(content=merged),
                tools=filtered,
            )
        )
