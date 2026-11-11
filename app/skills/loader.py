from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .exceptions import SkillValidationError

_DELIMITER = "---"


@dataclass
class Skill:
    name: str
    description: str
    allowed_tools: list[str]
    priority: int
    body: str
    triggers: list[str] = field(default_factory=list)


def load_skill(path: Path) -> Skill:
    if not path.is_file():
        raise SkillValidationError(f"skill 文件不存在: {path}")
    return _parse(path.read_text(encoding="utf-8"), source=str(path))


def _split_frontmatter(text: str, source: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _DELIMITER:
        raise SkillValidationError(f"{source}: 文件必须以 '---' frontmatter 起始行开头")
    for i in range(1, len(lines)):
        if lines[i].strip() == _DELIMITER:
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:]).strip()
    raise SkillValidationError(f"{source}: frontmatter 缺少结束的 '---' 分隔行")


def _parse(text: str, source: str) -> Skill:
    meta_raw, body = _split_frontmatter(text, source)

    try:
        meta = yaml.safe_load(meta_raw)
    except yaml.YAMLError as exc:
        raise SkillValidationError(f"{source}: frontmatter YAML 解析失败: {exc}") from exc
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise SkillValidationError(f"{source}: frontmatter 必须是键值映射，实际为 {type(meta).__name__}")

    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SkillValidationError(f"{source}: name 必须是非空字符串")

    description = meta.get("description")
    if description is None:
        description = ""
    elif not isinstance(description, str):
        raise SkillValidationError(f"{source}: description 必须是字符串")

    tools_raw = meta.get("allowed_tools")
    if not isinstance(tools_raw, list) or not tools_raw:
        raise SkillValidationError(f"{source}: allowed_tools 必须是非空列表")
    allowed_tools: list[str] = []
    for tool in tools_raw:
        if not isinstance(tool, str) or not tool.strip():
            raise SkillValidationError(f"{source}: allowed_tools 含非字符串或空白项: {tool!r}")
        if tool not in allowed_tools:
            allowed_tools.append(tool)

    priority = meta.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise SkillValidationError(f"{source}: priority 必须是整数，实际为 {priority!r}")

    triggers_raw = meta.get("triggers")
    if triggers_raw is None:
        triggers: list[str] = []
    elif not isinstance(triggers_raw, list):
        raise SkillValidationError(f"{source}: triggers 必须是列表，实际为 {type(triggers_raw).__name__}")
    else:
        triggers = []
        for trigger in triggers_raw:
            if not isinstance(trigger, str) or not trigger.strip():
                raise SkillValidationError(f"{source}: triggers 含非字符串或空白项: {trigger!r}")
            if trigger not in triggers:
                triggers.append(trigger)

    if not body:
        raise SkillValidationError(f"{source}: 正文（frontmatter 之后）不能为空")

    return Skill(
        name=name.strip(),
        description=description,
        allowed_tools=allowed_tools,
        priority=priority,
        body=body,
        triggers=triggers,
    )
