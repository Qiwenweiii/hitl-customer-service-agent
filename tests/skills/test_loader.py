from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.skills.exceptions import SkillValidationError
from app.skills.loader import Skill, load_skill

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID = """---
name: refund-processing
description: 退款处理 SOP
allowed_tools:
  - query_order
  - search_knowledge
  - refund_order
priority: 10
---

## Expert SOP

1. 安抚用户情绪。

## Constraints

- 禁止承诺具体到账时间。
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid_skill(tmp_path):
    skill = load_skill(_write(tmp_path, VALID))

    assert isinstance(skill, Skill)
    assert skill.name == "refund-processing"
    assert skill.description == "退款处理 SOP"
    assert skill.allowed_tools == ["query_order", "search_knowledge", "refund_order"]
    assert skill.priority == 10
    assert skill.body.startswith("## Expert SOP")
    assert "## Constraints" in skill.body


def test_missing_name_raises(tmp_path):
    text = VALID.replace("name: refund-processing\n", "")
    with pytest.raises(SkillValidationError, match="name"):
        load_skill(_write(tmp_path, text))


def test_empty_allowed_tools_raises(tmp_path):
    text = VALID.replace(
        "allowed_tools:\n  - query_order\n  - search_knowledge\n  - refund_order\n",
        "allowed_tools: []\n",
    )
    with pytest.raises(SkillValidationError, match="allowed_tools"):
        load_skill(_write(tmp_path, text))


def test_empty_body_raises(tmp_path):
    text = VALID.split("---\n\n## Expert SOP")[0] + "---\n\n"
    with pytest.raises(SkillValidationError, match="正文"):
        load_skill(_write(tmp_path, text))


def test_duplicate_allowed_tools_deduped(tmp_path):
    text = VALID.replace(
        "allowed_tools:\n  - query_order\n  - search_knowledge\n  - refund_order\n",
        "allowed_tools:\n  - query_order\n  - refund_order\n  - query_order\n",
    )
    skill = load_skill(_write(tmp_path, text))

    assert skill.allowed_tools == ["query_order", "refund_order"]


def test_triggers_missing_defaults_to_empty_list(tmp_path):
    skill = load_skill(_write(tmp_path, VALID))

    assert skill.triggers == []


def test_triggers_not_a_list_raises(tmp_path):
    text = VALID.replace("priority: 10\n", "priority: 10\ntriggers: 退款\n")
    with pytest.raises(SkillValidationError, match="triggers"):
        load_skill(_write(tmp_path, text))


def test_triggers_non_string_item_raises(tmp_path):
    text = VALID.replace("priority: 10\n", "priority: 10\ntriggers:\n  - 退款\n  - 3\n")
    with pytest.raises(SkillValidationError, match="triggers"):
        load_skill(_write(tmp_path, text))


def test_triggers_dedupe_preserves_order(tmp_path):
    text = VALID.replace(
        "priority: 10\n",
        "priority: 10\ntriggers:\n  - 退款\n  - 售后\n  - 退款\n  - 退货\n",
    )
    skill = load_skill(_write(tmp_path, text))

    assert skill.triggers == ["退款", "售后", "退货"]


def test_real_skill_md_has_triggers():
    skill = load_skill(REPO_ROOT / "skills" / "refund-processing" / "SKILL.md")

    assert "退款" in skill.triggers
    assert "可以退吗" in skill.triggers
    assert len(skill.triggers) == len(set(skill.triggers))


def test_skills_modules_do_not_import_app():
    # 允许 app.skills.exceptions（同包异常定义）；其余 app.* 一律禁止。
    allowed = "app.skills.exceptions"
    violations = []
    for py_file in sorted((REPO_ROOT / "app" / "skills").rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            for module in modules:
                if module.split(".")[0] == "app" and module != allowed:
                    violations.append(f"{py_file.name}: import {module}")

    assert violations == []
