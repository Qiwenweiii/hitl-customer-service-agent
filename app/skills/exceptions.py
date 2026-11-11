"""Skill 数据层的自定义异常。"""


class SkillValidationError(ValueError):
    """SKILL.md 缺失、frontmatter 格式错误或字段校验不通过时抛出。"""
