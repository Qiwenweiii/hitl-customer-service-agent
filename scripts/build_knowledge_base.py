#!/usr/bin/env python
"""知识库构建入口：把 data/knowledge/ 下的文档向量化并持久化到 Chroma。

用法：uv run python scripts/build_knowledge_base.py
可重复执行：每个源文档先删除自己的旧 chunk 再写入，不会产生重复数据。
"""

import sys
from pathlib import Path

# 直接以脚本方式运行时，项目根目录不在 sys.path 上，需要补上才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.knowledge_base import build_knowledge_base  # noqa: E402


def main() -> None:
    stats = build_knowledge_base()
    print(
        f"知识库构建完成：{stats['files']} 个文档 -> {stats['chunks']} 个 chunk"
        f"（替换旧 chunk {stats['replaced']} 个）"
    )
    print(f"Chroma 路径：{stats['persist_dir']}  collection：{stats['collection']}")
    print(f"复核 collection 实际向量数：{stats['vectors']}")


if __name__ == "__main__":
    main()
