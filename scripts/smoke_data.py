#!/usr/bin/env python
"""数据层冒烟：确认配置与 Mock 订单数据读得通，不需要大模型。

用法：
  uv run python scripts/smoke_data.py

原来这段冒烟在 ``app/main.py`` 里；VS2 把 ``app/main.py`` 收给应用组装（``create_app``），
它挪到这里，避免「main.py 到底是起服务还是打数据」的歧义。
要起 HTTP 服务请用 ``scripts/chat_api.py``。
"""

import sys
from pathlib import Path

# 直接以脚本方式运行时，项目根目录不在 sys.path 上，需要补上才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import KNOWLEDGE_DIR, LANGSMITH_PROJECT  # noqa: E402
from app.data_loader import load_orders  # noqa: E402


def main() -> None:
    orders = load_orders()
    print(f"项目: {LANGSMITH_PROJECT}")
    print(f"知识库目录: {KNOWLEDGE_DIR}")
    print(f"共加载 {len(orders)} 个订单:")
    for order in orders:
        print(f"  {order.order_id} | {order.product_name} | {order.amount:.2f} | {order.status}")


if __name__ == "__main__":
    main()
