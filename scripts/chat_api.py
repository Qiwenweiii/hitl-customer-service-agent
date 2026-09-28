#!/usr/bin/env python
"""同步 Chat API 启动入口：解析 --host/--port，把应用交给 uvicorn。

用法：
  uv run python scripts/chat_api.py                       # 默认 127.0.0.1:8000
  uv run python scripts/chat_api.py --host 0.0.0.0 --port 9000

冒烟（第一轮不传 thread_id，服务端会新建一个并在响应里回显）：
  curl -s -X POST http://127.0.0.1:8000/api/chat \\
    -H 'Content-Type: application/json' \\
    -d '{"message": "订单 ORD-2026-0001 现在什么状态"}'

  # 第二轮带上上一轮返回的 thread_id，就能接着同一个会话追问
  curl -s -X POST http://127.0.0.1:8000/api/chat \\
    -H 'Content-Type: application/json' \\
    -d '{"message": "那它到哪了", "thread_id": "上一轮返回的 id"}'

接口文档：启动后打开 http://127.0.0.1:8000/docs

职责边界：本脚本只解析参数与起服务。
Agent 与 Checkpointer 的创建在 ``app/main.py`` 的 ``create_app()``——一次启动装一个应用，
所有请求共用同一个图；HTTP 契约在 ``app/api/chat.py``；业务规则在 Service / Tool 层。

已知限制：
- 默认 checkpointer 是 InMemorySaver，只在当前进程内有效，重启服务后历史与待确认状态全部丢失；
  需要跨进程持久的会话，换掉 ``create_app(checkpointer=...)`` 的默认值即可。
- 取消订单／受理退款会触发人工确认，本版本端点不处理，会返回 501（详见 app/api/chat.py）。
- 没有认证、没有速率限制，只适合本地开发调试，不要直接暴露到公网。
- 传的是应用实例而不是导入串，所以不支持 uvicorn ``--reload``；改代码请手动重启。
"""

import argparse
import sys
from pathlib import Path

# 直接以脚本方式运行时，项目根目录不在 sys.path 上，需要补上才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from app.main import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="客服 Agent 同步 Chat API（POST /api/chat）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    args = parser.parse_args()

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
