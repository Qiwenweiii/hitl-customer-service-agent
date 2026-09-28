"""应用组装：把 Agent、Checkpointer 与 HTTP 路由装成一个可启动的 FastAPI 应用。

一个进程里只有这里决定「用哪个模型、用哪种会话存储、挂哪些路由」：

- Agent 与 checkpointer 都在 ``create_app`` 里创建一次，装进应用后由所有请求共用；
  路由模块（``app/api/chat.py``）只接注入的 agent，既不建模型也不建 checkpointer。
- 不引入依赖注入容器：参数就 ``model`` 与 ``checkpointer`` 两个，
  测试传一个假模型进来，整站（含路由与 checkpointer）就能离线跑通。

启动入口在 ``scripts/chat_api.py``（只解析 --host/--port 并交给 uvicorn）。
原先放在本模块的订单冒烟脚本挪到了 ``scripts/smoke_data.py``：本模块只承担应用组装。
"""

from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import create_customer_service_agent
from app.api import create_chat_app


def create_app(model: BaseChatModel | None = None, *, checkpointer=None) -> FastAPI:
    """组装 Chat API 应用。

    Args:
        model: 注入的对话模型；为空则按 ``.env`` 构建真实模型（建模型的动作在
            ``app/agent/graph.py``，本模块不碰）。
        checkpointer: 会话存储；为空则用 ``InMemorySaver``，
            即历史与待人工确认状态只在当前进程内有效，重启全部丢失。
            换成持久化 checkpointer 只需要改这一个入参。

    Returns:
        暴露 ``POST /api/chat`` 的 FastAPI 应用，内部已装好唯一的 Agent 实例。
    """
    agent = create_customer_service_agent(
        model,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
    return create_chat_app(agent)
