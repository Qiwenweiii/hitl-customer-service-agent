"""测试基础设施守卫：强制关闭 LangSmith/LangChain 追踪上报。

开启追踪时，langsmith 在每次 run 开始前会序列化输入上报，而 GenericFakeChatModel 的
`messages` 是一个迭代器——序列化过程会把它遍历排空，等模型节点真正执行时 next() 必然落空，
成片假红 `RuntimeError: generator raised StopIteration`（根因分析见《项目复盘笔记》6.10）。
追踪只是"看了一眼输入"却改掉了测试核心的消费进度，因此测试必须对环境里的追踪开关免疫，
而不是依赖每条命令手动带 LANGSMITH_TRACING=false 前缀。

CLI / eval/runner.py 不受此文件影响：真实模型下追踪上报无害，想观测仍可在 .env 里开启。
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_TRACING", None)
