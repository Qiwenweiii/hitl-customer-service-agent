# customer-service-agent

企业智能客服 Agent：基于 LangChain / LangGraph，打通「用户提问 → 工具选择 → 业务判定 → 数据写回 → 回答」闭环，
高风险动作（取消订单、受理退款）通过 LangGraph `interrupt` 暂停等待人工确认。

## 当前进度

已完成的能力切片（每一片都是 Tool → Service → data_loader → Mock JSON 的完整纵切）：

- **项目骨架**：uv + Python 3.12、Pydantic 模型、Mock 数据与配置。
- **订单查询**：`query_order`。
- **物流查询**：`query_logistics`（支持一单多包裹）。
- **RAG 知识库**：`data/knowledge/` 政策文档 → 切分 → Embedding → Chroma 持久化 → 向量召回 + CrossEncoder 重排 → `search_knowledge`。
- **LangGraph Agent**：系统提示约束工具选择与「以工具结果为准、禁止编造」，绑定上述 5 个工具。
- **HITL 人工确认**：`cancel_order`（取消订单）与 `refund_order`（受理退款），含命令行入口 `scripts/hitl_cli.py`。
- **短期多轮对话（Conversation Memory）**：同一个 `thread_id` 内共享上下文，"那它到哪了""帮我取消它"能接上上文那笔订单；
  换一个 `thread_id` 就是另一段全新对话，线程之间互不可见。载体只有注入的 checkpointer，不额外引入存储机制。
- **CLI 流式输出（Streaming）**：`scripts/stream_cli.py` 用 `agent.stream(stream_mode="messages")` 把模型回复逐字上屏，
  工具调用只显示按工具名映射的状态行（`[正在查询订单...]`），不打印工具结果原文与内部结构；
  Memory / `thread_id` / HITL 全部沿用，见[流式输出（CLI Streaming）](#流式输出cli-streaming)。
- **同步 Chat API**：`app/api/chat.py` 用 FastAPI 暴露 `POST /api/chat`，请求 `{"message", "thread_id"}` →
  响应 `{"answer", "thread_id"}`；`app/main.py` 的 `create_app()` 是唯一组装点，一个进程建一次 Agent 与 checkpointer，
  请求内不重建。API 层复用同一个 Agent，不写业务规则、不自建模型。本版本不含人工确认与认证，见[同步 Chat API（HTTP）](#同步-chat-apihttp)。
- **HTTP 流式（SSE）**：同一条 `POST /api/chat`，客户端带 `Accept: text/event-stream` 就改成 SSE 逐段推送，
  只有 `token`／`done`／`error` 三类事件；请求体与不带该头时的 JSON 响应一字未改，CLI 流式也不受影响。
  见[HTTP 流式（SSE）](#http-流式sse)。
- **测试**：309 个用例全部通过（`uv run pytest`）。
- **Agent 评测（最小集合）**：`eval/runner.py` + 12 条 case / 86 个确定性断言，覆盖路由、双意图、政策检索与未覆盖拒答、
  取消（批准 / 拒绝 / 不允许）、退款（受理 / 超期 / 在途 / 未付款）；真实模型全绿并已建立基线。
  覆盖面与判法见[Agent 评测](#agent-评测eval)与[已知缺口](#已知缺口)第 7 条。

尚未开始 / 已知缺口见文末[已知缺口](#已知缺口)。

## 技术栈

- Python 3.12 + [uv](https://docs.astral.sh/uv/)（受 macOS Intel x86_64 限制：sentence-transformers 依赖的 torch 最后有 x86_64 wheel 的版本是 2.2.2，仅支持到 cp312，请勿升级到 3.13；`pyproject.toml` 的 `[tool.uv].required-environments` 已锁定该平台）
- LangChain（`@tool`）+ LangGraph（`create_agent`、`interrupt`、checkpointer）
- Web：FastAPI + uvicorn（同步 Chat API），Pydantic v2 同时承担数据模型与 API 的请求／响应契约
- RAG：`chromadb`（持久化向量库）+ `sentence-transformers`（BAAI/bge 系列 Embedding 与 Reranker）
- python-dotenv、pytest、LangSmith（追踪）

## 目录结构

```
customer-service-agent/
├── app/
│   ├── agent/graph.py          # Agent 图、系统提示、TOOLS 清单、ChatOpenAI 构建
│   ├── agent/streaming.py      # 流式事件展示层：文本取值、工具名→状态文案、读待确认 payload
│   ├── api/                    # HTTP 层：请求/响应契约与路由，不写业务规则、不自建模型
│   │   └── chat.py             #   ChatRequest / ChatResponse、create_chat_app(agent)、POST /api/chat
│   ├── tools/                  # 工具层：LangChain @tool，面向 LLM 的调用时机说明 + 中文措辞
│   │   ├── order_tools.py      #   query_order / cancel_order（并被退款复用状态文案与裁决解析）
│   │   ├── refund_tools.py     #   refund_order：退款确认 payload 与措辞
│   │   ├── logistics_tools.py  #   query_logistics
│   │   └── rag_tools.py        #   search_knowledge
│   ├── services/               # 业务层：规则判定与结论枚举，不碰文件、不管措辞
│   │   ├── order_service.py    #   OrderService：查询 / 取消
│   │   ├── refund_service.py   #   RefundService：退款判定与受理（含签收时效与时钟注入）
│   │   └── logistics_service.py
│   ├── rag/                    # 检索层：knowledge_base（入库）+ rag_service（召回与重排）
│   ├── data_loader.py          # 数据层：Mock JSON 读写，状态写回的唯一入口
│   ├── models.py               # Order / LogisticsInfo 模型与状态枚举
│   ├── config.py               # 路径、RAG 参数、退款时效、环境变量读取
│   └── main.py                 # 应用组装：create_app() 启动期建一次 Agent + Checkpointer 再装路由
├── scripts/
│   ├── build_knowledge_base.py # 知识库构建（可重复执行）
│   ├── hitl_cli.py             # 取消 / 退款的人工确认命令行入口（非流式）
│   ├── smoke_data.py           # 数据层冒烟：读配置与 Mock 订单，不需要大模型（原 app/main.py）
│   ├── stream_cli.py           # 流式命令行入口：回复逐字上屏 + 工具状态行
│   ├── chat_api.py             # Chat API 启动入口（JSON + SSE）：只解析 --host/--port 并交给 uvicorn
│   └── streaming_demo.py       # 探索用的最小样例，不是入口（未处理 ToolMessage）
├── tests/                      # 12 个测试文件，按垂直切片组织
├── eval/                       # Agent 评测：真实模型跑 dataset，确定性断言 + 基线回退判定
│   ├── dataset/*.yaml          #   case 定义（用户输入 + 期望的工具/参数/中断/判定/数据/措辞）
│   ├── runner.py               #   唯一评测入口：装载校验 → 隔离执行 → 采集证据 → 断言 → 报表
│   ├── baseline.json           #   基线（--update-baseline 写入；回退时退出码 1）
│   ├── results/                #   每次运行的完整证据（调用链、工具返回、回答、数据快照）
│   ├── min_runner.py           #   演示代码：手写的单条链路（无 HITL），不是评测入口
│   └── min_hitl_runner.py      #   演示代码：手写的暂停 → 裁决 → 恢复链路，不是评测入口
├── data/
│   ├── mock/                   # orders.json / logistics.json
│   ├── knowledge/              # 退款退货、配送、售后保修、会员权益 4 份政策文档
│   └── chroma/                 # 持久化向量库（构建产物，已在 .gitignore 中）
├── .env.example
└── pyproject.toml
```

## 分层与职责

```
用户问题   # 入口：scripts/hitl_cli.py | stream_cli.py（CLI）、POST /api/chat（HTTP）——都汇到同一个 Agent
  └─ Agent（app/agent/graph.py：按工具描述选择工具，依据返回值作答）
       └─ Tool 层（app/tools/*）        # 何时该调用 / 何时不该调用 / 未找到时不要编造；把结论渲染成中文
            └─ Service 层               # 业务规则判定，返回结论枚举；不读写文件、不写话术
                 └─ data_loader.py      # 唯一的数据读写入口
                      └─ data/mock/*.json
```

| 层 | 约定 |
| --- | --- |
| Tool | 只做「调用检查、是否中断、措辞」。不出现业务规则常量（如时效天数），不直接 import `data_loader` |
| Service | 规则集中处：判定（`check_*`，只读）与执行（`OrderService.cancel_order` / `RefundService.apply_refund`，写库）分离，执行前重新判定，不信任上层早先的结论 |
| data_loader | 状态字段的唯一写入口；Service 与 Tool 都不直接改文件 |
| 展示（`agent/streaming.py`） | 流式事件的取数与查表：哪类消息有正文、哪个工具名对应哪句状态、一条 Agent 流怎么跑成文本增量（`stream_answer`）。不做 IO、不产生业务结论，展示文案也不由模型输出决定，更不认识 HTTP／SSE |
| API（`api/chat.py`） | HTTP 契约 ↔ Agent 契约的互转：字段校验、组装 `thread_id`、取最终答复正文、`Accept` 协商与 SSE 事件分帧／编码。不 import Service 与数据层、不自建模型、不留全局 app / 全局 agent |
| 组装（`main.py`） | 应用工厂 `create_app()`：一个进程建一次 Agent 与 Checkpointer，再交给路由；请求内不再创建。不写 HTTP 契约、不含业务规则、模块级不留全局 app |

一个能力一套文件：`tools/*_tools.py` 与 `services/*_service.py` 按能力（订单 / 退款 / 物流 / RAG）一一对应。
退款虽属订单域，但它是唯一带时效判定、只推进付款状态的动作，故独立成 `refund_tools.py` + `refund_service.py`；
跨切片复用的只有措辞与契约本身（`PAYMENT_STATUS_LABELS`、`is_approved`），不复制第二份规则实现。

这样切的好处：数据来源从 Mock JSON 换成真实订单中台时只改 Service 以下；措辞调整不会碰到规则。

## Agent 与工具

`app/agent/graph.py` 导出 `create_customer_service_agent(model=None, *, checkpointer=None)`，绑定 5 个工具：

| 工具 | 用途 | 是否写数据 |
| --- | --- | --- |
| `query_order(order_id)` | 订单状态、金额、商品、下单时间 | 否 |
| `query_logistics(order_id)` | 发货与否、运单号、快递公司、最新位置、是否签收 | 否 |
| `search_knowledge(query)` | 退款退货 / 配送 / 售后保修 / 会员权益等政策原文片段 | 否 |
| `cancel_order(order_id)` | 取消订单：检查 → 人工确认 → 写 `status=cancelled` | 是（需人工批准） |
| `refund_order(order_id)` | 受理退款：检查 → 人工确认 → 写 `payment_status=refunding` | 是（需人工批准） |

系统提示里写死的几条硬约束（有测试守卫）：规则问题走 `search_knowledge`、实际问题走数据工具、动手处理走
`cancel_order` / `refund_order`；工具返回「未找到」「未检索到」时如实转述；**取消成功不等于退款、退款已受理不等于款项到账**。

## 退款垂直切片

付款状态与订单状态**分开建模**：取消只改 `status`，款项是否退回由 `payment_status` 表达
（`unpaid` / `paid` / `refunding` / `refunded`）。因此「已取消但付过款」的订单仍然可以、也必须能发起退款。

`RefundService.check_refund`（`app/services/refund_service.py`）是唯一判定来源，`apply_refund` 复用它，结论枚举如下：

| 结论 | 触发条件 | 是否进入人工确认 |
| --- | --- | --- |
| `CAN_REFUND` | 已付款且在时效内；或未发货 / 已取消但已付款（不受签收时效约束） | 是 |
| `REFUNDING_SUBMITTED` | 人工批准后本次受理，付款状态写为 `refunding` | — |
| `REFUND_EXPIRED` | 超过自签收次日起 `REFUND_WINDOW_DAYS` 个自然日 | 否 |
| `NEED_REJECT_DELIVERY` | 已发货未签收（在途），按政策先拒收 | 否 |
| `NO_DELIVERY_TIME` | 状态为已签收但缺 `delivered_at`，数据异常 | 否 |
| `NOT_PAID` | 从未支付，无款可退 | 否 |
| `ALREADY_REFUNDING` / `ALREADY_REFUNDED` | 不重复受理 | 否 |
| `NOT_FOUND` | 订单不存在或订单号为空 | 否 |

时效口径：**自物流签收次日起 7 个自然日**（含周末与节假日），签收当日不计入，截止日当天仍可受理，
边界为 `now.date() <= delivered_at.date() + REFUND_WINDOW_DAYS`。数字定义在 `app/config.py`，
`tests/test_refund_slice.py` 有一条防漂移守卫，断言它与 `data/knowledge/refund_policy.md` 第二章口径一致。

时效判定依赖系统时间，所以 `RefundService` 接受 `now` 注入（默认 `datetime.now`），测试用固定时钟覆盖期内/超期边界；
`OrderService` 的查询与取消不依赖时间，已不再需要这个注入点。

**不满足退款条件的订单绝不会打扰人工，也不会写任何数据**；`interrupt` 之前保证只有读操作
（恢复时 Tool 节点从中断点前重放，写库必须留在 `interrupt` 之后，同样有测试守卫）。

工具输出示例（当前 Mock 数据实跑）：

```python
from app.tools.refund_tools import refund_order

refund_order.invoke({"order_id": "ORD-2026-0006"})   # 2026-09-09 签收
# 订单 ORD-2026-0006（便携咖啡机，899.00 元，付款状态：已付款） 已超过该单的退款时效
# （自签收次日起算，至今已超期 7 个自然日），本入口不受理退款申请。本次没有改动任何订单，
# 请如实告知用户已超期，并建议按售后政策走退货／换货／维修或转人工客服，不要说退款已受理。
# （超期天数按「今天」实时计算，此示例随系统日期变化）

refund_order.invoke({"order_id": "ORD-2026-0003"})   # 已发货未签收
# 订单 ORD-2026-0003（27英寸4K显示器，2399.00 元，付款状态：已付款） 已发货但尚未签收，
# 按政策应先拒收包裹，商品退回仓库后再进入退款流程。本次没有改动任何订单，不要受理在途退款。
```

## 多轮对话（短期记忆）

同一个 `thread_id` 的多次 `invoke` 共享上下文：`messages` 通道是追加式的，检查点里存着该线程之前的全部消息
（含工具返回值），因此第二轮的"那它现在到哪了""帮我取消它"能接上第一轮那笔订单。

```python
agent = create_customer_service_agent(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "session-1"}}   # thread_id 就是会话身份

agent.invoke({"messages": [HumanMessage("帮我查一下订单 ORD-2026-0004")]}, config)
agent.invoke({"messages": [HumanMessage("那它现在到哪了？")]}, config)   # 第二轮据此调用 query_logistics
```

- **载体只有一份**：注入的 checkpointer + `thread_id` + `State.messages`。`app/agent/graph.py` 内不保存全局历史，
  也不自建 checkpointer（有源码守卫测试）。
- **换 `thread_id` 就是换会话**：不同线程之间上下文互不可见。
- **不注入 checkpointer 不报错，但每一轮都是全新上下文**：多轮指代退化成"请提供订单号"，
  HITL 的暂停与恢复也一并失效（见[人工确认（HITL）](#人工确认hitl)）。
- 系统提示里的指代口径（有测试守卫）：订单号只能取自**对话历史中真实出现过的值**，不得拿提示中的示例订单号凑数；
  上文只指向唯一订单时才沿用，历史里出现多笔订单时列候选请用户说明、不猜；
  **沿用的只允许是订单号**，订单与物流状态本轮必须重新调用工具获取，上一轮人工批准的结论不等于本轮已获批准。

命令行入口一个进程一个会话，会话 id（即 `thread_id`）会打印在开头：

```bash
uv run python scripts/hitl_cli.py                      # 新会话
uv run python scripts/hitl_cli.py --session-id vip-1   # 固定会话 id，便于复现同一段上下文
```

## 人工确认（HITL）

`cancel_order` 与 `refund_order` 共用一套裁决契约：中断 payload 是带 `type` 的字典
（`order_cancellation_confirmation` / `order_refund_confirmation`），恢复时传 `Command(resume={"approved": ...})`。
在能正常送达裁决的取值里，只有 `{"approved": True}` 算批准，其余（`False`、`"yes"`……）一律按拒绝处理（fail-closed）。
解析实现只有一份：`app/tools/order_tools.py` 的 `is_approved`，`refund_tools.py` 直接复用，两条确认链路不会各判一次。

`{}` 和 `None` 是例外：它们**不是「按拒绝处理」，而是根本不会恢复执行**（langgraph 1.2.11 实测）。
空字典会被当成「零条记录的 resume-map」直接吞掉，图原地重复同一个中断；`None` 则在库内部抛 `UnboundLocalError`。
所以调用方必须自己把人工回应规整成 `{"approved": bool}` 再 `resume`，不得把上游 JSON 原样透传
（`scripts/hitl_cli.py` 只发 `{"approved": True/False}`，正是这个原因）。

```python
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from app.agent import create_customer_service_agent

agent = create_customer_service_agent(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "demo"}}          # 暂停与恢复、多轮上下文都用同一个 thread_id

result = agent.invoke({"messages": [HumanMessage("ORD-2026-0004 我要退款")]}, config)
while interrupts := result.get("__interrupt__"):
    print(interrupts[0].value["question"])                  # 展示待确认信息
    result = agent.invoke(Command(resume={"approved": True}), config)
print(result["messages"][-1].content)
```

交互式体验直接用命令行（会调用真实大模型）：

```bash
uv run python scripts/hitl_cli.py
```

注意：`checkpointer` 必须注入，否则图仍会抛出中断但 `Command(resume=...)` 会报错，本轮停在待确认状态
（订单数据不会被写坏）。当前入口用的是 `InMemorySaver`，只在进程内有效。

## 流式输出（CLI Streaming）

`agent.stream(..., stream_mode="messages")` 会把模型的分片（token）与工具结果一起当作事件吐出来，
CLI 消费这些事件，做到「回复边生成边上屏」，工具执行期间给一句进度提示。

```bash
uv run python scripts/stream_cli.py                      # 新会话
uv run python scripts/stream_cli.py --session-id vip-1   # 固定会话 id，便于复现同一段上下文
```

屏上只允许出现两类内容，其余一律不打：

| messages 流里的事件 | 上屏内容 |
| --- | --- |
| `AIMessageChunk` 的文本增量 | 原样逐字打屏（`print(text, end="", flush=True)`） |
| 工具调用（`tool_call_chunks` / `tool_calls`） | 按**工具名**查 `TOOL_STATUS_LABELS` 得到的一行状态，如 `[正在查询订单...]` |
| `ToolMessage`、事件 `metadata` | 什么都不打 |

- **状态文案由程序决定**：文案只在 `app/agent/streaming.py` 的 `TOOL_STATUS_LABELS` 里存一份，
  模型只负责决定「调哪个工具」，不参与措辞；未登记的工具名一律降级成 `[正在处理...]`，不把内部名字抛给用户。
- **一次调用只提示一次**：一次工具调用会被拆成多条分片下发（首片带工具名，后续片只有参数碎片），
  按 call id 去重，参数碎片与收尾空片都不再产生提示行。
- **工具结果原文不上屏**：`ToolMessage` 的 content 是喂给模型的中间数据（订单详情、政策片段、内部指令），
  打给用户既难看又是内部结构泄漏；messages 流里不只有 `AIMessageChunk`，
  展示层按消息类型分流，非 AI 消息一律取到空文本而不是抛 `AttributeError`。
- **分片类的 `.type` 不是 `"ai"`**：`AIMessage.type == "ai"`，而 `AIMessageChunk.type == "AIMessageChunk"`，
  按字符串比较判「是不是 AI 消息」会把整条流的正文静默丢掉（实测踩过，`stream_text` 按类型判定并有测试兜底）。
- **HITL 在流式下换一种读法**：图暂停时这条流只是结束，中断不会变成事件，`stream()` 的返回里也没有
  `__interrupt__`，待确认 payload 只能从 `agent.get_state(config).tasks[*].interrupts[*].value` 读；
  裁决仍旧发 `Command(resume={"approved": bool})`（形状约束见上一节），恢复后重开一条流，
  第一条事件就是工具结果，随后的文本增量就是最终答复。
- **确认话术不复制**：中断 payload 自带 `question` 字段（Tool 层写好），流式 CLI 直接拿它问人，
  不再维护第二张确认文案表。
- **既有链路没动**：`invoke()` 入口（`scripts/hitl_cli.py`、`eval/runner.py`）、Tool / Service / data_loader
  的业务逻辑与措辞都保持原样，流式只是新增一个消费端；Memory、`thread_id`、HITL 共用同一份 checkpointer。

## 同步 Chat API（HTTP）

`app/api/chat.py` 用 FastAPI 把同一个 Agent 暴露成 `POST /api/chat`：
请求 `{"message", "thread_id"}` → 响应 `{"answer", "thread_id"}`。它是 Agent 的又一个消费端，不是新的一层业务。
应用怎么装起来见 `app/main.py` 的 `create_app()`：Agent、Checkpointer、路由的唯一组装点。

```bash
uv run python scripts/chat_api.py                       # 默认 127.0.0.1:8000，文档在 /docs

# 第一轮不带 thread_id：服务端新建一个会话并在响应里回显
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "订单 ORD-2026-0001 现在什么状态"}'
# → {"answer":"...","thread_id":"chat-16d50d0f8bed"}

# 第二轮带上上一轮返回的 thread_id，就接得上同一会话
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "那它到哪了", "thread_id": "chat-16d50d0f8bed"}'
```

- **一个进程只组装一次**：`app/main.py` 的 `create_app(model=None, *, checkpointer=None)` 在启动阶段建好 Agent
  与 `InMemorySaver`，再交给路由；请求处理函数里不再出现任何构建调用（有工厂计数与源码守卫）。
  换 checkpointer、换模型只需要改 `create_app()` 一处，`scripts/chat_api.py` 只解析 `--host`／`--port` 并起服务。
  代价：传给 `uvicorn.run()` 的是应用实例而不是导入串，因此不支持 `--reload`；改代码要重启进程。
- **图靠注入，没有全局**：`create_chat_app(agent)` 只接一个可 `invoke` 的 Agent，模块里不留全局 app 与全局 agent，
  也不 import 模型构建（有源码守卫）。所以 API 测试塞一个假 Agent 就能跑完，一次大模型调用都不发。
- **`thread_id` 原样透传给 checkpointer**：同一个 `thread_id` 的多次请求共享上下文（实测第二次模型能数出两轮提问），
  换个值就是另一段对话；没带时服务端生成 `chat-<12位hex>` 并回显，客户端存它即可续聊。
  服务端不维护会话列表，`openapi.json` 里只有 `/api/chat` 这一个业务端点。
- **响应只有两个字段**：`answer` 取本轮最后一条 AI 消息的正文，复用展示层唯一的 `stream_text`
  （含内容块形态），因此 `ToolMessage` 原文、`tool_calls`、`metadata` 都不会出现在响应里 —— 有断言守卫。
- **校验先挡在 Agent 之前**：`message` 缺失／空／只有空白／类型错，以及任何未知字段，一律 422 且不消耗一次模型调用。
  未知字段选 `extra="forbid"` 而不是静默忽略：否则客户端会以为自己的 `approved: true` 生效了。
- **需要人工确认的请求明确拒绝（501）**：用户在 `message` 里要求取消／退款时图会停在待确认点，
  本端点没有回裁决的能力，宁可返回 501 也不回一条空 `answer` 假装处理过。
  实测：这一笔不写任何数据，同一会话之后的普通请求照常回答；代价是该线程留下一个待确认中断
  和一条没有工具结果的 AI 消息 —— 收到 501 应改用 [`scripts/hitl_cli.py`](#人工确认hitl) 或换一个全新会话，
  不要在同一段对话里反复重试同一笔办理。
- **上游失败不会被吞成 200**：模型调用报错时响应是 500，不是一条空 `answer`。
- **处理函数用普通 `def`**：`agent.invoke` 是阻塞调用，交给 FastAPI 线程池执行；图本身没有 await 能力，不伪装 async。

### HTTP 流式（SSE）

同一条 `POST /api/chat` 增加流式，**不新增端点、不给 `ChatRequest` 加字段**，靠 `Accept` 头协商：

```bash
uv run python scripts/chat_api.py

# 带 Accept: text/event-stream → SSE 逐段推送（curl -N 关掉本地缓冲才能看到分段到达）
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"message": "订单 ORD-2026-0002 到哪了"}'
# event: token
# data: {"content": "您好，"}
# event: token
# data: {"content": "您的订单 ORD-2026-0002"}
# ...
# event: done
# data: {}

# 不带这个头（或只带默认的 Accept: */*）→ 仍然是原来的 JSON 一次性响应
```

事件只有三类，客户端只需按顺序把 `token.content` 拼起来：

| 事件 | data | 含义 |
| --- | --- | --- |
| `token` | `{"content": "增量文本"}` | 模型正文的一个增量，可能出现多次 |
| `done` | `{}` | 本轮正常结束，只出现一次且必定是最后一个事件 |
| `error` | `{"message": "一句话"}` | 本轮失败或被拒绝，**终止事件**，后面不会再有 `done` |

- **只有显式 `Accept: text/event-stream` 才走流式**：`*/*` 是 curl 与多数 HTTP 客户端的默认值，
  把它算成流式请求会悄悄改变所有既有客户端拿到的响应形态。
- **分层仍旧干净**：Agent 流的读法在 `app/agent/streaming.py::stream_answer`（只产出纯文本增量，不认识 HTTP/SSE），
  SSE 的分帧与编码只在 API 层。有源码守卫禁止 `fastapi` / `StreamingResponse` / `event:` / `json.dumps` 出现在展示层。
- **`thread_id` 走响应头 `X-Thread-Id`**：生成规则与同步路径完全一致（没带就新建 `chat-<12位hex>`），
  因为 SSE 正文里没有承载它的字段；多轮记忆在流式下照常生效（同一 `thread_id` 的第二轮看得到第一轮，有真图测试兜底）。
- **出错不甩 traceback**：Agent 侧异常只回一句固定话术，异常原文可能含模型名与端点地址，只进服务端日志；
  流一旦开始响应头就发出去了，状态码只能是 200，失败信息一律靠 `error` 事件表达。
- **停在人工确认时不假装答完**：`stream_mode="messages"` 不会把中断当事件吐出来，流只是安静结束，
  因此每条流收尾时都读一次线程状态，只要它停在待确认点就改发 `error`（沿用同步路径那句「该操作没有执行」），
  而不是回 `done` 让客户端把「没处理」读成「答完了」。这条判定不能用「本轮有没有文本增量」代替：
  真实模型常态是先回一句「我来帮您取消」再调工具，那句客套话已经到了客户端，图却才刚停在确认位上
  （真实模型 smoke 实测撞出来的，`test_preamble_tokens_do_not_hide_a_parked_confirmation` 钉住它）。
  已经下发的增量不回滚，客户端拿到「半句 + `error`」；代价是每条流多一次 checkpoint 读。
- **工具原文与内部结构不进流**：只吐 `AIMessage` 系列的正文，`ToolMessage`、`tool_call_chunks`、事件 `metadata`
  都被展示层挡掉（与 CLI 同一份 `stream_text` 判定）。
- **本版做不到的**：没有心跳与断线续传（连接断了只能整轮重来）、客户端与服务器都没有「中途停止生成」、
  工具执行期间没有进度事件（要状态行请用 `scripts/stream_cli.py`）、错误也只有文本一种载体，
  没带状态码语义。见[已知缺口](#已知缺口)第 9 条。

## RAG 知识库

```
data/knowledge/*.md
  → 加载（记录 document_id / source 元数据）
  → RecursiveCharacterTextSplitter 切分（chunk 500 / overlap 60，分隔符含中英文句号逗号）
  → Embedding：BAAI/bge-small-zh-v1.5
  → Chroma 持久化：data/chroma/，collection：customer_service_knowledge

检索：向量粗排 RETRIEVAL_K=10 → CrossEncoder(bge-reranker-base) 重排
     → 相关性底线 MIN_RERANK_SCORE=0.1 过滤 → 取前 CONTEXT_TOP_K=3 拼装上下文
```

构建脚本可重复执行（每个源文档先删自己的旧 chunk 再写入）：

```bash
uv run python scripts/build_knowledge_base.py
```

**换 Embedding 模型必须重建整个知识库**（新旧向量不在同一空间），改 `EMBEDDING_MODEL` 后务必重跑上面的脚本。
`search_knowledge` 与检索链路都不依赖 `LLM_API_KEY`，向量库和 Reranker 都是惰性加载，不会拖慢 Agent 启动。

## 快速开始

```bash
# 1. 安装依赖（自动创建 .venv）
uv sync

# 2. 配置环境变量
cp .env.example .env      # 填入真实 API Key

# 3. 构建知识库（首次运行或政策文档变更后）
uv run python scripts/build_knowledge_base.py

# 4. 冒烟 / 测试
uv run python scripts/smoke_data.py
uv run pytest

# 5. 手动调用工具（模拟 Agent 调用，无需 LLM）
uv run python -c "from app.tools.order_tools import query_order; print(query_order.invoke({'order_id': 'ORD-2026-0001'}))"
```

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` | 大模型服务 API Key（OpenAI 兼容端点） |
| `LLM_MODEL` | 模型名 |
| `LLM_BASE_URL` | OpenAI 兼容端点地址 |
| `LANGSMITH_API_KEY` | LangSmith API Key |
| `LANGSMITH_TRACING` | 是否开启追踪（true/false） |
| `LANGSMITH_PROJECT` | LangSmith 项目名 |

真实密钥只保存在 `.env`（已加入 `.gitignore`），不要写入代码或提交到仓库。
若未配置有效的 `LANGSMITH_API_KEY`，请把 `LANGSMITH_TRACING` 设为 `false`；否则离线环境会在 pytest 输出里
刷屏 LangSmith 上报失败的报错（不影响功能，但会盖掉测试汇总行）。

`app/config.py` 里还固定了 `LLM_EXTRA_BODY = {"thinking": {"type": "disabled"}}`：GLM-4.5 系列默认开启动态思考，
且该端点会把思考正文混进响应的 `content` 字段，不显式关闭会在用户侧看到重复的回答。

## 测试

```bash
uv run pytest                        # 309 passed
uv run pytest tests/test_refund_slice.py
```

按垂直切片组织，除数据模型测试外每个切片一个文件（`test_query_order_slice`、`test_cancel_order_slice`、
`test_query_logistics_slice`、`test_refund_slice`、`test_agent_slice`、`test_conversation_memory_slice`、
`test_streaming_slice`、`test_chat_api_slice`、`test_http_streaming_slice`、`test_knowledge_base`、`test_rag_service`）。
几条贯穿性的约定，都有测试兜底：

- **写库测试只落在 `tmp_path` 副本**，`test_refund_slice.py` 末尾有哨兵断言真实 `data/mock/orders.json` 逐字节未被改动。
- **时效测试注入固定时钟**，不随日历漂移。
- **多轮测试用「回显式假模型」**（`test_conversation_memory_slice.py`）：它只根据自己实际收到的消息决定调哪个工具，
  解析上文订单号时把 SystemMessage 排除在外 —— 系统提示里的示例订单号恰好是真实存在的 `ORD-2026-0001`，
  不排除就无法证明这个号是从对话历史里拿到的。因此这些断言证明的是「上下文确实跨轮送达、且按线程隔离」这条链路，
  不是模型的理解能力；真实模型指代得对不对属于 eval 的职责。
- **架构守卫**用 `inspect` / `ast` 检查源码：写入必须在 `interrupt` 之后、规则不下沉到 Tool 层、
  Tool 不引用数据层、时效数字与政策文档一致、图模块不自建对话历史存储、状态文案只在 `TOOL_STATUS_LABELS` 有一份。
- **流式测试用「脚本化假流式模型」**（`test_streaming_slice.py`）：它把一次模型输出按真实协议形状拆成
  `tool_call_chunks`（首片带工具名、后续片只有参数碎片）与文本分片逐条下发，
  因此测到的是「分片 → 去重 → 上屏」这条链路，不是模型的选择能力。
  同一个模型同时实现 `_generate` 与 `_stream`，两条入口给出同一个决定，
  所以「加流式没有改坏 `invoke()`」与 `scripts/hitl_cli.py` 回归也是在断言里，不是靠假设。
- **API 测试分三层**（`test_chat_api_slice.py`）：假 Agent 只记录 `invoke` 收到的入参，用来证 HTTP 契约本身
  —— 校验挡在模型之前（非法请求断言 `calls == []`）、`thread_id` 透传、响应不外泄工具原文、上游报错转 500 而不是空答复；
  真图 + 假模型走 `TestClient`，用来证 `thread_id` 确实接到了 checkpointer（第二次请求模型数得出两轮提问）
  与被拒的取消一笔未改；应用工厂层把 `create_customer_service_agent` 换成记录调用的假工厂，
  证 Agent 与 Checkpointer 在一个进程里只建一次、之后每个请求复用同一个实例（第二轮「第 2 轮提问」同时也否证了每请求重建）。
  假答复文本刻意不含任何「泄漏标记」子串（有自检断言兜住），
  否则「响应不含内部结构」这条断言会失效或被自己的数据误伤。
- **HTTP 流式测试用「脚本化假流式 Agent」**（`test_http_streaming_slice.py`）：它按真实协议形状直接吐
  `(消息, metadata)` 事件，测试再把 SSE 正文按空行解析回事件来断言分帧、顺序与 JSON 转义（含正文里的换行）。
  两条关键守卫是「没带 `Accept: text/event-stream` 的请求必须走 `invoke`」（假 Agent 的 `stream()` 一被调用就报错）
  与「`error` 事件只有一句固定话术」（假异常里塞了内部字样，出现在正文即判失败）；
  另有真图 + 假流式模型跑 `create_app()`，证明流式没有绕开 checkpointer——同一 `thread_id` 第二轮仍数得出两轮提问。

## Agent 评测（eval）

pytest 回答「业务规则对不对」，eval 回答「模型会不会选错工具、会不会编造、会不会把受理说成到账」——
也就是目前只有系统提示在约束的那部分行为。

```bash
uv run python -m eval.runner                     # 跑全部 case
uv run python -m eval.runner --case E01,E05      # 只跑指定 case
uv run python -m eval.runner --update-baseline   # 把本次结果接受为基线
EVAL_TRACING=true uv run python -m eval.runner   # 需要在 LangSmith 看链路时再开 tracing（默认关掉以保证终端输出可读）
```

`eval/min_runner.py` 与 `eval/min_hitl_runner.py` 是**演示代码**：手写的单条 case 最小链路，用来说明
「invoke → interrupt → resume」本身是怎么回事，不参与评测。评测入口只有 `eval/runner.py` ——
只有它做 dataset 装载校验、全量断言、报表与基线比对，以及真实 `data/mock/orders.json` 的字节级哨兵。

一条 case 的链路：

```
dataset/*.yaml
  → 把 data/mock/orders.json 复制到临时目录并改道 data_loader.ORDERS_FILE
  → 真实模型 invoke；命中 interrupt 时按 case 的 hitl.decision 恢复一次
  → 采集 tool_calls / 工具原始返回 / 中断 payload / 最终回答 / 跑前跑后数据快照
  → 确定性断言 → 终端报表 + eval/results/run-<时间>.json + 与 baseline.json 比对
```

断言只有结构化那一类，没有 LLM-as-judge：`tools`（必调）、`not_tools`（禁调，例如「到哪了」不许调 `query_order`）、
`order_id`（不串单）、`interrupt`（该不该暂停 + payload 字段是否符合确认界面契约）、
`outcome`（按 Service / Tool 结论的特征短语判定，映射表 `OUTCOME_PHRASES` 缺项时直接报错，避免改了文案断言却静默失效）、
`data`（字段级 `[跑前, 跑后]`，同时验证「该写的写了」与「不该动的没动」）、
以及 `answer_include` / `answer_include_any` / `answer_exclude`（禁词包含工具文案里的原话，模型复读出来就是内部指令泄漏）。

当前 12 条 case（`eval/dataset/` 按切片分文件）：

- **routing**：E01 订单查询按实作答；E02「到哪了」必须走 `query_logistics` 而不是 `query_order`。
- **multi_intent**：E04 一句话同时问订单状态和物流位置，两个工具都要调（唯一一条 `warn`：漏查是回答质量问题，不碰数据边界）。
- **knowledge**：E03 到账时效按政策原文回答；E12 知识库没有收录支付方式条款，唯一正确的回答是如实说没找到。
- **cancel**：E05 人工批准后将状态写回；E06 已发货不接受人工确认（零中断）；E07 人工拒绝后订单数据一个字都不能改。
- **refund**：E08 期内受理（只推进付款状态，订单状态停在已签收）；E09 超期；E10 已发货未签收应先拒收；E11 未付款无款可退。

E10 是**已知的偶发红**，红在判法粒度而不是行为：禁词 `退款成功` 是裸子串匹配，模型若在最后补一句
「退款成功后，款项会按照原支付渠道退回」（政策条件句，业务结论仍是不受理）就会被判红。2026-09-26 实测 9 次红 2 次（约 22%），
这 9 次的工具链路、结论与数据快照全部正确。当前口径是**保留禁词不收紧**（收紧会漏判「退款成功」单独成句的真违规），
报红时先 `uv run python -m eval.runner --case E10` 单跑 3 次取多数再定性 —— n=1 的评测里，「回退」必须先排除抖动。

三条与安全边界同源、也与测试相同的约束：

- **数据隔离**：人工批准产生的写入只落在临时副本，runner 末尾有哨兵断言真实 `data/mock/orders.json` 逐字节未变。
- **时钟冻结**：带 `now` 的 case 注入 `RefundService(now=...)`，时效判定不随日历漂移。E08/E09 用它把「期内」与「超期」
  钉成常量；E08 断言中断 payload 里的 `days_remaining`，也顺带证明了冻结真的生效（漏注入时这个数会跟着系统日期变）。
- **裁决形状**：`hitl.decision` 必须是非空 dict，否则 runner 直接判该 case 失败 —— LangGraph 只认非空 resume 值，
  详见[人工确认（HITL）](#人工确认hitl)。

## 已知缺口

按当前阶段口径刻意保留、尚未实现的部分：

1. **退款时效只按「7 天无理由」判定**。政策第三章第 1 条承诺签收次日起 15 日内因质量问题可全额退款，
   该通道未建模 —— 签收第 8–15 天以质量问题申请会被判 `REFUND_EXPIRED` 并引导转人工/售后。
   **当前口径：本阶段只按 7 天计算，其余政策内容暂不纳入判定。**
2. **`refunded` 只读不写**。受理只写 `refunding`，款项到账由支付渠道异步决定，系统没有结算回调或对账入口，
   因此 `ALREADY_REFUNDED` 在真实数据中暂时不可达。
3. **退款金额恒为全额实付**。拒收扣运费、优惠券与积分抵扣拆分、部分退款（政策第四、五章）均未实现。
4. **政策第一章「下单后 2 小时内自助仅退款」未建模**：已付款未发货的订单一律判为 `CAN_REFUND`，
   由人工确认环节兜底。
5. **确认过程无持久化与审计**：`InMemorySaver` 进程退出即丢失待确认请求，谁批准、何时批准没有记录。
6. **数据层无并发保护**：`data_loader` 整读整写 JSON，取消与退款并发写回会互相覆盖。
   切到真实数据库时随 Service 层一并替换。
7. **评测覆盖面仍有洞**：`eval/` 的 12 条 case 已覆盖五个切片的主干分支，但 `ALREADY_REFUNDING`／`ALREADY_REFUNDED`／
   `NO_DELIVERY_TIME` 三条退款结论与「一单多包裹」还没有 case —— 它们都需要 Mock 数据里不存在的前置状态
   （已受理过退款的单、状态与签收时间互相矛盾的单），要补得先加数据快照。
   判法上仍然每条只跑一次（n=1，没有一致率）、只做确定性结构断言（没有 LLM-as-judge，回答是否得体靠必含/禁词加
   人工翻 `eval/results/`），对抗类输入（提示注入、跨用户订单）刻意未纳入。
   n=1 的代价已经实测到：真实模型的措辞抖动足以让 blocking case 偶发红（见上文的 E10），只能靠单跑复现来定性，
   要根治得给每条 case 跑 n 次统计一致率。
   另外 `knowledge` 两条依赖 HuggingFace 可达性：Embedding 与 Reranker 冷启动都会向 hub 发 HEAD 请求，
   网络抖动时 `search_knowledge` 直接抛 `SSLError`、整条 case 判失败（这个 transformers 版本下 `HF_HUB_OFFLINE=1`
   会走另一条必经的 hub API，反而更早就失败）。
8. **短期记忆只做「同线程累积」，四条边界都在**：历史只活在 `InMemorySaver` 里，进程退出即失忆、跨进程不能续聊；
   没有窗口裁剪与摘要，消息随轮次线性增长，长会话最终会撞模型上下文上限；`thread_id` 就是访问凭证，
   没有会话归属与权限校验（拿到别人的 thread_id 就能接着那段对话说）；
   `eval/dataset/` 目前 12 条 case 全是单轮，**多轮指代是否真的解对没有机器判定**，
   pytest 只证明上下文送达与线程隔离，实际效果靠人工翻 `scripts/hitl_cli.py`。
9. **流式已上到网络，但边界还在**：`stream_mode="messages"` 现在有两个消费端 —— CLI 的逐字上屏与
   HTTP 的 SSE（`Accept: text/event-stream`，只有 `token`／`done`／`error` 三类事件）。
   没做的是心跳与断线续传（连接断了只能整轮重来，半条回答无法续上）、客户端与服务器的「中途停止生成」，
   以及 WebSocket 与前端页面；HTTP 通道刻意不带状态行 —— 工具执行期间客户端只能干等，
   要进度提示仍只能用 `scripts/stream_cli.py`。错误也只有一个 `message` 字符串，没有状态码语义
   （流一旦开始就只能回 200）。图停在人工确认时流式只能改发 `error` 事件，办理仍旧不可用（见第 10 条）。
   `eval/runner.py` 仍旧走 `invoke()`，同一条 case 在流式与非流式下是否给出同一个回答没有机器判定。
10. **Chat API 只做到「单轮问答 + SSE 流式」**：两条通道都不办理人工确认
    （同步返回 501、流式发 `error` 事件，该线程留下一个待确认中断与一条没有工具结果的 AI 消息，
    只能换会话或改用 `hitl_cli.py`）；
    没有认证与授权，缺口 8 里的「`thread_id` 就是访问凭证」在网络端口上更该优先补 ——
    任何能访问端口的人拿到别人的 `thread_id` 就能接着那段对话说；没有速率限制与请求体上限，
    长回答会一直占着一次线程池槽位；会话仍是进程内 `InMemorySaver`，重启即全部丢失，也没有会话与确认的持久化审计；
    `eval/runner.py` 不经 HTTP，所以「同一句话经 API 与经 CLI 是否给出同一个回答」没有机器判定。
