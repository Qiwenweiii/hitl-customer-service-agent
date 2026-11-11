# Customer Service Agent 项目最终复盘

> 项目定位：一个面向电商客服场景的综合型 LLM Agent 项目
> 核心技术：Python + LangChain + LangGraph + RAG + Tool Calling + HITL + Streaming + HTTP API（FastAPI / SSE）+ Agent Evaluation + 会话归属校验
> 开发方式：Vertical Slice + AI Coding + pytest + Agent Evaluation
>
> 图形文档：[系统架构图](docs/架构图.md)（分层、依赖方向、被架构守卫禁掉的边）与
> [流程图](docs/流程图.md)（一次请求、HITL 暂停恢复、HTTP JSON/SSE 双形态、RAG、评测隔离）。
> 图**只在这两份文件里维护**，本书与 README 都只链接 —— 同一张图画三份必然漂移。

------

# 1. 项目概述

## 1.1 项目目标

构建一个能够处理电商客服问题的 Agent，例如：

- 查询订单
- 查询物流
- 查询企业客服知识
- 取消订单
- 申请退款

项目重点不是简单实现一个“聊天机器人”，而是完整实践：

- Agent
- Tool Calling
- Service Layer
- RAG
- HITL
- Streaming 输出通道
- HTTP API（JSON 与 SSE）
- LangGraph
- pytest
- Agent Evaluation
- AI Coding Workflow
- 工程化测试与验收

------

# 2. 为什么需要 Agent

普通的：

```text
用户问题
 ↓
LLM
 ↓
回答
```

适合：

- 知识问答
- 文本生成
- 总结
- 改写

但客服系统需要处理真实业务：

```text
查询订单
查询物流
查询退款规则
取消订单
申请退款
```

这些操作需要：

- 查询真实数据
- 执行业务规则
- 修改数据
- 进行人工确认

因此需要：

```text
用户
 ↓
LLM
 ↓
Agent
 ↓
Tools
 ↓
Services
 ↓
Data / RAG
```

Agent 的核心价值是：

> **让 LLM 不只是“回答问题”，而是能够根据用户意图选择并调用外部能力完成任务。**

------

# 3. Agent 架构与完整执行流程

这是整个项目最核心的架构。

## 3.1 整体架构

```text
                    用户
                     │
                     ▼
                    LLM
                     │
             理解意图 / 选择 Tool
                     │
                     ▼
                   Agent
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Tool        Tool       Tool
          │          │          │
          ▼          ▼          ▼
       Service    Service      RAG
          │          │          │
          ▼          ▼          ▼
        Data       Data      Knowledge
```

项目中可以简单理解为：

```text
LLM 决定做什么
Agent 负责组织执行
Tool 提供能力入口
Service 决定业务上怎么做
Data / RAG 提供实际数据
```

这张图画的是**执行链**，刻意省掉了入口层（CLI / HTTP）与展示层（流式事件筛选）。
完整的分层、依赖方向，以及被 20 条源码级架构守卫禁掉的边，只在
[docs/架构图.md](docs/架构图.md) 维护。

------

## 3.2 LLM 的角色

LLM 主要负责：

### ① 理解用户意图

例如：

> “帮我看看订单 10001 到哪了？”

LLM 判断：

```text
意图：查询物流
订单号：10001
```

------

### ② 选择 Tool

例如选择：

```text
query_logistics
```

------

### ③ 提供 Tool 参数

例如：

```json
{
  "order_id": "10001"
}
```

------

### ④ 根据 Tool 返回结果决定下一步

例如：

```text
query_order
    ↓
得到订单信息
    ↓
发现用户还需要物流
    ↓
query_logistics
```

所以 LLM 不一定只调用一次 Tool。

------

### ⑤ 生成最终回答

Tool 返回：

```text
订单 10001
商品：iPhone 16
物流状态：运输中
最新节点：深圳转运中心
```

LLM 再将这些结果组织成：

> 您的订单目前正在运输中，最新物流节点是深圳转运中心。

------

## 3.3 Agent 的角色

Agent 可以理解为：

> **连接 LLM 与各种外部能力的执行编排层。**

它负责：

```text
用户输入
 ↓
LLM
 ↓
是否需要 Tool？
 ↓
调用 Tool
 ↓
获得 Tool Result
 ↓
再次交给 LLM
 ↓
是否继续调用 Tool？
 ├── 是 → 再调用
 └── 否 → 最终回答
```

因此：

```text
Agent ≠ LLM
```

更准确地说：

```text
Agent = LLM + Tools + 执行循环/编排机制 + 状态
```

------

## 3.4 Tool 的角色

Tool 是 Agent 可以使用的能力。

当前项目：

```text
query_order
query_logistics
search_knowledge
cancel_order
refund_order
```

Tool 的主要职责：

1. 向 LLM 暴露能力
2. 定义参数
3. 调用 Service
4. 返回结构化结果
5. 必要时处理 HITL

例如：

```text
query_order
    ↓
OrderService.get_order()
    ↓
返回订单信息
```

Tool 不应该承担大量业务规则。

例如：

```text
❌ Tool 自己判断：
if status == "paid":
    ...
```

更合理：

```text
Tool
 ↓
OrderService
 ↓
业务规则
```

------

## 3.5 Service 的角色

Service 是业务逻辑层。

例如：

```text
OrderService
LogisticsService
RefundService
```

Service 负责：

- 数据查询
- 数据修改
- 业务规则
- 状态判断
- 业务结果

例如取消订单：

```text
OrderService.check_cancel()
```

负责判断：

```text
订单不存在？
订单已经取消？
订单已发货？
订单是否允许取消？
```

而不是让 LLM 自己判断。

------

## 3.6 Data 层

当前项目使用：

```text
data/mock/orders.json
data/mock/logistics.json
```

通过：

```text
app/data_loader.py
```

加载为 Pydantic 模型。

架构上：

```text
Service
 ↓
data_loader
 ↓
JSON
```

未来可以替换成：

```text
Service
 ↓
Repository / API
 ↓
MySQL / PostgreSQL / Redis / 第三方 API
```

上层 Agent / Tool 基本不需要改变。

这就是 Service 边界存在的价值之一。

------

## 3.7 完整 Agent 执行流程

例如用户：

> “我的订单 10001 到哪了？”

完整流程：

```text
① 用户输入
        │
        ▼
② LLM 理解用户意图
   “用户想查询物流”
        │
        ▼
③ LLM 选择 Tool
   query_logistics
        │
        ▼
④ LLM 生成参数
   {"order_id": "10001"}
        │
        ▼
⑤ Agent 执行 Tool
        │
        ▼
⑥ Tool 调用 Service
   LogisticsService
        │
        ▼
⑦ Service 查询 Data
        │
        ▼
⑧ Tool 返回物流结果
        │
        ▼
⑨ Tool Result 回到 LLM
        │
        ▼
⑩ LLM 判断：
   是否还需要其他 Tool？
        │
        ├── 是 → 再调用 Tool
        │
        └── 否
             ↓
        ⑪ LLM 生成最终回答
```

------

## 3.8 多 Tool 调用

例如：

> “订单 10001 是什么商品，现在到哪了？”

可能需要：

```text
query_order
     ↓
订单信息

query_logistics
     ↓
物流信息
```

然后：

```text
两个 Tool Result
       ↓
      LLM
       ↓
最终回答
```

这也是 Agent 与简单 RAG 问答的重要区别之一。

------

## 3.9 Tool Calling 的本质

可以把 Tool Calling 理解成：

```text
LLM：

“我判断现在需要调用
query_order，
参数是 order_id=10001。”
```

LLM 本身并没有真正执行：

```python
query_order(...)
```

而是输出一个结构化的 Tool Call：

```text
tool_name
arguments
```

Agent 运行时发现：

```text
LLM 要调用 query_order
```

于是实际执行：

```python
query_order(order_id="10001")
```

再把结果返回给 LLM。

所以：

> **LLM 负责提出调用请求，Agent 负责执行调用。**

------

# 4. RAG 在 Agent 中的角色

RAG 并不是 Agent 本身。

当前项目中：

```text
search_knowledge
       ↓
RAGService
       ↓
Retriever
       ↓
Reranker
       ↓
Knowledge Base
```

RAG 用于回答：

- 企业客服政策
- 退款规则
- 配送规则
- 售后说明

例如：

> “退款期限是多少？”

Agent：

```text
search_knowledge
```

RAG 返回相关知识：

```text
普通商品签收后 7 天内可以申请退款……
```

然后：

```text
RAG Result
 ↓
LLM
 ↓
最终回答
```

------

## 4.1 为什么 RAG 不能直接完成退款

RAG 的职责是：

```text
查知识
```

而不是：

```text
执行操作
```

例如：

```text
RAG：
“符合以下条件可以退款。”

但它不能：
“修改订单 payment_status。”
```

因此：

```text
知识问题
→ RAG

业务操作
→ Tool + Service
```

------

## 4.2 当前项目 RAG 架构

```text
search_knowledge
       ↓
RAGService.query()
       ↓
Retriever
       ↓
Reranker
       ↓
相关性判断
       ↓
Context
       ↓
Agent LLM
       ↓
最终回答
```

一个关键设计：

> `search_knowledge` 只返回知识上下文，不负责生成最终答案。

因此不会出现：

```text
Agent LLM
   ↓
search_knowledge
   ↓
RAG 内部再调用一个 LLM
   ↓
两个 LLM 混在一起
```

当前设计更加清晰：

```text
Agent 是唯一最终回答者。
```

------

# 5. HITL：Human-in-the-Loop

HITL：

> Human in the Loop

即：

```text
AI 执行任务
 ↓
遇到需要人工确认的操作
 ↓
暂停
 ↓
等待用户
 ↓
用户确认
 ↓
继续执行
```

------

## 5.1 为什么取消订单需要 HITL

例如：

> “帮我取消订单 10001。”

不能：

```text
LLM
 ↓
直接修改订单
```

更合理：

```text
用户请求
 ↓
Agent
 ↓
cancel_order
 ↓
检查是否允许取消
 ↓
需要人工确认
 ↓
interrupt()
 ↓
用户确认
 ↓
继续
 ↓
修改订单
```

------

## 5.2 HITL 的关键点

当前项目中：

```text
interrupt()
```

放在 Tool 层。

Service 不负责：

```text
interrupt()
```

因为：

```text
Service = 业务逻辑
Tool = Agent 交互能力
```

------

## 5.3 为什么不是所有修改操作都必须 HITL

HITL 的判断标准不是：

> “只要修改数据，就一定要人工确认。”

而应该考虑：

- 操作风险
- 用户预期
- 是否不可逆
- 是否涉及资金
- 是否涉及重要业务状态
- 产品要求

因此 HITL 是一个**业务风险设计问题**。

------

## 5.4 thread_id 与 Checkpointer

HITL 暂停后需要恢复。

因此需要：

```text
Checkpointer
+
thread_id
```

例如：

```text
thread_id = "demo-user-001"
```

第一次执行：

```text
Agent
 ↓
interrupt
 ↓
保存状态
```

恢复：

```text
Command(
    resume={"approved": True}
)
```

并使用：

```text
同一个 thread_id
```

才能继续原来的执行状态。

注意：

```text
thread_id
≠ Python 线程 ID
≠ order_id
≠ customer_id
```

它表示的是：

> **一次 Agent 执行/对话状态的持久化标识。**

同一个 `thread_id` 还承载第二件事：

```text
短期对话记忆
```

因为 `messages` 通道是追加式的（见第 8 章）。

所以是一份存储、两个用途：

```text
一份 Checkpointer + 一个 thread_id
 ├── HITL 的暂停与恢复
 └── 多轮对话的上下文
```

换 `thread_id`，就是换一段全新对话。

2026-10-17 之后这段多了一个前提：**thread_id 是地址，不是凭证**。
续写一段会话需要同时出示该会话新建时签发的 `session_token`，否则一律 403（见第 10.8 节）。

------

## 5.5 当前 HITL Resume Contract

当前项目明确规定：

```python
Command(
    resume={
        "approved": True
    }
)
```

或者：

```python
Command(
    resume={
        "approved": False
    }
)
```

这是当前项目的恢复协议。

不使用：

```python
Command(resume={})
```

因为当前 LangGraph 版本下空字典不会正常恢复。

------

# 6. Cancel Order：取消订单

## 6.1 业务规则

当前项目：

```text
pending → 可以取消
paid → 可以取消
shipped → 不允许
delivered → 不允许
cancelled → 已取消
不存在 → 找不到
```

------

## 6.2 业务判断放在哪里

放在：

```text
OrderService
```

例如：

```text
check_cancel()
```

这是一个只读检查。

然后：

```text
cancel_order()
```

真正执行修改。

结构：

```text
Tool
 ↓
OrderService.check_cancel()
 ↓
CAN_CANCEL
 ↓
HITL
 ↓
用户确认
 ↓
OrderService.cancel_order()
 ↓
update_order_status()
```

------

## 6.3 为什么先检查，再 HITL

不能一上来：

```text
interrupt()
```

然后告诉用户：

> “是否确认取消？”

如果订单本身：

```text
已经发货
```

那么根本不应该进入确认流程。

正确：

```text
先判断资格
 ↓
不允许 → 直接返回
允许 → HITL
```

------

# 7. Refund Order：退款

退款单独设计为一个 Vertical Slice。

## 7.1 为什么退款与取消分开

取消订单：

```text
order_status
```

退款：

```text
payment_status
```

两者不是一回事。

例如：

```text
订单：
order_status = cancelled
payment_status = paid
```

表示：

> 订单已经取消，但钱还没有退。

因此用户之后可能继续：

> “我要退款。”

------

## 7.2 当前退款规则

当前项目采用简化规则：

```text
未付款
→ NOT_PAID

已经退款中
→ ALREADY_REFUNDING

已经退款
→ ALREADY_REFUNDED

超过退款期限
→ REFUND_EXPIRED

需要拒收后退款
→ NEED_REJECT_DELIVERY

没有配送时间
→ NO_DELIVERY_TIME

满足条件
→ REFUNDING_SUBMITTED
```

退款期限：

```text
7 天
```

------

## 7.3 退款流程

```text
用户
 ↓
Agent
 ↓
refund_order
 ↓
RefundService
 ↓
检查：
  是否存在
  是否付款
  是否已经退款
  是否超过期限
  是否需要拒收
 ↓
允许退款
 ↓
HITL
 ↓
用户确认
 ↓
执行退款状态变更
```

------

## 7.4 为什么退款规则不交给 RAG

例如：

```text
RAG：
“退款期限为 7 天。”
```

这只能提供知识。

真正判断：

```text
这个订单现在是否可以退款？
```

应该由：

```text
RefundService
```

执行。

因为这是：

> **确定性的业务规则。**

而不是知识问答。

------

## 7.5 当前退款系统的简化范围

当前项目没有实现真实支付系统中的：

- 支付网关
- 支付回调
- 退款流水
- 幂等控制
- 对账
- 部分退款
- 真实支付状态同步

这是刻意控制项目范围，而不是遗漏。

------

# 8. Memory：短期对话记忆

当前 Agent 是：

> **多轮、任务型 Agent。**

同一 `thread_id` 内，上一轮的话会被带回来。

第一轮：

```text
用户：
ORD-2026-0004 帮我查一下

Agent：
调用 query_order → 返回结果
```

第二轮：

```text
用户：
那它现在到哪了？

Agent：
从对话历史解析出“它” = ORD-2026-0004
 ↓
调用 query_logistics
```

本轮的改动**不是新增一个 Memory 组件**。

------

## 8.1 记忆来自哪里

> LangGraph Agent 的短期 Memory 可以通过 `messages + Checkpointer + thread_id` 实现：Agent 执行过程中产生的 HumanMessage、AIMessage、ToolMessage 等会自动进入 `messages`，Checkpointer 负责保存/恢复 Agent State，`thread_id` 放在调用时的 `config["configurable"]` 中，用于标识和隔离不同会话。

来自 LangGraph 本来就有的两样东西：

```text
Checkpointer
+
thread_id
```

`messages` 通道是追加式 reducer。

所以：

```text
同一个 thread_id 再次 invoke
 ↓
上一轮的 Human / AI / Tool 消息都在上下文里
```

没有引入：

```text
Redis
数据库
向量库
独立的 Memory 组件
```

**记忆在这里不是新模块，是状态管理。**

但载体一旦缺失不会报错：

```text
不注入 checkpointer
 ↓
invoke 照样接受 thread_id
 ↓
只是没有地方落状态
 ↓
每一轮都是全新上下文
```

这与 5.4 的 HITL、以及 6.1 的 `Command(resume={})` 是同一类失败：
**库把配置缺失降级成了静默行为。**

------

## 8.2 真正要改的是提示词

机制一直都在。

禁住多轮的是系统提示自己原来的规则：

```text
本轮没有订单号 → 先向用户索要订单号
```

这条把「指代」判成了「缺信息」。

所以这一轮改的是口径：

```text
本轮没有订单号
 ↓
先看对话历史里有没有唯一可沿用的订单号
 ↓
有 → 照常调用对应工具
没有 → 才向用户索要
```

三条硬约束一起写进提示：

```text
历史里出现过多个订单 → 列候选让用户选，不许猜
沿用的只能是订单号 → 状态、物流本轮必须重新查
上一轮人工批准 ≠ 本轮已批准 → 结论一律由本轮工具返回值判定
```

------

## 8.3 怎么验证的

12 个离线测试，**不调用真实 LLM**。

用一个「回显式假模型」把断言钉在链路上，而不是钉在模型智力上：

```text
假模型只能看到真正送达它的那份 messages
 ↓
第二轮解出正确订单号
 ↓
这个号只能来自历史
```

一个必须记住的陷阱：

```text
系统提示里的示例订单号 = ORD-2026-0001
真实 Mock 数据里也有 ORD-2026-0001
 ↓
扫描历史时不排除 SystemMessage
 ↓
模型没记住上文也能答对，测试假绿
```

反向也验证过：

```text
checkpointer=checkpointer 改成 checkpointer=None
 ↓
12 条测试里 7 条立刻转红
```

------

## 8.4 长期 Memory 仍然没有做

当前只到“同线程累积”：

```text
用户
 ↓
thread_id
 ↓
Checkpointer（进程内）
 ↓
Agent
```

跨会话的用户画像与长期偏好仍不在范围内。

------

## 8.5 当前边界

```text
进程退出 → 记忆与待确认请求一起丢失
会话很长 → 消息线性增长，会撞模型上下文上限
拿到别人的 thread_id → 过去就能接着那段对话说；2026-10-17 已补归属校验（续轮须带该会话的 session_token，见 10.8）
多轮指代 → 没有评测 case，pytest 只证明上下文送达与线程隔离
```

------

# 9. Streaming：输出通道

前面几章解决的是：

```text
Agent 做的事对不对
```

这一章解决的是另一件事：

```text
用户等多久、以及等待过程中看到什么
```

它不新增任何业务能力，改的是**输出形态**。

------

## 9.1 为什么需要流式

`invoke` 的形态是：

```text
用户提问
 ↓
（Agent 内部：调工具 → 拿结果 → 再想 → 再调 → 组织语言）
 ↓
完整答复一次性返回
```

问题在最后一段：多轮 Tool Calling 时，整条链路可能十几秒，而这段时间屏上什么都没有。

流式把它变成：

```text
用户提问
 ↓
第一句「我帮您查一下」→ 立刻上屏
 ↓
调工具 → 一行状态提示
 ↓
工具结果回来 → 正文逐字继续
```

关键是：

> **用户看到的总耗时没变，但"从无声到有声"的时间变了。**

这是体感差异，不是性能差异。

------

## 9.2 `stream_mode` 选哪个

LangGraph 的 `stream` 有多种读法：

```text
"values"      每步之后的完整状态
"updates"     每个节点产出的增量
"messages"    模型与工具的逐条消息（含文本分片）
```

本项目选：

```text
stream_mode="messages"
```

因为只有它能给到**token 级**的文本增量：

```text
values / updates 的粒度是"一步"
messages 的粒度是"一个分片"
```

代价是 messages 模式混着吐多种消息（见 9.3），需要自己筛。

------

## 9.3 哪些事件能给人看

一条流里实际会出现：

```text
AIMessageChunk   模型正文分片          → 给人看
ToolMessage      工具返回原文          → 不给
tool_calls       调用结构（JSON 参数）  → 不给
metadata         节点名/模型名/thread   → 不给
```

因此展示层的唯一入口是：

```text
stream_text(message)
 ↓
不是 AI 消息 → 返回空串
是 AI 消息   → 取文本
```

为什么 `ToolMessage` 必须挡掉：

它的内容是喂给模型的中间数据——订单 JSON、政策原文、内部指令。
直接打到终端等于把原始工具结果甩给用户，既难看又泄露内部字段。

------

## 9.4 状态文案由程序写死

工具执行时给用户看的那行提示：

```text
[正在查询订单...]
```

来自一张**按工具名索引的常量表**：

```text
TOOL_STATUS_LABELS = {
    "query_order":  "[正在查询订单...]",
    ...
}
未登记的工具 → "[正在处理...]"
```

不是：

```text
让 LLM 生成一句进度文案
```

原因不是美观，是安全：

```text
对外输出的任何文本都是可被诱导的出口。
模型只决定"调哪个工具"，不决定"给用户看什么"。
```

未登记的工具也只能落到那句通用提示，**不回显原始工具名**。

------

## 9.5 两个分片陷阱

### 陷阱一：`type` 不是 `"ai"`

```text
AIMessage.type       == "ai"
AIMessageChunk.type  == "AIMessageChunk"
```

所以判定不能写成：

```python
if message.type == "ai":   # 整条流的正文全丢掉
```

本项目写成：

```python
isinstance(message, AIMessage)   # Chunk 是 AIMessage 的子类，一条判定覆盖两种
```

### 陷阱二：一次工具调用被拆成多片

```text
分片 1  带工具名 + 残缺参数 JSON
分片 2  只有参数碎片
分片 3  只有参数碎片
```

如果每片都打一行状态，屏上会出现三遍「正在查询订单」。

处理方式：

```text
按 call id 去重（没有 id 时退化成 index）
seen 集合由每条流自己持有
```

`seen` 不是跨轮状态——它是**一条流内的去重记忆**。

------

## 9.6 HITL 在流式下怎么读

`invoke` 时代读中断很直接：

```python
result["__interrupt__"]
```

流式不行：

```text
stream_mode="messages" 不把中断当事件吐出来
图暂停时，流只是安静地结束
```

所以待确认信息只能从状态里读：

```text
agent.get_state(config)
 ↓
state.tasks[*].interrupts
 ↓
interruption.value 就是确认 payload
```

裁决后用同一条流函数再开一次：

```python
Command(resume={"approved": True})
```

恢复流的第一条事件是工具结果，随后的文本增量才是最终答复。

这与第 8.1 节那条失败模式同源：
**框架把"没有完成"降级成了"看起来已经完成"。**

------

## 9.7 这一层落在哪里

```text
app/agent/streaming.py    展示层（纯函数，不 print、不 input、不含业务规则）
scripts/stream_cli.py     CLI 入口（逐字上屏 + 状态行 + 人工裁决）
```

分层与依赖方向见 [docs/架构图.md](docs/架构图.md)，
一次请求在流式下的完整事件序列见 [docs/流程图.md](docs/流程图.md)。

------

# 10. HTTP API：从 CLI 到服务

CLI 能跑通，不代表别人能用。这一章把入口换成 HTTP。

------

## 10.1 变的是入口，不是能力

```text
新增：
  POST /api/chat

没有新增：
  业务规则、Tool、Service、RAG、HITL 判定
```

四个入口共用同一个装配函数（各自是独立进程，进程内只组装一次 Agent）：

```text
scripts/chat_api.py       HTTP
scripts/hitl_cli.py       CLI（可裁决）
scripts/stream_cli.py     CLI 流式（可裁决）
eval/runner.py            评测
```

------

## 10.2 API 层只做契约翻译

这一层的职责边界写死成三件事：

```text
请求/响应的 Pydantic 契约
thread_id 的组装
把 Agent 结果翻译成响应
```

明确不做：

```text
不判断能不能取消、该不该退款
不复述订单字段
不写第二份 content 解析（复用 stream_text）
```

应用组装也只有一处：

```text
app/main.py  create_app(model, checkpointer)
 ↓
建 Agent 一次 → 注入 app/api/chat.py create_chat_app(agent)
```

好处是测试可以塞一个假 Agent 进来，整站离线跑通，不需要真实大模型。

------

## 10.3 `thread_id` 就是会话身份

```text
请求带 thread_id → 延续那段对话
请求不带        → 服务端新建并回显
```

JSON 响应里有字段承载它；SSE 正文没有，所以走响应头：

```text
X-Thread-Id: chat-xxxx
```

同一份 checkpointer + 一个 `thread_id`，仍然是两个用途（第 5.4 节）：

```text
多轮上下文
HITL 的暂停与恢复
```

VS4（2026-10-17）起，"会话身份"拆成了两层：

```text
thread_id      会话地址：决定状态挂在哪一段对话上
session_token  会话凭证：新建时由服务端签发，证明持有者有权续写这个地址
```

细则见第 10.8 节。

------

## 10.4 一个端点，两种形态

选择是：

```text
POST /api/chat  +  Accept: text/event-stream   → SSE
POST /api/chat  +  其他                         → JSON
```

而不是：

```text
新增字段 stream=true
新增端点 /api/chat/stream
```

一个必须小心的细节：

```text
curl 与多数 HTTP 客户端默认 Accept 是 */*
```

`*/*` 表示"什么都行"，不是"要流式"。
判定只认显式的 `text/event-stream`，否则这次改动会悄悄改变所有既有客户端拿到的响应形态。

------

## 10.5 SSE 只有三类事件

```text
token   {"content": "增量文本"}
done    {}                    正常收尾
error   {"message": "..."}    终止，后面不再有 done
```

客户端把依次到达的 `token` 正文拼起来即为完整答复。

分帧靠一行 `data:`，所以正文里的换行必须转义：

```python
json.dumps(data, ensure_ascii=False)
```

- 转义换行 → 一条事件永远只占一行，不会把 SSE 的分帧撞断；
- `ensure_ascii=False` → 中文不变成 `\uXXXX`。

出错时给客户端的是一句固定话术，异常原文与 traceback 只进服务端日志。

------

## 10.6 HTTP 上的 HITL：明确拒绝

用户完全可能在 `message` 里直接说「帮我取消 ORD-2026-0001」。
图会停在待确认点，而这个端点**没有回答确认问题的能力**。

此时不能：

```text
回一条空 answer   → 客户端会以为处理过了
```

本项目的做法是两条路径都明确拒绝：

```text
JSON 分支   → 检查 result["__interrupt__"] → HTTPException(501, ...)
SSE 分支    → 收尾读 get_state → 发 error 事件，不发 done
```

拒绝文案只说明"这一笔没有办理、订单数据未被改动"，
**不顺着用户意图给出办理结果**（否则等于用文案假装办成了）。

收到 501 之后：这一次不写数据，同一会话之后的普通请求照常回答；
代价是该线程留下一个待确认中断和一条没有工具结果的 AI 消息。
所以客户端应该改用人工确认入口，或换一个全新会话。

------

## 10.7 一条只有真实模型才会暴露的 bug

第一版 SSE 收尾是这样判的：

```text
本轮有没有文本增量 → 有就发 done
```

离线测试全绿。真实模型 smoke 时红了。

因为模型常态是：

```text
先回一句「好的，我来帮您查询是否可以取消」
 ↓
再调用 cancel_order
 ↓
图停在确认位
```

那句客套话已经作为 `token` 到达客户端了，图其实才刚开始执行。
按"有没有增量"来判断，就会把「没办理」报成「答完了」，而 SSE 的事件发出去就**收不回**。

修复后的口径：

```text
每条流收尾都无条件读一次线程状态
停在待确认点 → error，否则才 done
```

对应的回归测试钉的是"判别性"，四条断言分别挡住四个方向：

```text
确认位必须在 get_state 之前判定
状态读取必须真的发生（state_reads == 1）
停在确认位时不能有 done
出错时也不能有 done
```

这条 bug 的根源在测试替身：

```text
假流式模型的事件形状是按"我以为协议长什么样"写的
我以为"停在确认位时不会有任何文本增量"
于是那条错误捷径正好通过了所有断言
```

结论：

> **通道类能力的完成条件，包含一次真实模型 smoke。**

------

## 10.8 会话归属校验（VS4，2026-10-17）

前面 10.1–10.7 留下一个暴露面：

```text
thread_id 是可以被伪造和猜到的字符串，
而服务端过去对它无条件信任
```

这一切片把"会话身份"拆成地址与凭证两层：

```text
新建会话（请求不带 thread_id）
 ↓ 服务端生成 thread_id，同时签发 session_token
 ↓ JSON 在响应体回显；SSE 用 X-Session-Token 响应头回显（仅此一次，存储里只有 sha256）
续轮（带 thread_id）
 ↓ 必须同时携带 session_token，且与该 thread 绑定相符
不符 → 403
```

三个设计点，各防一种具体攻击：

1. **签发只出现在"新建会话"分支。** 校验失败绝不 fallback 签发，否则伪造任意 thread_id 都能当场领到合法凭据，
   归属校验形同虚设；有专门用例断言"未知 thread 不自动签发"。
2. **错误凭证绝不触发轮换。** 若校验失败就轮换 token，攻击者等于获得"猜一次 → 被签发一份新凭据"的免费通道；
   防接管的关键用例钉的是"错误凭证被拒之后，原 token 仍然有效"。
3. **"未知 thread"与"凭据不符"共用同一条 403 文案。** 错误响应不能泄露某个 thread_id 是否存在。

存储侧（`app/session_store.py`）：

```text
SessionStore（create_app 启动时创建一次，所有请求共用，threading.Lock 保护）
 ├── 内存里只有 token 的 sha256 哈希，没有明文可丢
 ├── hmac.compare_digest 定时比较，防通过响应时间逐位猜哈希
 └── 存储读写收在三个私有方法里，之后换 SQLite 不动接口与调用处
```

边界要如实说清：

```text
session_token 不是身份认证 —— 它只证明"持有者可续写该会话"
匿名者仍可随时新建会话；用户级认证、鉴权、凭证过期、速率限制仍在 V2 清单（第 22 章）
```

测试侧：`tests/test_session_ownership_slice.py` 15 个离线用例（SessionStore 单测 + HTTP），
外加一条装配守卫 `test_create_app_always_injects_a_session_store`——
`create_chat_app` 的 `session_store=None` 分支只作为内部测试 seam 存在，生产入口不可能走到无校验路径。
本切片还更新了 6 条把"裸 thread_id 期望 200"写死的旧用例：那些断言固化的正是漏洞本身，
契约变更时它们集体转红，恰恰证明旧用例对契约有判别性。

------

## 10.9 当前边界

```text
无用户级认证：会话归属校验已落地（2026-10-17，见 10.8），但匿名新建会话仍然敞开，没有身份/角色/权限
HITL 不办理：两个高风险动作在 HTTP 侧只能被拒答
InMemorySaver：重启失忆，挂起的确认也丢
无心跳与续传：断线之后只能重问
eval/runner.py 仍走 invoke：流式与 HTTP 只有 pytest 覆盖
```

这三条边界原本全部排在 V2 路线的"把通道补成可用服务"那一格（第 22 章）；
其中 thread_id 归属校验一项已于 2026-10-17 落地（见 10.8），其余仍待补。

------

# 11. pytest 与 Agent Evaluation

这是项目工程化非常重要的一部分。

两者不是重复的。

------

## 11.1 pytest 测什么

pytest 主要测试：

```text
代码
业务规则
Service
Tool
数据处理
输出通道契约（流式事件筛选、HTTP JSON 与 SSE）
分层与依赖方向（读源码的架构守卫）
```

当前规模：

```text
353 个用例 = 295 个测试函数参数化展开（16 个文件）
其中 20 条是读源码的架构守卫
复跑无需前缀：tests/conftest.py 在收集阶段强制关闭 LangSmith 追踪
（追踪上报序列化会排空假模型迭代器、造成 35 个成片假红，根因见《项目复盘笔记》6.10）
```

例如：

```text
订单状态 paid
 ↓
check_cancel()
 ↓
CAN_CANCEL
```

或者：

```text
订单不存在
 ↓
返回 NOT_FOUND
```

这些都是确定性测试。

------

## 11.2 Agent Evaluation 测什么

Evaluation 主要测试：

> **Agent 整体行为是否正确。**

例如用户：

> “查一下订单 10001 到哪了。”

要检查：

```text
是否选择 query_logistics？
参数是否正确？
Tool 是否真的执行？
最终回答是否包含正确结果？
```

这不是普通 Unit Test 最擅长测试的。

------

## 11.3 两者关系

```text
Unit / Integration Test
        ↓
      pytest
        ↓
测试代码和业务逻辑

        +

Agent Evaluation
        ↓
测试 Agent 行为
```

最终再加：

```text
Manual / UAT
```

完整体系：

```text
pytest
  ↓
Agent Evaluation
  ↓
Manual / UAT
```

------

# 12. Agent Evaluation 实现

当前项目没有直接依赖 LangSmith Evaluation，而是自己实现了最小 Evaluation Runner。

------

## 12.1 Evaluation 核心流程

```text
Dataset
   ↓
读取 Case
   ↓
真实 Agent
   ↓
记录：
  Tool Calls
  Tool Arguments
  Tool Results
  Final Answer
  Data Changes
   ↓
Assertions
   ↓
Report
```

------

## 12.2 Dataset

当前使用 YAML：

```text
eval/dataset/
```

每个 case 描述：

```text
用户输入
期望 Tool
期望参数
期望结果
最终回答要求
数据变化要求
```

例如可以表达：

```text
用户：
查询订单 10001

期望：
query_order
order_id = 10001
```

------

## 12.3 Runner

正式 Evaluation 入口：

```bash
uv run python -m eval.runner
```

注意：

```text
eval/runner.py
```

才是正式 Evaluation Runner。

------

## 12.4 min_runner 的定位

项目中的：

```text
eval/min_runner.py
eval/min_hitl_runner.py
```

只是：

> 用于理解 Evaluation / HITL 最小原理的教学 Demo。

不是正式 Evaluation 入口。

------

## 12.5 当前 Evaluation Case

目前：

```text
12 / 12 PASS
```

共：

```text
86 / 86 assertions PASS
```

包括：

- Tool 路由
- 参数
- Tool Result
- 最终回答
- HITL
- 数据变化
- 错误场景
- 多 Tool
- RAG
- 退款

几个必须写清楚的时间口径（都是从 `eval/results/` 与 `baseline.json` 反查出来的）：

```text
最后一次全量 86/86 全绿  2026-09-25 19:14（run-20260925-191419）
baseline 固化在          同日更早的 17:12 那一轮（baseline.json generated_at = 17:13:58）
2026-09-26 那次全量      85/86，唯一一条红是 E10 的禁词抖动
                        工具链路、业务结论、数据快照都正确，
                        模型末尾补了一句政策条件句被裸子串匹配判红
                        报红后连续单跑 E10 5 轮：1 红 4 绿
```

以及一条覆盖边界：

```text
eval/runner.py 走 invoke()
流式与 HTTP 通道只有 pytest 覆盖，没有进评测
```

**"跑过"不等于"全绿"，"业务侧全绿"也不等于"新通道被验证过"。**

------

## 12.6 为什么没有 LLM-as-Judge

当前项目 V1 没有加入：

```text
LLM Judge
```

原因是当前更关注：

```text
Tool 是否选对
参数是否正确
业务结果是否正确
状态是否正确
```

这些可以通过确定性断言验证。

例如：

```text
assert tool_name == "query_order"
```

比：

```text
LLM 判断“这个回答看起来是否合理”
```

更加稳定。

------

## 12.7 当前 Evaluation 的限制

当前 Evaluation：

```text
n = 1
```

没有：

```text
大规模测试集
概率统计
LLM-as-Judge
复杂 adversarial cases
Prompt Injection 专项测试
```

因此它是：

> **一个面向项目工程实践的 V1 Evaluation 系统。**

而不是完整的企业级 Evaluation Platform。

------

# 13. Evaluation 的反向验证

不仅要测试：

```text
正确代码 → PASS
```

还应该测试：

```text
故意破坏 → FAIL
```

当前项目做了 reverse validation。

例如故意修改：

```text
Tool 名称
参数
最终回答
数据变化
HITL 判断
```

然后对应 assertion 能够失败。

最终验证：

```text
18 个 targeted fake evidence breakages
→ 对应断言均能失败
```

这证明：

> Evaluation 不只是“自己跑自己绿”，而是真的具备发现错误的能力。

------

# 14. 为什么 AI Coding 仍然需要人工

AI Coding 并不意味着：

```text
AI 写代码
→ 人什么都不用管
```

真正的工作方式应该是：

```text
人
 ↓
定义需求
 ↓
决定架构
 ↓
决定业务规则
 ↓
告诉 AI
 ↓
AI 写实现
 ↓
人 Review
 ↓
测试
 ↓
人工验证
```

------

## 14.1 AI 适合做什么

AI 很适合：

- 编写重复代码
- 补充测试
- 实现 Service
- 实现 Tool
- 修改类型
- 重构
- 生成 YAML Case
- 查找潜在问题
- 根据明确架构补代码

------

## 14.2 人必须负责什么

人需要负责：

### ① 需求

例如：

> 退款和取消必须分离。

------

### ② 架构

例如：

```text
Tool
 ↓
Service
 ↓
Data
```

------

### ③ 业务规则

例如：

```text
paid 可以取消
shipped 不可以取消
```

------

### ④ 测试策略

例如：

> 必须验证退款后的 payment_status。

------

### ⑤ Review

重点看：

```text
架构有没有被破坏？
业务规则有没有错？
异常场景有没有漏？
Tool 有没有越界？
```

------

### ⑥ 最终验证

包括：

```text
pytest
Evaluation
Manual Test
```

------

## 14.3 AI 生成测试，人应该看什么

不需要逐行阅读 AI 生成的几百行测试。

重点看：

```text
1. 测了哪些业务规则？
2. 关键场景有没有覆盖？
3. assert 是否真的检查结果？
4. 有没有测试“必然失败”的错误？
5. 故意破坏代码后，测试会不会红？
```

核心原则：

> **AI 可以写测试代码，但人必须定义“什么才算正确”。**

------

# 15. Vertical Slice 开发方式

这个项目采用：

> Vertical Slice

而不是：

```text
先把所有 Tool 写完
再把所有 Service 写完
再写测试
最后拼起来
```

而是：

```text
一个业务能力
 ↓
完整打通
 ↓
验证
 ↓
再做下一个
```

------

## 15.1 query_order

完整链路：

```text
用户
 ↓
Agent
 ↓
query_order
 ↓
OrderService
 ↓
Data
 ↓
Tool Result
 ↓
LLM
 ↓
回答
```

然后：

```text
pytest
+
manual test
+
smoke test
```

------

## 15.2 query_logistics

同样完整打通：

```text
Agent
 ↓
query_logistics
 ↓
LogisticsService
 ↓
logistics.json
```

然后测试。

------

## 15.3 search_knowledge

```text
Agent
 ↓
search_knowledge
 ↓
RAGService
 ↓
Retriever
 ↓
Reranker
 ↓
Knowledge Base
```

------

## 15.4 cancel_order

```text
Agent
 ↓
cancel_order
 ↓
OrderService
 ↓
Eligibility
 ↓
HITL
 ↓
Data Mutation
```

------

## 15.5 refund_order

```text
Agent
 ↓
refund_order
 ↓
RefundService
 ↓
Eligibility
 ↓
HITL
 ↓
Payment State
```

------

## 15.6 Streaming（输出通道第一条）

```text
agent.stream(stream_mode="messages")
 ↓
展示层（stream_text / tool_status_lines / pending_confirmation）
 ↓
scripts/stream_cli.py 逐字上屏 + 状态行 + 人工裁决
```

与 15.1~15.5 的差别：

```text
业务能力一条没改
新增的只有"事件怎么选、什么文案能给人看"
```

完成条件里多了一项以前没有的：**必须跑一次真实模型**（见 10.7）。

------

## 15.7 Chat API（同步 JSON）

```text
POST /api/chat
 ↓
ChatRequest 契约
 ↓
agent.invoke(...)
 ↓
ChatResponse(answer, thread_id)
```

以及一条容易漏做的：

```text
触发 HITL → 明确拒绝（501），而不是回空 answer
```

------

## 15.8 HTTP 流式（SSE）

```text
Accept: text/event-stream
 ↓
StreamingResponse
 ↓
token* → done / error
```

这条切片的改动几乎全在"怎么把没做完说清楚"上：
收尾无条件读线程状态（10.7），出错只发固定话术，`thread_id` 走响应头回显。

------

## 15.9 会话归属校验（VS4）

```text
thread_id     → 会话地址（服务端生成，客户端不能自报）
session_token → 会话凭证（新建时签发，存储只留 sha256，仅此一次回显）
校验失败      → 403：不签发、不轮换、不透露该 thread 是否存在
```

这条切片一行业务规则都没改，改的是 HTTP 契约里**谁会吃到拒绝**（第 10.8 节）。
最硬的一步不在实现，而在 6 条把"裸 thread_id 期望 200"写死的旧用例集体转红——
它们固化的正是漏洞本身，更新到新契约之前先确认了这份"红"是有判别性的。

------

## 15.10 RAG 消融评测（2026-10-19）

```text
Baseline   纯向量：similarity_search_with_score，显式按距离升序（不依赖 retrieve() 的隐式降序）
Treatment  向量 + CrossEncoder 重排，不应用阈值 —— 主对比组
Online     线上链路：重排后过滤 >= MIN_RERANK_SCORE 再截断
判定       结果条目的 metadata (document_id, chunk_index) 与金标完全一致
```

这是第一次给检索层**自己的质量指标**——此前 RAG 只有 pytest 契约和架构守卫钉住"行为对不对"，
从来没测过"召回好不好"。23 条 chunk 级金标（`eval/rag_ablation_queries.json`，标注逐条对过真实切分内容）
跑出来的结果没有站在直觉那一边：Hit@3 纯向量 91.3%，加重排反而 87.0%；
唯一的重排负收益是"申请售后要准备点啥"被 rerank 从第 2 位压到第 8 位（0.064 vs Top1 0.698），
而两条三组全 miss 的口语化 query，期望片段在向量召回里就只排第 6/7——**问题在召回层，不在重排层**。
20 个 chunk 的小语料 + n=23，这个结论只作方向参考；生产链路一字未动，
`app/rag/` 也没为评测改过一行——评测脚本自己用 `snapshot_download(local_files_only=True)` 注入本地缓存模型解决离线运行。

## 15.11 Skill 机制（VS5，2026-10-22）

```text
阶段0  skills/refund-processing/SKILL.md + app/skills/loader.py
       frontmatter 解析与校验（name/allowed_tools/priority/body/triggers），失败抛 SkillValidationError
阶段1  app/skills/middleware.py::wrap_model_call
       按 state["active_skill"] 查预加载表：SOP 追加进 system prompt，tools 按白名单过滤并重排
阶段2  ::before_model
       最后一条 HumanMessage 按 triggers 规则命中→写 active_skill；未命中显式 None；AI/Tool 轮返回 None 不动 state
阶段3  tests/skills/test_skill_hitl_integration.py
       真实 interrupt + InMemorySaver + Command(resume) 钉住：跨 resume 保持 / 跨轮清除 / 工具循环不清
```

这条切片补的是**装配**：退款这类复杂业务的 SOP 措辞从写死的系统提示里搬进可插拔的 SKILL.md，
由 `SkillMiddleware`（langchain 1.4 `AgentMiddleware`）按会话状态动态拼进模型请求——
"未加载 skill 时模型看不见 refund_order"由 `request.override(tools=...)` 在请求组装点强制，
不改全局 `TOOLS`，Tool/Service 层一个字没动。

几个裁决型设计，全部是有意选择而非实现便利：
- **SOP 正文禁硬编码业务阈值**：金额档位、时效天数、到账时限一律写成"以政策检索结果和工具返回为准"，
  资格与分流结论只能来自工具返回——提示词里的数字会随政策变更静默漂移，还会诱导模型拿旧数字自行判定。
- **构造期预加载、重启生效**：skill 集合在进程启动时冻结，坏文件启动时被筛掉，运行期零磁盘 IO；
  单个 SKILL.md 校验失败只跳过该 skill、降级为无 skill 透传，配置事故不升级成对话中断。
- **"不表态"与"清除"分成两种返回值**：`before_model` 对非 HumanMessage 返回 `None`（保持 skill 跨工具循环
  与 HITL 恢复），对未命中的 HumanMessage 返回 `{"active_skill": None}`（用户改口即作废绑定）。
  若两者合并成"总是写 state"，恢复轮的 ToolMessage 会把 skill 洗掉，最终答复失去红线约束。

验证 28 个用例全离线：loader 11（含真实 SKILL.md 解析与 triggers 校验）、middleware 14（注入/路由单测 + 链式集成）、
HITL 集成 3（A 跨 resume 保持、B 跨轮清除、C 工具循环不清）；另有一条源码级依赖守卫——
`app/skills/` 禁止 import `app` 下任何其他模块（同目录 `exceptions` 除外），数据层不许反向寄生上层。
集成测试的 spy 必须挂在 `wrap_model_call` 的**下游 handler** 上记录请求：第一版探针记了注入前的入参 request，
"active_skill 命中却没注入"假象就是这么造出来的——断言对象要钉在真正到达模型的那份请求上。

------

前 5 条切片补的是**业务能力**，中间 3 条补的是**输出通道**（没新增一行业务规则，102 个用例全在测通道自身），
第 9 条补的是**边界**：同样不新增业务，第一次给 HTTP 契约加上"拒绝谁"的语义（403），16 个用例全离线。
第 10 条（15.10）性质又不同：不新增任何代码能力，第一次让"检索质量"变成一个可量化、可复跑的数字。
第 11 条（15.11）补的是**装配**：业务规则仍一行业没写，但 SOP 与工具可见性第一次变成可插拔的配置包。

------

# 16. 测试策略

当前项目遵循：

```text
实现
 ↓
Unit / Integration Test
 ↓
工程验证
 ↓
Manual Test
 ↓
Vertical Slice 完成
 ↓
Real-model Smoke Test
```

特别注意：

> **Vertical Slice 没有完成之前，不运行真实模型 Smoke Test。**

原因：

- 浪费 Token
- 反馈噪声大
- 代码本身还没稳定
- 很难判断问题到底来自代码还是模型

------

## 16.1 还有一层测的不是行为

上面那条流水线测的都是"跑得对不对"。
分层本身靠另一类测试守住：

```text
20 条源码级架构守卫
 ↓
用 inspect / ast 读实现
 ↓
断言依赖方向没有被写反
```

例如：

```text
Tool 层不许写业务规则（规则只在 Service 判定）
API 层不许 import services / data_loader / tools
应用组装只有一处：启动入口不许自己建 Agent 或 checkpointer
app 包里不许出现 print（IO 只属于入口脚本）
展示层不许认识 HTTP 与 SSE
app/skills/* 不许 import 除 exceptions 外的任何 app 模块（loader 保持纯配置读取）
```

每条都对应 `tests/` 里一个读源码的测试函数，例如
`test_api_layer_holds_no_business_rules`、`test_streaming_layer_knows_nothing_about_http_or_sse`、
`test_skills_modules_do_not_import_app`。

为什么它必须写成测试：

```text
写在文档里的架构约定，只约束读得到文档的人
写在测试里的架构约定，每次 pytest 都约束所有人
```

对应清单见 [docs/架构图.md](docs/架构图.md) 的"被守卫禁掉的边"一节。

------

# 17. 最终工程验收

当前项目最终状态：

```text
pytest：
353 passed（16 个文件 / 295 个测试函数参数化展开；2026-10-22 裸跑复验，`tests/conftest.py` 已强制关闭追踪兜底，见《项目复盘笔记》6.10）
其中 20 条是读源码的架构守卫
```

Agent Evaluation：

```text
12 / 12 PASS
86 / 86 assertions PASS
```

但要分清两件事：

```text
最后一次 86/86 全绿是 2026-09-25
2026-09-26 那次全量是 85/86，红的正是已知的 E10 禁词抖动
```

以及一条边界：

```text
评测 runner 走 invoke()
09-26 晚之后落地的流式与 HTTP 通道只有 pytest 覆盖，没有进 eval
```

Ruff：

```text
2026-10-22 复跑：Found 2 errors
两处均为 eval/rag_ablation.py 的 F541（表格字符串多余 f 前缀），不在 Skill 链路——
是 10-19 切片落盘后从未被 lint 覆盖过、这次复跑才暴露的老问题；--fix 可自动修，本轮白名单外未修
上一次全过仍是 2026-10-10
```

（仍未进 `dev` 依赖组，检查靠 `uvx ruff@0.6.9 check .` 临时拉版本。）

真实 Mock 数据：

```text
orders.json
未发生非预期变化
（2026-10-22 复跑时发现 ORD-2026-0002 有一行 payment_status 漂移——归属不明、不在任何测试链路上——
已 Edit 恢复为 HEAD 值，sha256 重新等于历史哨兵 761700bb…；过程见《项目复盘笔记》6.10）
```

入口清单（各自切片期间都经过真实模型跑通；本轮只重新复跑了 pytest 与 ruff，没有逐条重新启动这四个入口）：

```bash
uv run python scripts/chat_api.py       # HTTP（JSON 与 SSE）
uv run python scripts/stream_cli.py     # CLI 流式，可裁决人工确认
uv run python scripts/hitl_cli.py       # CLI 同步，可裁决人工确认
uv run python scripts/smoke_data.py     # 数据层冒烟（原先在 app/main.py，已挪出）
```

正式 Evaluation：

```bash
uv run python -m eval.runner
```

需要配真实模型密钥，会消耗 Token。

------

## 17.1 最终项目验证矩阵

| 验证方式         | 目的                 |
| ---------------- | -------------------- |
| pytest           | 代码与业务逻辑       |
| 源码级架构守卫   | 依赖方向与分层边界   |
| Agent Evaluation | Agent 行为           |
| Manual Test      | 真实交互             |
| Smoke Test       | 真实 LLM 集成        |
| Ruff             | 代码质量             |
| 数据完整性检查   | 防止测试污染真实数据 |

覆盖边界要说清楚：

```text
业务规则与数据变化   pytest + Evaluation
分层与依赖方向       pytest（读源码的守卫）
流式与 HTTP 通道     只有 pytest，没有进 Evaluation
Skill 装配与注入     只有 pytest，没有进 Evaluation（模型服从红线层未覆盖 skill 场景）
回答得体程度         人工翻结果文件，没有机器判定
```

------

# 18. 当前项目已知限制

当前项目不是生产级电商系统。

主要限制：

### 数据层

目前：

```text
JSON Mock
```

不是：

```text
MySQL / PostgreSQL
```

------

### 退款

没有真实支付系统。

------

### Evaluation

目前：

```text
n=1
```

测试规模有限。

还有一条更容易被忽略的边界：

```text
eval/runner.py 走 invoke()
流式与 HTTP 通道落在 pytest 里，没有进评测
"同一句话经 API 与经 CLI 是否给出同一个回答"目前没有机器判定
```

------

### Memory

短期记忆只做到"同线程累积"，四条边界里除第三条已于 2026-10-17 补上外都还在：

```text
进程退出即失忆（无跨进程持久化）
无窗口裁剪与摘要（长会话会撞上下文上限）
thread_id 没有用户级权限校验（会话归属校验已落地，见第 10.8 节）
多轮指代没有评测 case
```

长期会话记忆（跨会话的用户画像与偏好）暂未加入。

其中第三条的性质在 09-28 之后变了：

```text
只有 CLI 时   拿到 thread_id 需要能登录那台机器
有 HTTP 之后  任何能访问那个端口的人都能接着别人的对话说
2026-10-17   归属校验落地：续轮不带有效 session_token（未知或伪造的 thread）一律 403
```

------

### 安全

尚未针对：

```text
Prompt Injection
Tool Abuse
越权调用
恶意参数
```

进行完整安全测试。

HTTP 侧最前面的一格在 2026-10-17 补上了一半（会话归属校验，见第 10.8 节）：

```text
没有用户级认证：无注册/登录/角色，匿名者仍可随时新建会话
没有细粒度鉴权、没有速率限制
会话归属：伪造或未知的 thread_id、缺凭证或凭证不对的续轮 → 一律 403（已实测，非设想）
```

这是范围裁剪的结果，不是遗漏 —— "接着别人的对话说"已被归属凭证堵住，
但上面所有身份与配额类风险仍是现实。

------

### RAG

当前知识库规模较小。

------

### 输出通道

流式与 SSE 已可用，但只到"能正常跑完"这一层：

```text
无心跳、无断线续传：连接断了客户端只能重问
工具状态行只在 CLI 有，HTTP 侧没有（SSE 只有 token/done/error 三类事件）
流式中途出错时响应头早已发出，状态码改不了，只能发 error 事件
答复正文没有落库，客户端断线后无法回捞
```

------

### HITL

使用：

```text
InMemorySaver
```

因此应用重启后不会保留挂起状态。

并且 HTTP 侧**不办理人工确认**：

```text
JSON 分支 → 501 明确拒绝
SSE 分支  → error 事件收尾，不发 done
```

也就是说两个真正"办事"的动作（取消、退款）对网络另一侧完全不可用，
只能通过 CLI 入口裁决。

这些都属于后续 V2/V3 的演进方向。

### Skill

- 目前只有 refund-processing 一个 skill；路由是 triggers 子串规则匹配，没有语义/LLM 路由（skill_router），
  也没有多 skill 编排与组合。
- 构造期预加载换来实现运行期零磁盘 IO，代价是 skill 变更必须重启进程，不支持热更新。
- 注入层（SOP 送达、白名单生效）由 pytest 钉死；但"模型是否服从 SOP 红线"属于行为层，
  `eval/runner.py` 尚无对应 case——这一格是空的，与流式/HTTP 通道同一类验证空洞。

------

# 19. 核心工程原则总结

## 原则 1：LLM 不负责业务规则

不要：

```text
让 LLM 判断订单是否可以退款
```

应该：

```text
LLM
 ↓
Tool
 ↓
Service
 ↓
确定性业务规则
```

------

## 原则 2：Tool 不应该变成业务逻辑大杂烩

Tool 应该：

```text
定义能力
 ↓
接收参数
 ↓
调用 Service
 ↓
返回结果
```

复杂业务规则进入 Service。

------

## 原则 3：RAG 负责知识，不负责业务操作

```text
RAG：
“规则是什么？”

Service：
“这个订单现在是否符合规则？”
```

------

## 原则 4：HITL 根据风险决定

不是：

```text
所有写操作 → HITL
```

而是：

```text
根据业务风险与产品要求决定
```

------

## 原则 5：Agent Evaluation 与 pytest 不互相替代

```text
pytest
→ 代码正确吗？

Evaluation
→ Agent 行为正确吗？

Manual
→ 用户实际使用是否正常？
```

------

## 原则 6：AI Coding 不等于 AI 决策

AI 可以负责：

```text
实现
测试
重构
辅助分析
```

人负责：

```text
需求
架构
业务规则
验收标准
最终判断
```

------

## 原则 7：模型不决定"给用户看什么"

不要：

```text
让 LLM 生成进度文案
把工具原始返回直接甩到屏上
```

应该：

```text
LLM
 ↓
决定调用哪个 Tool

程序
 ↓
决定哪个事件给人看、配哪句状态文案
```

理由不是风格，是出口：

```text
任何对外可见的文本，都是可能被诱导的出口。
```

------

## 原则 8：框架会把"没完成"降级成"看起来正常"

本项目撞到的三起同源事故：

```text
不注入 checkpointer     → invoke 照样接受 thread_id，只是每轮都是全新上下文
图停在人工确认位        → messages 流不吐中断事件，只是安静结束
流式中途出错            → 响应头已发出，状态码改不了
```

它们都不抛异常。

所以判定不能靠返回值，要主动查状态：

```text
每条流收尾都无条件 get_state 一次
停在待确认点 → 明确说明"这一笔没办理"，而不是发一个看起来成功的信号
```

推论：

> **通道类能力的完成条件，包含一次真实模型 smoke。**
> 假模型证明的是"链路按我设计的形状接通"，真实 smoke 证明的是"那个形状确实存在"。

------

# 20. 面试版项目描述

可以将项目浓缩成：

> 我独立设计并实现了一个基于 LangGraph 的电商客服 Agent，支持订单查询、物流查询、企业知识库问答、订单取消和退款等场景。系统采用 Tool + Service 分层，LLM 负责意图理解和 Tool Calling，Service 负责确定性的业务规则，RAG 用于企业知识检索，高风险操作通过 LangGraph HITL 实现人工确认。
>
> 在工程化方面，我用 pytest 验证 Service、Tool 的业务逻辑和 HTTP 契约，共 353 个用例，其中 20 条是用 `inspect`/`ast` 读源码的架构守卫，用来守住分层方向；同时自行实现 Agent Evaluation Runner，对 Tool 路由、参数、Tool Result、最终回答、HITL 和数据状态做确定性断言，12 条 case / 86 个断言，最后一次全量 86/86 全绿与基线固化都在 2026-09-25，之后无回退。我还做了评测器反向验证，故意破坏实现来确认对应断言真的会红。
>
> 输出侧做了三条通道切片：CLI 流式（`stream_mode="messages"`，逐字上屏加按工具名映射的状态行）、HTTP 同步 JSON、以及同一个端点靠 `Accept` 头协商出来的 SSE 流。人工确认在 HTTP 侧明确拒答（501 / `error` 事件），而不是回一条看起来成功的答复。
>
> 安全边界补了一块会话归属校验：新建会话时签发与 thread_id 绑定的 session_token（存储只留 sha256、比对用 `hmac.compare_digest` 定时比较），续轮必须携带，未知或伪造的 thread 一律 403；签发只允许出现在新建分支，校验失败既不签发也不轮换（防接管）。整条链路 15 个离线用例加一条装配守卫钉住，`session_store=None` 只作为测试 seam 存在。用户级认证、速率限制与凭证持久化仍在 V2 清单上。
>
> 装配层做了 Skill 机制：客服 SOP 以 SKILL.md（frontmatter 元数据 + SOP 正文 + 红线约束 + trigger 路由词）作为可插拔配置，经 AgentMiddleware 注入——`before_model` 按 trigger 词做确定性路由标记 `active_skill`，`wrap_model_call` 把 SOP 并进系统提示并按 `allowed_tools` 白名单收窄模型可见工具。SOP 里零硬编码业务阈值（天数/金额全部以政策检索与工具返回为准），路由用真实 graph + `Command(resume=...)` 探针实测过跨 HITL 恢复的可见性，28 个用例钉住注入层与清除语义。
>
> 项目开发采用 Vertical Slice 和 AI Coding Workflow，每完成一个完整业务能力后再进行测试和真实模型验证，而不是先堆积大量孤立代码。

------

# 21. 面试中最值得讲的几个技术点

如果面试官继续深入，可以重点讲：

### ① 为什么需要 Service 层？

因为：

```text
Tool 是 Agent 接口
Service 是业务逻辑
```

Service 可以被：

```text
Agent
Web API
后台管理系统
```

共同复用。

------

### ② 为什么 RAG 不直接负责回答？

因为：

```text
RAG = Retrieval
```

当前架构让：

```text
Agent LLM
```

统一负责最终回答，避免多个 LLM 层级混杂。

------

### ③ 为什么取消订单要 HITL？

因为取消订单属于具有业务后果的操作。

流程：

```text
检查资格
 ↓
人工确认
 ↓
执行修改
```

------

### ④ 为什么不用 LLM 判断退款资格？

因为退款资格是：

```text
确定性的业务规则
```

应该由：

```text
RefundService
```

判断。

LLM 只负责理解用户意图并触发能力。

------

### ⑤ pytest 与 Agent Evaluation 有什么区别？

一句话：

> **pytest 测代码是否正确，Agent Evaluation 测 Agent 是否做对了事情。**

------

### ⑥ 为什么 SSE 靠 `Accept` 头协商，而不是加字段或加端点？

候选方案：

```text
请求体加 stream=true
新增端点 /api/chat/stream
带 Accept: text/event-stream 才走流式    ← 选了这条
```

选它的原因：

```text
能力仍然只有一个（一次问答）
形态只是响应的编码方式
```

而这里有个必须处理的坑：

```text
curl 与多数客户端默认 Accept: */*
```

`*/*` 是"随便"，不是"要流式"。
判定只认显式的 `text/event-stream`，否则老客户端会被悄悄换成另一种响应形态。

------

### ⑦ 为什么流式出错还是 HTTP 200？

```text
StreamingResponse 开始吐第一个字节时，响应头已经发出去了
```

此时状态码改不了。

所以本项目把"是否成功"编码进事件流本身：

```text
done    → 正常收尾
error   → 终止，且后面不再有 done
```

客户端不能只看 200，必须读到 `done` 才算答完。

------

### ⑧ 为什么 HTTP 上的 HITL 选择拒答？

因为这个端点没有裁决通道，而错误选项有两个：

```text
回一条空 answer        → 客户端以为处理过了
顺着用户意图编一句结果 → 用文案假装办成了
```

本项目选：

```text
明确拒绝，并说明"该操作没有执行、订单数据未被改动"
```

这与第 6.3 节"先判资格再 HITL"是同一个原则的两面：
**不能办的时候，不要装成能办。**

------

### ⑨ 这条 bug 为什么离线测试全绿、真实模型才红？

```text
旧判定：本轮有没有文本增量 → 有就发 done
```

假流式模型是按"我以为协议长什么样"写的：

```text
我以为停在确认位时不会有任何文本增量
```

真实模型会先说一句「好的，我来帮您查询是否可以取消」再调工具，
于是那句客套话把「没办理」盖成了「已答完」。

修复：

```text
收尾无条件读一次线程状态（get_state）
```

讲这条的价值在于说明测试替身的局限：

> 假模型证明链路按设计形状接通，真实 smoke 证明那个形状确实存在。

------

### ⑩ 为什么签发凭证只允许出现在"新建会话"分支？

因为归属校验最常见的做错方式，就是那行"救回来"的代码：

```text
校验失败 → 顺手签发一份新凭证 → 伪造任意 thread_id 都能当场领到合法的
凭证不对 → 轮换 → 攻击者获得"猜一次 → 被发一份新凭据"的免费通道
两类失败分开报 → 这个端点退化成"某个 thread_id 是否存在"的探针
```

所以契约被钉成三条否定式：签发只在新建分支、错误凭证绝不轮换（防接管用例断言原 token 仍然有效）、
未知 thread 与凭证不符共用一条 403。这三条都不靠文档提醒，各自有离线用例兜底。

------

# 22. 项目 V2 可以怎么演进

V1 的能力闭环已经形成，V2.1 的输出通道也已经落地。现在缺的不是新功能，是**边界**：

```text
V1     单轮任务型 Agent（Tool Calling + RAG + Service + HITL + Evaluation）
        ↓
V2     短期 Conversation Memory：同 thread_id 多轮 + 系统提示的指代口径        ← 已落地
        ↓
V2.1   输出通道：CLI 流式 → HTTP JSON → HTTP SSE                              ← 已落地
        ↓
V2.2   把通道补成可用服务
        ├── 认证与 thread_id 归属校验（归属校验已于 2026-10-17 落地，见 10.8；用户级认证仍待）
        ├── HTTP 上的人工确认办理通道（待确认列表 + 裁决端点）
        └── 持久化 checkpointer（重启不再失忆）
        ↓
V2.3   上下文治理（窗口裁剪与摘要、多轮指代评测 case）
        ↓
V3     长期用户画像、跨轮任务、n>1 稳定性评测、LLM-as-a-Judge、
       Prompt Injection 防护与 Tool 权限、真实数据库、Observability
```

**V2.2 排在 V2.3 前面，这个顺序是通道打通之后改的主意。**
在此之前"上下文治理"看着是下一格，因为它最技术、最像正事。但 HTTP 一通，两件事变了性质：

```text
① thread_id 就是访问凭证
   → 从"进程内的不便"变成"任何能访问端口的人都能接着别人的对话说"
   → 2026-10-17 已堵住这一半：续轮不带有效 session_token 只会拿到 403（见 10.8）

② 取消与退款是这套系统唯一真正"办事"的能力
   → 而它们在 HTTP 侧只能被拒答
```

这两条不补，继续做上下文裁剪属于给一栋没有门的楼装空调。

配套的评测升级按同样口径取舍：

```text
先补数据快照，让剩余三条退款结论可达
再让 eval/runner.py 至少有一条经过 HTTP 或经过 stream() 的路径
然后才谈 n>1 与 LLM-as-a-Judge
```

还可以进一步加入：

```text
真实订单 API
真实物流 API
真实支付/退款系统
前端客服界面
```

但这些不应该为了“技术栈看起来多”而一次性加入。

**一项没有机器判定的能力，不该继续往上叠。**

------

# 23. 最终认知总结

整个项目最终可以浓缩成一张图：

```text
                         用户
                          │
                          ▼
                         LLM
                理解意图 / Tool Calling
                          │
                          ▼
                        Agent
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           Tool         Tool         Tool
             │            │            │
             ▼            ▼            ▼
         Service       Service        RAG
             │            │            │
             ▼            ▼            ▼
            Data         Data      Knowledge Base
             │            │            │
             └────────────┼────────────┘
                          ▼
                       Tool Result
                          │
                          ▼
                         LLM
                          │
                  继续调用 / 最终回答
                          │
                          ▼
                         用户
```

这张图只画执行链。
入口层（CLI / HTTP）、展示层（流式事件筛选）、状态层（Checkpointer + `thread_id`）
与它们之间的依赖方向，只在 [docs/架构图.md](docs/架构图.md) 维护。

其中：

```text
LLM
→ 决定做什么

Agent
→ 负责组织和执行

Tool
→ 提供能力入口

Service
→ 执行确定性的业务规则

RAG
→ 提供知识

Data
→ 提供真实业务数据

HITL
→ 在高风险节点引入人工确认

Checkpointer + thread_id
→ 承载多轮上下文与挂起的确认（一份存储、两个用途）

展示层
→ 决定哪些事件给人看、配哪句状态文案（不由模型决定）

HTTP 入口
→ 同一份答复的另一种形态；办理类动作在这里明确拒答

pytest
→ 验证代码与业务逻辑

源码级架构守卫
→ 验证分层方向没有被写反

Agent Evaluation
→ 验证 Agent 行为

Manual Test
→ 验证真实使用体验

Real-model Smoke Test
→ 验证"我对协议的想象"确实成立
```

最终形成一个完整的 LLM 应用工程闭环：

```text
需求
 ↓
架构设计
 ↓
Vertical Slice
 ↓
AI Coding
 ↓
Unit / Integration Test
 ↓
Agent Evaluation
 ↓
Manual Test
 ↓
Real-model Smoke Test
 ↓
工程验收
```

这也是这个项目相比单纯“写一个 LangChain Agent Demo”更有价值的地方。

最后一组切片尤其能说明这件事：
09-26 晚到 09-29 那四天，业务规则一行业没改，只加了输出通道，
而新增的 102 个用例全在测通道自身——哪些事件能给人看、状态文案由谁决定、
没办成的时候怎么明确说“没办成”。

> 项目真正练的不是"接上一个 LLM"，而是**每加一块能力，就同时把它的边界钉成可验证的东西**。