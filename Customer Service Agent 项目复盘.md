# Customer Service Agent 项目最终复盘

> 项目定位：一个面向电商客服场景的综合型 LLM Agent 项目
> 核心技术：Python + LangChain + LangGraph + RAG + Tool Calling + HITL + Agent Evaluation
> 开发方式：Vertical Slice + AI Coding + pytest + Agent Evaluation

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
拿到别人的 thread_id → 就能接着那段对话说
多轮指代 → 没有评测 case，pytest 只证明上下文送达与线程隔离
```

------

# 9. pytest 与 Agent Evaluation

这是项目工程化非常重要的一部分。

两者不是重复的。

------

## 9.1 pytest 测什么

pytest 主要测试：

```text
代码
业务规则
Service
Tool
数据处理
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

## 9.2 Agent Evaluation 测什么

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

## 9.3 两者关系

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

# 10. Agent Evaluation 实现

当前项目没有直接依赖 LangSmith Evaluation，而是自己实现了最小 Evaluation Runner。

------

## 10.1 Evaluation 核心流程

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

## 10.2 Dataset

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

## 10.3 Runner

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

## 10.4 min_runner 的定位

项目中的：

```text
eval/min_runner.py
eval/min_hitl_runner.py
```

只是：

> 用于理解 Evaluation / HITL 最小原理的教学 Demo。

不是正式 Evaluation 入口。

------

## 10.5 当前 Evaluation Case

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

------

## 10.6 为什么没有 LLM-as-Judge

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

## 10.7 当前 Evaluation 的限制

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

# 11. Evaluation 的反向验证

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

# 12. 为什么 AI Coding 仍然需要人工

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

## 12.1 AI 适合做什么

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

## 12.2 人必须负责什么

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

## 12.3 AI 生成测试，人应该看什么

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

# 13. Vertical Slice 开发方式

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

## 13.1 query_order

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

## 13.2 query_logistics

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

## 13.3 search_knowledge

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

## 13.4 cancel_order

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

## 13.5 refund_order

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

## 14. 测试策略

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

# 15. 最终工程验收

当前项目最终状态：

```text
pytest：
195 passed
```

Agent Evaluation：

```text
12 / 12 PASS
86 / 86 assertions PASS
```

Ruff：

```text
clean
```

真实 Mock 数据：

```text
orders.json
未发生非预期变化
```

应用：

```bash
python -m app.main
```

正常退出。

HITL：

```bash
python scripts/hitl_cli.py
```

可以正常运行。

正式 Evaluation：

```bash
uv run python -m eval.runner
```

正常运行。

------

## 15.1 最终项目验证矩阵

| 验证方式         | 目的                 |
| ---------------- | -------------------- |
| pytest           | 代码与业务逻辑       |
| Agent Evaluation | Agent 行为           |
| Manual Test      | 真实交互             |
| Smoke Test       | 真实 LLM 集成        |
| Ruff             | 代码质量             |
| 数据完整性检查   | 防止测试污染真实数据 |

------

# 16. 当前项目已知限制

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

------

### Memory

短期记忆只做到"同线程累积"，四条边界都还在：

```text
进程退出即失忆（无跨进程持久化）
无窗口裁剪与摘要（长会话会撞上下文上限）
thread_id 没有归属与权限校验
多轮指代没有评测 case
```

长期会话记忆（跨会话的用户画像与偏好）暂未加入。

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

------

### RAG

当前知识库规模较小。

------

### HITL

使用：

```text
InMemorySaver
```

因此应用重启后不会保留挂起状态。

这些都属于后续 V2/V3 的演进方向。

------

# 17. 核心工程原则总结

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

# 18. 面试版项目描述

可以将项目浓缩成：

> 我独立设计并实现了一个基于 LangGraph 的电商客服 Agent，支持订单查询、物流查询、企业知识库问答、订单取消和退款等场景。系统采用 Tool + Service 分层，LLM 负责意图理解和 Tool Calling，Service 负责确定性的业务规则，RAG 用于企业知识检索，高风险操作通过 LangGraph HITL 实现人工确认。
>
> 在工程化方面，我使用 pytest 验证 Service 和 Tool 的业务逻辑，并自行实现 Agent Evaluation Runner，对 Tool 路由、参数、Tool Result、最终回答、HITL 和数据状态进行确定性评估。目前 12 个核心 Evaluation Case 全部通过，共 86 个断言通过，同时完成了 Evaluation 反向验证，确保错误实现能够被测试发现。
>
> 项目开发采用 Vertical Slice 和 AI Coding Workflow，每完成一个完整业务能力后再进行测试和真实模型验证，而不是先堆积大量孤立代码。

------

# 19. 面试中最值得讲的几个技术点

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

# 20. 项目 V2 可以怎么演进

当前 V1 已经形成完整闭环。

未来可以逐步增加：

```text
V2
├── Conversation Memory（短期部分已落地：同 thread_id 多轮 + 指代口径）
├── 上下文治理（跨进程持久化 / 窗口裁剪 / 会话归属校验）
├── PostgreSQL
├── Redis
├── Web API
├── 前端客服界面
└── 更完整 Evaluation

V3
├── Prompt Injection 防护
├── 权限控制
├── Tool 权限
├── 更复杂 Agent Workflow
├── Streaming
├── Observability
└── Production Deployment
```

还可以进一步加入：

```text
真实订单 API
真实物流 API
真实支付/退款系统
```

但这些不应该为了“技术栈看起来多”而一次性加入。

------

# 21. 最终认知总结

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

pytest
→ 验证代码与业务逻辑

Agent Evaluation
→ 验证 Agent 行为

Manual Test
→ 验证真实使用体验
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