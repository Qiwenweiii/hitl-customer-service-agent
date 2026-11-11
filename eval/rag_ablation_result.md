# RAG 消融结果

- 数据集：eval/rag_ablation_queries.json（23 条）
- 参数：RETRIEVAL_K=10，CONTEXT_TOP_K=3，MIN_RERANK_SCORE=0.1
- 模型：Embedding=BAAI/bge-small-zh-v1.5，Reranker=BAAI/bge-reranker-base
- 判定：结果条目的 metadata (document_id, chunk_index) 与期望完全一致记为命中；排名为 1-based，未进前 K 记 miss
- 量纲说明：向量距离越小越相似，CrossEncoder 分数越大越相关，两者不可直接比较

## 总体汇总
| 配置 | Hit Rate@3 | MRR | ΔHit vs Baseline | ΔMRR vs Baseline |
|---|---|---|---|---|
| Baseline（纯向量） | 91.3% | 0.870 | +0.0% | +0.000 |
| Treatment（向量+重排，无阈值） | 87.0% | 0.826 | -4.3% | -0.043 |
| Online-like（线上链路） | 87.0% | 0.826 | -4.3% | -0.043 |

## 按 query_type 分组
| query_type | 样本数 | Baseline Hit@3 | Treatment Hit@3 | Online Hit@3 | Baseline MRR | Treatment MRR |
|---|---|---|---|---|---|---|
| direct | 8 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| colloquial | 7 | 71.4% | 57.1% | 57.1% | 0.571 | 0.500 |
| cross_topic（样本量小，结论仅作方向参考） | 4 | 100.0% | 100.0% | 100.0% | 1.000 | 0.875 |
| negation_condition（样本量小，结论仅作方向参考） | 4 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |

## 每条 query 明细
| id | query | 类型 | 期望片段 | Baseline 排名 | Treatment 排名 | Online 排名 |
|---|---|---|---|---|---|---|
| 1 | 7天无理由退货的条件是什么？ | direct | refund_policy.md#2 | 1 | 1 | 1 |
| 2 | 退款到账时间一般是多久？ | direct | refund_policy.md#4 | 1 | 1 | 1 |
| 3 | 冰箱、洗衣机这类大家电的保修期是多久？ | direct | after_sales_policy.md#1 | 1 | 1 | 1 |
| 4 | 申请售后之后从审核到维修完成整个流程有时限要求吗？ | direct | after_sales_policy.md#3 | 1 | 1 | 1 |
| 5 | 金卡会员需要近12个月累计实付满多少才能升级？ | direct | membership_policy.md#1 | 1 | 1 | 1 |
| 6 | 积分抵扣的比例是多少？单笔订单最多能抵扣几成？ | direct | membership_policy.md#4 | 1 | 1 | 1 |
| 7 | 运费怎么收？多少金额才包邮？ | direct | shipping_policy.md#2 | 1 | 1 | 1 |
| 8 | 普通现货商品付款后多久发货？ | direct | shipping_policy.md#1 | 1 | 1 | 1 |
| 9 | 我买错了能退吗 | colloquial | refund_policy.md#2 | miss | miss | miss |
| 10 | 东西坏了咋办 | colloquial | after_sales_policy.md#2 | miss | miss | miss |
| 11 | 快递怎么一直不动啊 | colloquial | shipping_policy.md#4 | 2 | 2 | 2 |
| 12 | 积分怎么攒 | colloquial | membership_policy.md#3 | 1 | 1 | 1 |
| 13 | 会员等级什么时候重新评定？达不到要求会掉级吗 | colloquial | membership_policy.md#2 | 1 | 1 | 1 |
| 14 | 明天就想收到，加急送要加钱吗 | colloquial | shipping_policy.md#2 | 1 | 1 | 1 |
| 15 | 申请售后要准备点啥 | colloquial | after_sales_policy.md#1 | 2 | miss | miss |
| 16 | 质量问题退货的取件费，和普通退货的运费是一个说法吗？ | cross_topic | refund_policy.md#3 | 1 | 1 | 1 |
| 17 | 用积分抵过的订单退款后，积分会退回来吗？会不会过期？ | cross_topic | membership_policy.md#4 | 1 | 1 | 1 |
| 18 | 订单已经发货了，拒收和在途拦截是一回事吗？要收费不？ | cross_topic | shipping_policy.md#5 | 1 | 2 | 2 |
| 19 | 铂金会员的退货免运费，7天无理由退货也能用吗？ | cross_topic | membership_policy.md#5 | 1 | 1 | 1 |
| 20 | 拆封了还能七天无理由退货吗？ | negation_condition | refund_policy.md#2 | 1 | 1 | 1 |
| 21 | 自己不小心进水弄坏了，还在保修期内，能免费修吗？ | negation_condition | after_sales_policy.md#2 | 1 | 1 | 1 |
| 22 | 定制商品支持7天无理由退货吗？ | negation_condition | refund_policy.md#2 | 1 | 1 | 1 |
| 23 | 积分过期了还能用来抵扣吗 | negation_condition | membership_policy.md#4 | 1 | 1 | 1 |

## 三组都未命中（标注或判定口径疑点）
- id=9「我买错了能退吗」（期望 refund_policy.md#2）：召回在第 6 位、重排后被挤到 5 位；可能是测试集标注问题
- id=10「东西坏了咋办」（期望 after_sales_policy.md#2）：召回在第 7 位、重排后被挤到 4 位；可能是测试集标注问题

## Baseline 命中但 Treatment 未命中（重排负收益）
- id=15「申请售后要准备点啥」（期望 after_sales_policy.md#1，Baseline 第 2 位）：重排把它从第 2 位压到第 8 位，挤出前 3；期望片段 rerank 分 0.0640，Top1 rerank 分 0.6977，分差 0.6336
