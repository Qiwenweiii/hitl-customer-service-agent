#!/usr/bin/env python
"""RAG 消融对比：Baseline（纯向量）vs Treatment（向量+重排，无阈值）vs Online-like（线上链路）。

用法：uv run --no-sync python eval/rag_ablation.py

只读评测：复用生产 RAGService，不改动 app/rag/ 任何代码，不写任何持久化状态，可重复运行。
参数全部来自配置：RETRIEVAL_K / CONTEXT_TOP_K 来自 app.config，MIN_RERANK_SCORE 来自 app.rag.rag_service。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CONTEXT_TOP_K, RETRIEVAL_K  # noqa: E402
from app.rag.rag_service import MIN_RERANK_SCORE, RAGService  # noqa: E402

DATASET_PATH = ROOT / "eval" / "rag_ablation_queries.json"
RESULT_PATH = ROOT / "eval" / "rag_ablation_result.md"

# 报告展示规则：样本数低于该值的分组结论仅作方向参考（与检索参数无关，是统计口径）
SMALL_SAMPLE_MIN = 5

GROUP_LABELS = {
    "baseline": "Baseline（纯向量）",
    "treatment": "Treatment（向量+重排，无阈值）",
    "online_like": "Online-like（线上链路）",
}


def is_hit(doc, expected_doc: str, expected_index: int) -> bool:
    metadata = doc.metadata
    return (
        metadata.get("document_id") == expected_doc
        and int(metadata.get("chunk_index", -1)) == expected_index
    )


def rank_of(docs, expected_doc: str, expected_index: int) -> int | None:
    """期望片段在有序列表中的 1-based 排名；未出现返回 None。"""
    for position, doc in enumerate(docs, start=1):
        if is_hit(doc, expected_doc, expected_index):
            return position
    return None


def evaluate_query(service: RAGService, case: dict) -> dict:
    query = case["query"]
    expected_doc = case["expected_document_id"]
    expected_index = case["expected_chunk_index"]

    # --- Baseline：纯向量。不依赖 retrieve() 的隐式降序，
    # 用 similarity_search_with_score 拿到距离后显式升序排序（Chroma 距离越小越相似），保证可复现。
    scored = service.vector_store.similarity_search_with_score(query, k=RETRIEVAL_K)
    base_ranked = [doc for doc, _ in sorted(scored, key=lambda pair: pair[1])]
    base_rank = rank_of(base_ranked[:CONTEXT_TOP_K], expected_doc, expected_index)
    base_full_rank = rank_of(base_ranked, expected_doc, expected_index)

    # --- Treatment：向量召回 + CrossEncoder 全排序，取前 CONTEXT_TOP_K，不应用任何阈值。
    # 主对比组，回答"重排本身有没有用"。
    candidates = service.retrieve(query)
    reranked = service.rerank(query, candidates)
    treat_docs = [doc for _, doc in reranked]
    treat_rank = rank_of(treat_docs[:CONTEXT_TOP_K], expected_doc, expected_index)
    treat_full_rank = rank_of(treat_docs, expected_doc, expected_index)
    expected_scores = [score for score, doc in reranked if is_hit(doc, expected_doc, expected_index)]
    expected_rerank_score = expected_scores[0] if expected_scores else None
    top1_rerank_score = reranked[0][0] if reranked else None

    # --- Online-like：复刻线上链路（过滤 score >= MIN_RERANK_SCORE 再截断前 K），
    # 只用于观察阈值过滤损失了多少召回，不参与主对比。
    online_docs = [doc for score, doc in reranked if score >= MIN_RERANK_SCORE]
    online_rank = rank_of(online_docs[:CONTEXT_TOP_K], expected_doc, expected_index)
    online_lost_by_threshold = expected_rerank_score is not None and expected_rerank_score < MIN_RERANK_SCORE

    return {
        "id": case["id"],
        "query": query,
        "query_type": case["query_type"],
        "expected": f"{expected_doc}#{expected_index}",
        "ranks": {"baseline": base_rank, "treatment": treat_rank, "online_like": online_rank},
        "diagnostics": {
            "base_full_rank": base_full_rank,
            "treat_full_rank": treat_full_rank,
            "expected_rerank_score": expected_rerank_score,
            "top1_rerank_score": top1_rerank_score,
            "online_lost_by_threshold": online_lost_by_threshold,
        },
    }


def metrics(rows, group: str) -> tuple[float, float]:
    """返回 (HitRate@K, MRR)；排名在结果列表内才计分，未命中记 0。"""
    reciprocals = [
        1.0 / row["ranks"][group] if row["ranks"][group] is not None else 0.0 for row in rows
    ]
    hit_rate = sum(1 for r in reciprocals if r > 0) / len(rows)
    return hit_rate, sum(reciprocals) / len(rows)


def fmt_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "miss"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_summary_table(rows) -> list[str]:
    base_hit, base_mrr = metrics(rows, "baseline")
    lines = [
        f"| 配置 | Hit Rate@{CONTEXT_TOP_K} | MRR | ΔHit vs Baseline | ΔMRR vs Baseline |",
        "|---|---|---|---|---|",
    ]
    for group in GROUP_LABELS:
        hit, mrr = metrics(rows, group)
        delta_hit, delta_mrr = hit - base_hit, mrr - base_mrr
        sign = "+" if delta_hit >= 0 else ""
        sign_m = "+" if delta_mrr >= 0 else ""
        lines.append(
            f"| {GROUP_LABELS[group]} | {fmt_pct(hit)} | {mrr:.3f} "
            f"| {sign}{fmt_pct(delta_hit)} | {sign_m}{delta_mrr:.3f} |"
        )
    return lines


def build_type_tables(rows) -> list[str]:
    lines = [
        f"| query_type | 样本数 | Baseline Hit@{CONTEXT_TOP_K} | Treatment Hit@{CONTEXT_TOP_K} "
        f"| Online Hit@{CONTEXT_TOP_K} | Baseline MRR | Treatment MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for qtype in ("direct", "colloquial", "cross_topic", "negation_condition"):
        subset = [r for r in rows if r["query_type"] == qtype]
        if not subset:
            continue
        b_hit, b_mrr = metrics(subset, "baseline")
        t_hit, t_mrr = metrics(subset, "treatment")
        o_hit, _ = metrics(subset, "online_like")
        note = "（样本量小，结论仅作方向参考）" if len(subset) < SMALL_SAMPLE_MIN else ""
        lines.append(
            f"| {qtype}{note} | {len(subset)} | {fmt_pct(b_hit)} | {fmt_pct(t_hit)} | {fmt_pct(o_hit)} "
            f"| {b_mrr:.3f} | {t_mrr:.3f} |"
        )
    return lines


def build_detail_table(rows) -> list[str]:
    lines = [
        f"| id | query | 类型 | 期望片段 | Baseline 排名 | Treatment 排名 | Online 排名 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        r = row["ranks"]
        lines.append(
            f"| {row['id']} | {row['query']} | {row['query_type']} | {row['expected']} "
            f"| {fmt_rank(r['baseline'])} | {fmt_rank(r['treatment'])} | {fmt_rank(r['online_like'])} |"
        )
    return lines


def build_all_miss_section(rows) -> list[str]:
    missed = [r for r in rows if all(v is None for v in r["ranks"].values())]
    if not missed:
        return ["（无）"]
    lines = []
    for row in missed:
        diag = row["diagnostics"]
        cause = (
            f"向量召回前 {RETRIEVAL_K} 内未见期望片段"
            if diag["base_full_rank"] is None
            else f"召回在第 {diag['base_full_rank']} 位、重排后被挤到 {diag['treat_full_rank']} 位"
        )
        hint = (
            "跨文档 query：答案可能落在另一文档的 chunk，单一期望片段判定口径偏严"
            if row["query_type"] == "cross_topic"
            else "可能是测试集标注问题"
        )
        lines.append(f"- id={row['id']}「{row['query']}」（期望 {row['expected']}）：{cause}；{hint}")
    return lines


def build_regression_section(rows) -> list[str]:
    """重排负收益案例：Baseline 进前 K 但 Treatment 掉出前 K。"""
    cases = [
        r
        for r in rows
        if r["ranks"]["baseline"] is not None and r["ranks"]["treatment"] is None
    ]
    if not cases:
        return ["（无：重排没有把任何 Baseline 命中的期望片段挤出前 " + str(CONTEXT_TOP_K) + "）"]
    lines = []
    for row in cases:
        diag = row["diagnostics"]
        exp_score = diag["expected_rerank_score"]
        top1_score = diag["top1_rerank_score"]
        if diag["treat_full_rank"] is None:
            analysis = f"期望片段不在向量召回候选里（Baseline 全列表第 {diag['base_full_rank']} 位），两组候选集不一致，需核对 retrieve 与 similarity_search 路径"
        elif diag["treat_full_rank"] > CONTEXT_TOP_K:
            analysis = (
                f"重排把它从第 {diag['base_full_rank']} 位压到第 {diag['treat_full_rank']} 位，挤出前 {CONTEXT_TOP_K}；"
                f"期望片段 rerank 分 {exp_score:.4f}，Top1 rerank 分 {top1_score:.4f}，"
                f"分差 {top1_score - exp_score:.4f}"
            )
        else:
            analysis = "排名在阈值内但被 Online 截断逻辑影响（异常，需检查）"
        lines.append(f"- id={row['id']}「{row['query']}」（期望 {row['expected']}，Baseline 第 {row['ranks']['baseline']} 位）：{analysis}")
    return lines


def build_service() -> RAGService:
    """构造评测用 RAGService。

    当前网络不可达 huggingface.co，且该版 transformers 对仓库 id 形式的
    CrossEncoder 仍会发起 model_info 网络探测；把 Reranker 解析为本地缓存
    snapshot 路径后经公开构造参数注入，走离线缓存。Embedding 由
    HF_HUB_OFFLINE 环境变量下的惰性属性正常加载。不改 app/rag 任何代码。
    """
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder

    from app.config import RERANKER_MODEL

    local_path = snapshot_download(RERANKER_MODEL, local_files_only=True)
    return RAGService(reranker=CrossEncoder(model_name_or_path=local_path))


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["queries"]

    service = build_service()
    stored = len(service.vector_store.get().get("ids") or [])
    if stored == 0:
        raise SystemExit("Chroma collection 为空，请先运行 scripts/build_knowledge_base.py")

    print(f"知识库 {stored} 个 chunk；检索参数 RETRIEVAL_K={RETRIEVAL_K}, "
          f"CONTEXT_TOP_K={CONTEXT_TOP_K}, MIN_RERANK_SCORE={MIN_RERANK_SCORE}")
    print("注意：Baseline 的分数是 Chroma 向量距离（越小越相似），Treatment/Online 是 CrossEncoder 相关度分（越大越相关），量纲不同不可直接比较。\n")

    rows = [evaluate_query(service, case) for case in cases]

    summary = build_summary_table(rows)
    by_type = build_type_tables(rows)
    detail = build_detail_table(rows)
    all_miss = build_all_miss_section(rows)
    regressions = build_regression_section(rows)

    print("## 总体汇总（样本 {}）".format(len(rows)))
    print("\n".join(summary))
    print()
    print(f"## 按 query_type 分组（样本数 < {SMALL_SAMPLE_MIN} 的组结论仅作方向参考）")
    print("\n".join(by_type))
    print()

    online_hit, _ = metrics(rows, "online_like")
    treat_hit, _ = metrics(rows, "treatment")
    lost = sum(1 for r in rows if r["diagnostics"]["online_lost_by_threshold"])
    print(f"Online-like 相对 Treatment：HitRate 差 {fmt_pct(online_hit - treat_hit)}，"
          f"因阈值过滤直接损失期望片段的 query 共 {lost} 条。")

    md = [
        "# RAG 消融结果",
        "",
        f"- 数据集：eval/rag_ablation_queries.json（{len(rows)} 条）",
        f"- 参数：RETRIEVAL_K={RETRIEVAL_K}，CONTEXT_TOP_K={CONTEXT_TOP_K}，MIN_RERANK_SCORE={MIN_RERANK_SCORE}",
        f"- 模型：Embedding=BAAI/bge-small-zh-v1.5，Reranker=BAAI/bge-reranker-base",
        "- 判定：结果条目的 metadata (document_id, chunk_index) 与期望完全一致记为命中；排名为 1-based，未进前 K 记 miss",
        "- 量纲说明：向量距离越小越相似，CrossEncoder 分数越大越相关，两者不可直接比较",
        "",
        "## 总体汇总",
        *summary,
        "",
        "## 按 query_type 分组",
        *by_type,
        "",
        "## 每条 query 明细",
        *detail,
        "",
        "## 三组都未命中（标注或判定口径疑点）",
        *all_miss,
        "",
        "## Baseline 命中但 Treatment 未命中（重排负收益）",
        *regressions,
        "",
    ]
    RESULT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\n结果报告已写入：{RESULT_PATH}")


if __name__ == "__main__":
    main()
