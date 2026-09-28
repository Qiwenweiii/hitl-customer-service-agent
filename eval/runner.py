"""最小 Agent 评测 runner：dataset → 真实执行 → 采集证据 → 确定性断言 → 报表。

只测「模型 + 系统提示 + 工具契约」的组合行为；Service 层业务规则由 tests/ 的 pytest 切片负责，这里不重复。

用法：
  uv run python -m eval.runner                       # 跑全部 case
  uv run python -m eval.runner --case E01,E05        # 只跑指定 case
  uv run python -m eval.runner --update-baseline     # 把本次结果接受为基线
  EVAL_TRACING=true uv run python -m eval.runner     # 需要看 LangSmith 链路时再开 tracing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# 必须在导入 app.* 之前落定：load_dotenv 不会覆盖已存在的环境变量，
# 而 tracing 打开时终端会被 LangSmith 的上报日志盖掉，读不到 case 输出。
os.environ["LANGSMITH_TRACING"] = "true" if os.getenv("EVAL_TRACING") == "true" else "false"

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import data_loader
from app.agent.graph import create_customer_service_agent
from app.config import LLM_MODEL
from app.services.refund_service import RefundService
from app.tools import refund_tools

EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "dataset"
BASELINE_FILE = EVAL_DIR / "baseline.json"
RESULTS_DIR = EVAL_DIR / "results"
REAL_ORDERS_FILE = data_loader.ORDERS_FILE
REAL_ORDERS_TEXT = REAL_ORDERS_FILE.read_text(encoding="utf-8")

# outcome → 工具文案里的特征短语。评测只能看到工具输出的字符串，因此用它做代理判定；
# 表里没有的 outcome 会直接报错，避免「改了文案 → 断言静默失效」。
OUTCOME_PHRASES: dict[str, tuple[str, ...]] = {
    "CANCELLED": ("已取消", "状态已更新"),
    "CANCELLED_ALREADY": ("此前已经取消",),
    "NOT_ALLOWED": ("不支持直接取消",),
    "NOT_FOUND": ("未找到订单号为",),
    "NOT_PAID": ("尚未付款",),
    "ALREADY_REFUNDING": ("正在处理中", "无需重复申请"),
    "ALREADY_REFUNDED": ("款项此前已退回",),
    "REFUND_EXPIRED": ("已超过该单的退款时效",),
    "NEED_REJECT_DELIVERY": ("应先拒收包裹",),
    "NO_DELIVERY_TIME": ("缺少签收时间",),
    "REFUNDING_SUBMITTED": ("退款申请已受理",),
}

TOP_KEYS = {"id", "slice", "severity", "user", "now", "hitl", "expect"}
EXPECT_KEYS = {
    "tools",
    "not_tools",
    "order_id",
    "interrupt",
    "outcome",
    "data",
    "answer_include",
    "answer_include_any",
    "answer_exclude",
}


@dataclass
class Assertion:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseRun:
    id: str
    slice: str
    severity: str
    user: str
    answer: str = ""
    error: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)
    interrupts: list[dict] = field(default_factory=list)
    data_before: dict = field(default_factory=dict)
    data_after: dict = field(default_factory=dict)
    llm_calls: int = 0
    elapsed: float = 0.0
    assertions: list[Assertion] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # 必须有断言且全过：一条断言都没有的 case 不能算通过。
        return not self.error and bool(self.assertions) and all(a.passed for a in self.assertions)


def load_cases(only: set[str] | None = None) -> list[dict]:
    """读取 dataset 下所有 YAML 并做字段校验：写错的 key 必须当场报错，不能变成静默通过。"""
    cases: list[dict] = []
    for path in sorted(DATASET_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for case in raw.get("cases", []):
            unknown = set(case) - TOP_KEYS
            if unknown:
                raise ValueError(f"{path.name} case {case.get('id')}: 未知字段 {sorted(unknown)}")
            if not {"id", "user", "expect"} <= set(case):
                raise ValueError(f"{path.name}: case 缺少 id/user/expect")
            expect = case["expect"]
            unknown = set(expect) - EXPECT_KEYS
            if unknown:
                raise ValueError(f"case {case['id']}: expect 未知字段 {sorted(unknown)}")
            if not expect:
                raise ValueError(f"case {case['id']}: expect 为空，等于零断言空判")
            for oid, fields in expect.get("data", {}).items():
                for fld, pair in fields.items():
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError(
                            f"case {case['id']}: data.{oid}.{fld} 必须是 [跑前, 跑后] 两元素列表"
                        )
            case.setdefault("slice", path.stem)
            case.setdefault("severity", "blocking")
            cases.append(case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"case id 重复: {ids}")
    if only:
        missing = only - set(ids)
        if missing:
            raise ValueError(f"dataset 里没有这些 case: {sorted(missing)}")
        cases = [c for c in cases if c["id"] in only]
    return cases


def _snapshot(path: Path) -> dict[str, dict[str, str]]:
    orders = json.loads(path.read_text(encoding="utf-8"))
    return {
        o["order_id"]: {"status": o["status"], "payment_status": o["payment_status"]}
        for o in orders
    }


@contextmanager
def isolated_data(case: dict):
    """把订单数据改道到只读真实文件得到的副本：人工批准产生的写入不落真实数据。"""
    tmp = Path(tempfile.mkdtemp(prefix=f"eval-{case['id']}-"))
    copy = tmp / "orders.json"
    shutil.copyfile(REAL_ORDERS_FILE, copy)

    original_file = data_loader.ORDERS_FILE
    original_refund_service = refund_tools._refund_service
    data_loader.ORDERS_FILE = copy
    if case.get("now"):
        # 时效类 case 必须冻结时钟，否则同一条 case 会随日历从「期内」漂移成「超期」。
        moment = datetime.fromisoformat(case["now"])
        refund_tools._refund_service = RefundService(now=lambda: moment)
    try:
        yield copy
    finally:
        data_loader.ORDERS_FILE = original_file
        refund_tools._refund_service = original_refund_service
        shutil.rmtree(tmp, ignore_errors=True)


def run_case(case: dict, agent) -> CaseRun:
    run = CaseRun(
        id=case["id"],
        slice=case["slice"],
        severity=case["severity"],
        user=case["user"],
    )
    started = time.time()
    config = {"configurable": {"thread_id": case["id"]}}
    with isolated_data(case) as copy:
        run.data_before = _snapshot(copy)
        try:
            result = agent.invoke({"messages": [HumanMessage(case["user"])]}, config)
            decision = (case.get("hitl") or {}).get("decision")
            resumes = 0
            while result.get("__interrupt__"):
                run.interrupts.append(dict(result["__interrupt__"][0].value))
                if resumes >= 1:
                    # 裁决已送达却再次中断：LangGraph 只在 resume 值为非空 dict 时才会真正恢复
                    # （README「人工确认（HITL）」一节记过这个坑），反复中断必须判失败。
                    run.error = "中断未被裁决，图仍在待确认状态"
                    break
                if decision is None or not isinstance(decision, dict) or not decision:
                    run.error = f"hitl.decision 必须是非空 dict，实际 {decision!r}"
                    break
                result = agent.invoke(Command(resume=decision), config)
                resumes += 1

            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    run.llm_calls += 1
                    run.tool_calls.extend(
                        {"name": c["name"], "args": c["args"]} for c in message.tool_calls
                    )
                elif isinstance(message, ToolMessage):
                    run.tool_outputs.append({"name": message.name, "text": str(message.content)})
            run.answer = str(result["messages"][-1].content)
        except Exception as exc:  # noqa: BLE001 - 任一环节抛异常都要落到 case 报表里，不能中断整轮评测
            run.error = run.error or f"{type(exc).__name__}: {exc}"
        run.data_after = _snapshot(copy)
    run.elapsed = time.time() - started
    return run


def evaluate(case: dict, run: CaseRun) -> None:
    expect = case["expect"]
    names = [c["name"] for c in run.tool_calls]
    blob = " ".join(o["text"] for o in run.tool_outputs)

    missing = [t for t in expect.get("tools", []) if t not in names]
    run.assertions.append(
        Assertion("tools", not missing, f"未调用 {missing}，实际调用链 {names}" if missing else "")
    )
    hit_forbidden = [t for t in expect.get("not_tools", []) if t in names]
    run.assertions.append(
        Assertion("not_tools", not hit_forbidden, f"调用了禁调工具 {hit_forbidden}" if hit_forbidden else "")
    )

    if order_id := expect.get("order_id"):
        wrong = [c for c in run.tool_calls if "order_id" in c["args"] and c["args"]["order_id"] != order_id]
        run.assertions.append(
            Assertion(
                "order_id",
                not wrong,
                "; ".join(f"{c['name']} 传了 {c['args']['order_id']}" for c in wrong),
            )
        )

    if spec := expect.get("interrupt"):
        problems = []
        if len(run.interrupts) != 1:
            problems.append(f"中断次数 {len(run.interrupts)}，期望 1 次")
        else:
            payload = run.interrupts[0]
            problems.extend(
                f"{key}: 期望 {want!r}，实际 {payload.get(key)!r}"
                for key, want in spec.items()
                if payload.get(key) != want
            )
        run.assertions.append(Assertion("interrupt", not problems, "; ".join(problems)))
    else:
        run.assertions.append(
            Assertion("interrupt", not run.interrupts, "不该出现 HITL 中断，却出现了")
        )

    if outcome := expect.get("outcome"):
        phrases = OUTCOME_PHRASES.get(outcome)
        if phrases is None:
            raise KeyError(f"OUTCOME_PHRASES 里没有 {outcome} 的特征短语映射，请补充")
        absent = [p for p in phrases if p not in blob]
        run.assertions.append(
            Assertion(
                "outcome",
                not absent,
                f"期望 {outcome}，工具输出缺少特征短语 {absent}；实际: {blob[:200]}" if absent else "",
            )
        )

    if data := expect.get("data"):
        problems = []
        for oid, fields in data.items():
            for fld, (want_before, want_after) in fields.items():
                got_before = run.data_before.get(oid, {}).get(fld)
                got_after = run.data_after.get(oid, {}).get(fld)
                if got_before != want_before:
                    problems.append(f"{oid}.{fld} 跑前应为 {want_before}，实际 {got_before}")
                if got_after != want_after:
                    problems.append(f"{oid}.{fld} 跑后应为 {want_after}，实际 {got_after}")
        run.assertions.append(Assertion("data", not problems, "; ".join(problems)))

    answer_checks = (
        ("answer_include", lambda items: [t for t in items if t not in run.answer], "回答缺少"),
        (
            "answer_include_any",
            lambda items: [] if any(t in run.answer for t in items) else items,
            "回答未出现其中任意一项",
        ),
        ("answer_exclude", lambda items: [t for t in items if t in run.answer], "回答出现禁词"),
    )
    for key, find_offenders, reason in answer_checks:
        items = expect.get(key, [])
        if not items:
            continue
        offenders = find_offenders(items)
        detail = f"{reason} {offenders}；回答: {run.answer[:200]}" if offenders else ""
        run.assertions.append(Assertion(key, not offenders, detail))


def load_baseline() -> dict | None:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return None


def write_baseline(runs: list[CaseRun]) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL,
        "cases": {
            r.id: {
                "status": "pass" if r.passed else "fail",
                "passed": sum(a.passed for a in r.assertions),
                "total": len(r.assertions),
                "failed": [a.name for a in r.assertions if not a.passed],
            }
            for r in runs
        },
    }
    BASELINE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n已写入基线: {BASELINE_FILE.relative_to(EVAL_DIR.parent)}（{len(runs)} 条 case）")


def report(runs: list[CaseRun], baseline: dict | None, update: bool) -> int:
    print(f"\n{'=' * 72}")
    for run in runs:
        status = "PASS" if run.passed else "FAIL"
        print(
            f"{run.id:5} {run.slice:9} {status}  "
            f"{sum(a.passed for a in run.assertions)}/{len(run.assertions)} 断言  "
            f"llm_calls={run.llm_calls}  {run.elapsed:.1f}s"
        )
        if run.error:
            print(f"      ! {run.error}")
        for assertion in run.assertions:
            if not assertion.passed:
                print(f"      x {assertion.name}: {assertion.detail}")

    blocking = [r for r in runs if r.severity == "blocking"]
    failed = [r for r in blocking if not r.passed]
    groups = []
    for severity in ("blocking", "warn"):
        group = [r for r in runs if r.severity == severity]
        if group:
            groups.append(f"{severity} {sum(r.passed for r in group)}/{len(group)}")
    print(f"\n汇总: {' | '.join(groups)}")

    regressions: list[str] = []
    if baseline:
        previous = baseline.get("cases", {})
        regressions = [
            r.id for r in runs if previous.get(r.id, {}).get("status") == "pass" and not r.passed
        ]
        if regressions:
            print(f"相比基线回退: {regressions}")
        else:
            print("相比基线: 无回退")
    else:
        print("（还没有基线，用 --update-baseline 接受本次结果）")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"run-{stamp}.json"
    out.write_text(
        json.dumps([asdict(r) for r in runs], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"证据已落盘: {out.relative_to(EVAL_DIR.parent)}")

    if update:
        write_baseline(runs)
    return 1 if failed or regressions else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="最小 Agent 评测 runner")
    parser.add_argument("--case", help="只跑指定 case id，逗号分隔")
    parser.add_argument("--update-baseline", action="store_true", help="把本次结果接受为基线")
    args = parser.parse_args()

    only = set(args.case.split(",")) if args.case else None
    cases = load_cases(only)
    agent = create_customer_service_agent(checkpointer=InMemorySaver())

    print(f"评测 {len(cases)} 条 case（真实模型，tracing={os.environ['LANGSMITH_TRACING']}）")
    runs: list[CaseRun] = []
    for case in cases:
        print(f"\n[{case['id']}] 用户：{case['user']}", flush=True)
        run = run_case(case, agent)
        try:
            evaluate(case, run)
        except (KeyError, ValueError) as exc:
            # dataset 写错（如 outcome 没有短语映射）只该废掉这一条 case，不该毁掉整轮报表。
            run.error = run.error or f"case 配置无效: {exc}"
        runs.append(run)

    exit_code = report(runs, load_baseline(), args.update_baseline)

    # 哨兵：真实 Mock 数据必须逐字节未被改动（写入只落在 isolated_data 的副本上）。
    assert (
        REAL_ORDERS_FILE.read_text(encoding="utf-8") == REAL_ORDERS_TEXT
    ), "真实 data/mock/orders.json 被改动了！评测的数据隔离失效"
    print("真实 data/mock/orders.json 未被改动: OK")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:  # case id 写错、dataset 字段拼错都属于用法问题，不需要 traceback
        raise SystemExit(f"评测未启动: {exc}")
