import shutil
import tempfile
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver

import yaml
from langgraph.types import Command

from app.agent.graph import create_customer_service_agent
from app import data_loader


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "eval" / "dataset" / "cancel.yaml"


def load_case():
    data = yaml.safe_load(DATASET_FILE.read_text(encoding="utf-8"))
    return data["cases"][0]


def run_case(agent, case):
    config = {"configurable": {"thread_id": "eval-hitl"}}

    first_result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": case["user"],
                }
            ]
        },
        config=config,
    )

    print(first_result)

    interrupts = first_result.get("__interrupt__", [])

    if not interrupts:
        return None, None, "没有发生 interrupt"

    interrupt = interrupts[0]

    print("\n\nInterrupt:")
    print(interrupt.value)

    decision = case["hitl"]["decision"]

    print("\nHuman Decision:")
    print(decision)

    final_result = agent.invoke(
        Command(resume=decision),
        config=config,
    )

    return interrupt, final_result, None


def evaluate(case, interrupt, final_result):
    expect = case["expect"]

    # 1. 检查 interrupt
    interrupt_value = interrupt.value

    if interrupt_value.get("type") != expect["interrupt"]["type"]:
        return False, "interrupt type 不正确"

    if interrupt_value.get("current_status") != expect["interrupt"]["current_status"]:
        return False, "interrupt current_status 不正确"

    # 2. 检查 Tool
    messages = final_result["messages"]

    tool_calls = [
        call
        for message in messages
        if hasattr(message, "tool_calls")
        for call in message.tool_calls
    ]

    actual_tools = [call["name"] for call in tool_calls]

    for tool in expect.get("tools", []):
        if tool not in actual_tools:
            return False, f"缺少 Tool: {tool}"

    for tool in expect.get("not_tools", []):
        if tool in actual_tools:
            return False, f"调用了不应该调用的 Tool: {tool}"

    # 3. 检查最终回答
    final_answer = messages[-1].content

    for text in expect.get("answer_include", []):
        if text not in final_answer:
            return False, f"回答缺少: {text}"

    for text in expect.get("answer_exclude", []):
        if text in final_answer:
            return False, f"回答包含禁止内容: {text}"

    return True, "PASS"


def main():
    case = load_case()

    checkpointer = InMemorySaver()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_orders = Path(temp_dir) / "orders.json"

        shutil.copy(
            data_loader.ORDERS_FILE,
            temp_orders,
        )

        original_orders_file = data_loader.ORDERS_FILE
        data_loader.ORDERS_FILE = temp_orders

        try:
            agent = create_customer_service_agent(
                checkpointer=checkpointer,
            )

            interrupt, final_result, error = run_case(
                agent,
                case,
            )

            if error:
                print(f"\nFAIL: {error}")
                return

            passed, message = evaluate(
                case,
                interrupt,
                final_result,
            )

            print("\nFinal Answer:")
            print(final_result["messages"][-1].content)

            if passed:
                print("\nResult: PASS")
            else:
                print(f"\nResult: FAIL - {message}")

        finally:
            data_loader.ORDERS_FILE = original_orders_file


if __name__ == "__main__":
    main()
