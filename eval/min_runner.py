from pathlib import Path

import yaml

from app.agent.graph import create_customer_service_agent


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "eval" / "dataset" / "routing.yaml"


def load_cases():
    data = yaml.safe_load(DATASET_FILE.read_text(encoding="utf-8"))
    return data["cases"]


def run_case(agent, case):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": case["user"]}]}
    )

    messages = result["messages"]


    tool_calls = [
        call
        for message in messages
        if hasattr(message, "tool_calls")
        for call in message.tool_calls
    ]

    tool_outputs = [
        message.content
        for message in messages
        if message.type == "tool"
    ]

    final_answer = messages[-1].content

    return tool_calls,tool_outputs, final_answer


def evaluate(case, tool_calls,tool_outputs, final_answer):
    expect = case["expect"]

    actual_tools = [call["name"] for call in tool_calls]

    # 检查应该调用的 Tool
    for tool in expect.get("tools", []):
        if tool not in actual_tools:
            return False, f"缺少 Tool: {tool}"

    # 检查不应该调用的 Tool
    for tool in expect.get("not_tools", []):
        if tool in actual_tools:
            return False, f"调用了不应该调用的 Tool: {tool}"

    # 检查订单号
    expected_order_id = expect.get("order_id")

    if expected_order_id:
        for call in tool_calls:
            if "order_id" in call["args"]:
                if call["args"]["order_id"] != expected_order_id:
                    return False, "Tool 参数中的 order_id 不正确"

    # 检查 Tool Outputs
    tool_output_text="\n".join(str(output) for output in tool_outputs)

    if expected_order_id and expected_order_id not in tool_output_text:
        return False, "Tool Output 中没有找到预期订单号"

    # 检查回答必须包含的内容
    for text in expect.get("answer_include", []):
        if text not in final_answer:
            return False, f"回答缺少: {text}"

    # 至少包含一个
    answer_include_any = expect.get("answer_include_any", [])

    if answer_include_any:
        if not any(text in final_answer for text in answer_include_any):
            return False, "回答没有包含任何一个预期内容"

    # 检查回答不能包含的内容
    for text in expect.get("answer_exclude", []):
        if text in final_answer:
            return False, f"回答包含禁止内容: {text}"

    return True, "PASS"


def main():
    cases = load_cases()
    agent = create_customer_service_agent()

    for case in cases:
        tool_calls,tool_outputs, final_answer = run_case(agent, case)


        passed, message = evaluate(
            case,
            tool_calls,
            tool_outputs,
            final_answer,
        )

        print(f"\n[{case['id']}] {case['user']}")
        print(f"Tools: {[call['name'] for call in tool_calls]}")
        print(f"Tool Outputs: {tool_outputs}")
        print(f"Answer: {final_answer}")
        print(f"Result: {message}")


if __name__ == "__main__":
    main()
