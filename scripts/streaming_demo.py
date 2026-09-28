from langchain_core.messages import HumanMessage

from app.agent import create_customer_service_agent


def main():
    agent=create_customer_service_agent()

    question=input('用户：')

    for chunk in agent.stream(
        {"messages":[HumanMessage(content=question)]},
        stream_mode="messages",
    ):
        # print(
        #     "NODE:",
        #     metadata.get("langgraph_node"),
        #     "CONTENT:",
        #     chunk.content,
        #     "TOOL_CALLS:",
        #     chunk.tool_calls,
        # )
        print(chunk)

if __name__ == '__main__':
    main()


