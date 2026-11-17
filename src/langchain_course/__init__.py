from dotenv import load_dotenv

load_dotenv()

from enum import StrEnum

# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class Operator(StrEnum):
    ADD = "ADDITION"
    SUB = "SUBTRACTION"
    MULT = "MULTIPLICATION"
    DIV = "DIVISION"


@tool
def basic_arithmetic(operator: Operator, operand1: float, operand2: float) -> float:
    """
    Applies the mathematical operation to the two operands operand1 and operand2.
    For example, basic_arithmetic("ADDITION", 1.5, 2.6) returns 4.1 (1.5 + 2.6)

    Args:
        operator (Operator): The applied mathematical operation. One of ["ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION"]
        operand1 (float): The left operand to the operator
        operand2 (float): The right operand to the operator

    Returns:
        result (float) | error (String): The result of applying the operator to operand1 and operand 2 or a String describing an error.
    """
    match operator:
        case Operator.ADD:
            return operand1 + operand2
        case Operator.SUB:
            return operand1 - operand2
        case Operator.MULT:
            return operand1 * operand2
        case Operator.DIV:
            return operand1 / operand2
        case _:
            return f"Error: invalid arguments to function basic_arithmetic with operator {operator} and operands {operand1} and {operand2}."


def main() -> None:
    # llm = ChatOllama(model="gpt-oss:latest", temperature=0.1)
    llm = ChatOllama(model="qwen3.5:latest", temperature=0.1)
    tools = [basic_arithmetic]
    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="What is 83 + 95 * 3014? You must use tools to answer this question. Do not use your internal knowledge."
            )
        }
    )

    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"User Message: {message.content}")
        elif isinstance(message, AIMessage):
            if len(message.tool_calls) > 0:
                for tool_call in message.tool_calls:
                    print(
                        f"Tool Called: {tool_call['name']}. Arguments: {tool_call['args']}. ",
                        end="",
                    )
            else:
                print(f"AI Message: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Output: {message.content}")


if __name__ == "__main__":
    main()
