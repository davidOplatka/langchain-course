from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

MAX_AGENT_ITERATIONS = 10
MODEL = "qwen3.5:latest"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >>Executing get_product_price(product='{product}')")
    prices = {
        "phone": 499.99,
        "tablet": 349.99,
        "laptop": 1299.99,
        "headphones": 249.99,
        "keyboard": 49.99,
        "mouse": 19.99,
    }

    return prices.get(product, 0)


@tool
def apply_discount(product_price: float, discount_tier: str) -> float:
    """Apply a discount tier to a product price and return the price of the product.

    Available tiers: bronze, silver, gold
    """
    print(
        f"    >> Executing apply_discount(product_price={product_price}, discount_tier={discount_tier})"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 25}

    discount = discount_percentages.get(discount_tier, 0)

    return round(product_price * (1 - discount / 100), 2)


def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    # Attaches the tools to the LLM for tool calling
    llm_with_tools = llm.bind_tools(tools)

    print(f"User query: {query}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "Your job is to help determine the prices of products from a catalog."
                f"You have access to the following tools: {tools}.\n\n"
                "Instructions:",
                "1. NEVER guess or assume the price for any product.",
                "You MUST look up the price from the catalog using the get_product_price tool.\n",
                "2. Only call the apply_discount tool after you have determined the original sales price from the catalog.\n",
                "3. If the user does not specify a discount tier, do not apply one yourself.",
                "You may ask the user if they have a discount as a follow-up.",
            )
        ),
        HumanMessage(content=query),
    ]


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    result = run_agent("What is the price of a laptop if I have a gold discount?")
