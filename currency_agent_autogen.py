from typing import Literal, Annotated
import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool

open_ai = OpenAIChatCompletionClient(model="gpt-4o-mini")

CurrencySymbol = Literal["USD", "EUR"]

def exchange_rate(base: CurrencySymbol, quote: CurrencySymbol) -> float:
    return 1.0 if base == quote else (1/1.1 if (base, quote)==("USD","EUR") else 1.1)

def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"],
    quote_currency: Annotated[CurrencySymbol, "Quote currency"],
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"

# 👇 *return_direct=True* makes the tool’s return value the assistant reply
tool = FunctionTool(
    currency_calculator,
    description="Currency exchange calculator.",
    strict=True,
)

currency_agent = AssistantAgent(
    name="chatbot",
    system_message=(
        "For currency-exchange tasks, use only the provided function.\n"
        "After you output the result, end the same message with:\n"
        "'Is there anything else I can do?'"
    ),
    model_client=open_ai,
    tools=[tool],
    reflect_on_tool_use=True,
)

def get_currency_agent():
    return currency_agent

async def main() -> None:
    print("💬  Type your request (or 'exit' to quit).")
    while True:
        prompt = await asyncio.get_running_loop().run_in_executor(
            None, input, "\nYou: "
        )
        if prompt.strip().lower() in {"exit", "quit", "bye"}:
            print("chatbot: Goodbye! 👋")
            break

        # 1️⃣  Await the coroutine …
        task_result = await currency_agent.run(task=prompt)

        # 2️⃣  … grab the assistant’s final reply (last message) …
        final_reply = task_result.messages[-1].content

        # 3️⃣  … and print it.
        print(f"chatbot: {final_reply}")

if __name__ == "__main__":
    asyncio.run(main())
