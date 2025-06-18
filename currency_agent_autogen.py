from typing import Literal, Annotated
import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool

# ───────────────────────────────────────────────────────────────────────────────
open_ai = OpenAIChatCompletionClient(model="gpt-4o-mini")

CurrencySymbol = Literal["USD", "EUR"]

def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")

def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"],
    quote_currency: Annotated[CurrencySymbol, "Quote currency"],
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"

tool = FunctionTool(currency_calculator, description="Currency exchange calculator.", strict=True)

# Key change ⬇︎: Tell the LLM to *finish* every answer with the follow-up line.
currency_agent = AssistantAgent(
    name="chatbot",
    system_message=(
        "For currency-exchange tasks, use only the provided function.\n"
        "After you output the result, end the same message with:\n"
        "'Is there anything else I can do?'"
    ),
    model_client=open_ai,
    tools=[tool],
)

def get_currency_agent():
    return currency_agent

async def main() -> None:
    print("💬  Type your request (or 'exit' to quit).")
    while True:
        prompt = await asyncio.get_running_loop().run_in_executor(None, input, "\nYou: ")
        if prompt.strip().lower() in {"exit", "quit", "bye"}:
            print("chatbot: Goodbye! 👋")
            break

        # stream the assistant’s reply (which already contains the follow-up sentence)
        stream = currency_agent.run_stream(task=prompt)
        await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())
