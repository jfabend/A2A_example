from typing import Literal, Annotated
import asyncio

from pydantic import BaseModel, Field
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool

# ── set up the OpenAI client ────────────────────────────────────────────────────
open_ai = OpenAIChatCompletionClient(model="gpt-4o-mini")

# ── currency-exchange helper and tool definition ───────────────────────────────
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

tool = FunctionTool(
    currency_calculator,
    description="Currency exchange calculator.",
    strict=True
)

# ── assistant agent ────────────────────────────────────────────────────────────
agent = AssistantAgent(
    name="chatbot",
    system_message=(
        "For currency-exchange tasks, only use the function you have been "
        "provided with. Reply when the task is done."
    ),
    model_client=open_ai,
    tools=[tool],
)

# ── interactive main loop ──────────────────────────────────────────────────────
async def main() -> None:
    print("💬  Type your request (or 'exit' to quit).")
    while True:
        # get user input without blocking the event loop
        prompt = await asyncio.get_running_loop().run_in_executor(
            None, input, "\nYou: "
        )
        if prompt.strip().lower() in {"exit", "quit", "bye"}:
            print("chatbot: Goodbye! 👋")
            break

        # run the agent and stream the result to the console
        stream = agent.run_stream(task=prompt)
        await Console(stream)

        # follow-up prompt
        print("\nchatbot: Is there anything else I can do?")

if __name__ == "__main__":
    asyncio.run(main())
