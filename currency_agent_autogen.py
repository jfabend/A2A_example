
from typing import Literal
import os
import asyncio

from pydantic import BaseModel, Field
from typing_extensions import Annotated
from autogen_ext.models.openai import OpenAIChatCompletionClient

from database_agent import AgentResponse

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

import autogen
from autogen.cache import Cache
from autogen_core.tools import FunctionTool

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

agent = AssistantAgent(
    name="chatbot",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply when the task is done.",
    model_client=open_ai,
    tools=[tool],
    output_content_type=AgentResponse
)

async def main() -> None:
    stream = agent.run_stream(task="How much is 100 USD in Euros?")
    await Console(stream)


asyncio.run(main())