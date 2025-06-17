from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

import asyncio
from typing import Any, AsyncIterable, Dict, Literal, Sequence, Union, cast, List
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

from typing_extensions import ClassVar


#StructuredResponse = Union[dict, BaseModel]
StructuredResponseSchema = Union[dict, type[BaseModel]]


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    messages: List[BaseMessage]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    status: Literal["input_required", "completed", "error"] = "input_required"
    structured_response: dict

    model_config = ConfigDict(extra="forbid")

class AgentResponse(BaseModel):
    status: Literal["input_required", "completed", "error"]
    message: str

    # Pydantic-v2 style – forces `additionalProperties: false`
    model_config = ConfigDict(extra="forbid")

# Define a dummy state class
class DummyState(TypedDict, total=False):
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps
    structured_response: AgentResponse   # parsed result will be stored her

def get_database_agent(memory_saver):

    def generate_structured_response(state: DummyState,
                                    config: RunnableConfig) -> DummyState:
        last_user_msg = state["messages"][-1].content

        open_ai = ChatOpenAI(model="gpt-4o-mini").with_structured_output(AgentResponse)

        parsed: AgentResponse = open_ai.invoke([
            ("system", "If you get a question, you must return a number - even if you do not know the answer. In that case, just make something up."),
            ("human", last_user_msg)
            ])   # parsed is already a Pydantic object
        return {
            "structured_response": parsed,          # keep it in state if you want
            "messages": state["messages"],          # pass through other state
            "remaining_steps": state["remaining_steps"]
        }
    
    # Create the LangGraph StateGraph
    graph = StateGraph(DummyState)

    # Add a node to the graph
    graph.add_node("dummy_node", generate_structured_response)

    # Set the entry point
    graph.set_entry_point("dummy_node")

    # Define the transitions: from dummy_node to END
    graph.add_edge("dummy_node", END)

    # Compile the graph into an executable form
    database_agent = graph.compile(checkpointer=memory_saver)

    return database_agent
