import asyncio
from typing import Any, AsyncIterable, Dict, Literal, Sequence, Union
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from database_agent import get_database_agent
from database_agent import AgentResponse
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import add_messages

StructuredResponse = Union[dict, BaseModel]

memory = MemorySaver()

#TODO INSTEAD OF SINGLE A2A WRAPPERS FOR EACH AGENT WE COULD CREATE AN ABSTRACT A2A AGENT CLASS

# def _fetch_mcp_tools_sync() -> list:
#     """
#     Helper function: runs the async MultiServerMCPClient code in a synchronous manner.
#     Fetches the remote tools from your MCP server(s).
#     """
#     servers_config = {
#         "currency_server": {
#             "transport": "sse",
#             "url": "http://127.0.0.1:3000/sse",
#         }
#     }

#     async def _fetch_tools():
#         async with MultiServerMCPClient(servers_config) as client:
#             return client.get_tools()

#     # Run the async method in a blocking (sync) fashion
#     return asyncio.run(_fetch_tools())

class DatabaseAgent:

    def __init__(self):
        self.graph = get_database_agent(memory)

    def invoke(self, query, sessionId) -> str:
       config = {"configurable": {"thread_id": sessionId}}
       print("call Database Agent")
       self.graph.invoke({"messages": [("user", query)]}, config)
       print("got answer from Database agent")
       return self.get_agent_response(config)

    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the exchange rates...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the exchange rates..",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        print("Current State of the database agent after calling it: ")
        print(current_state)
        structured_response = current_state.values.get("structured_response")

        print("Structured response of the database agent after calling it: ")
        print(structured_response)

        if structured_response and isinstance(structured_response, AgentResponse):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
