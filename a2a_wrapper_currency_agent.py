from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, Dict
from typing_extensions import Final
from database_agent import AgentResponse
from currency_agent_autogen import currency_agent, get_currency_agent



class CurrencyAgent:
    """Adapter around the Autogen *AssistantAgent* so that it behaves like the
    `DatabaseAgent` wrapper (invoke / stream API, same return envelope).
    """
    def __init__(self):
        self.autogen_agent = get_currency_agent()

    # For parity with the database wrapper – consumers can check these to know
    # which MIME types we can handle.
    SUPPORTED_CONTENT_TYPES = [
        "text",
        "text/plain",
    ]

    # ──────────────────────────────────────────────────────────────────────
    # Synchronous interface
    # ──────────────────────────────────────────────────────────────────────
    async def invoke(self, query: str, sessionId: str | None = None) -> Dict[str, Any]:
        """Return the TaskResult produced by Autogen, asynchronously."""
        task_result = await self.autogen_agent.run(task=query)
        final_msg = task_result.messages[-1].content

        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_msg,
            "structured_response": {
                "status": "completed",
                "message": final_msg,
            },
            "state": task_result,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Streaming interface
    # ──────────────────────────────────────────────────────────────────────
    async def stream(
        self,
        query: str,
        sessionId: str | None = None,
    ) -> AsyncIterable[Dict[str, Any]]:
        """Yield the assistant's reply token‑by‑token (or chunk‑by‑chunk).

        The underlying Autogen `.run_stream()` returns an **async** iterator that
        yields either strings or structured message objects. We convert each
        chunk into the standard envelope so that the UI can handle it exactly
        the same way it handles the database wrapper.
        """

        # Obtain the async generator from Autogen.
        stream_gen = self.autogen_agent.run_stream(task=query)

        # Autogen provides an *async* generator, but some versions may return an
        # *iterator* of Futures. We normalise by wrapping with `aiter`. This will
        # fail fast with a helpful message if the object is not iterable.
        async for chunk in stream_gen:  # type: ignore[func-returns-value]
            # Similar to the blocking path: attempt `.content` then `str()`.
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": getattr(chunk, "content", str(chunk)),
            }

        # Final completion envelope so that the caller knows the stream ended.
        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": "",  # Content already streamed above.
        }

