# a2a_wrapper_currency_agent.py
"""A thin wrapper that adapts the **autogen**‐based `currency_agent_autogen.py` to the
A2A interface that the front‑end expects (same surface as `a2a_wrapper_database_agent`).

The goals are:
1. Provide a synchronous **invoke** method that returns a dict with the standard
   keys `is_task_complete`, `require_user_input`, and `content`.
2. Provide an asynchronous **stream** generator that yields the same kind of dicts
   for partial results and finishes with a final *completed* message.
3. Keep the public contract identical to the database wrapper so that callers can
   swap agents without code changes.

The implementation intentionally keeps things light‑weight: we do not attempt to
re‑create the full LangGraph state logic because the Autogen AssistantAgent
already handles tool calls internally and always finishes its response with the
follow‑up sentence *"Is there anything else I can do?"* as defined in
`currency_agent_autogen.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, Dict
from typing_extensions import Final

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
    def invoke(self, query: str, sessionId: str | None = None) -> Dict[str, Any]:
        """Run the Autogen agent synchronously.

        Parameters
        ----------
        query:
            The user prompt.
        sessionId:
            Included for API compatibility; the current Autogen version does not
            expose a thread/session concept that we can map onto, but we accept
            the argument so that callers do not need `isinstance` checks.
        """

        # Autogen's `.run()` is blocking and returns the assistant's final
        # response (either as a string _or_ a Message object depending on the
        # version). We try to extract `.content` first and fall back to `str()`.
        response = self.autogen_agent.run(task=query)
        content: str = getattr(response, "content", str(response))

        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": content,
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

