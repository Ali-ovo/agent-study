"""Define the agent's tools."""

import os
import uuid
from functools import lru_cache
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_tavily import TavilySearch
from langgraph.store.base import BaseStore


@lru_cache(maxsize=1)
def get_tavily_search() -> TavilySearch:
    """Return a shared Tavily search tool instance."""
    return TavilySearch(
        max_results=5,
        tavily_api_key=os.environ["TAVILY_API_KEY"],
    )


def get_agent_tools() -> list[BaseTool]:
    """Tools exposed to the LLM."""
    return [get_tavily_search(), upsert_memory]


async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"
