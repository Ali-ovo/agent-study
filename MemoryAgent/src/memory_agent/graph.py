"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime
from typing import cast

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    user_id = runtime.context.user_id
    model = runtime.context.model
    system_prompt = runtime.context.system_prompt

    # Retrieve the most recent memories for context
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    # This helps the model understand the context and temporal relevance
    sys = system_prompt.format(user_info=formatted, time=datetime.now().isoformat())

    # Load the chat model from the runtime context
    llm = utils.load_chat_model(model)

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.
    msg = await llm.bind_tools(tools.get_agent_tools()).ainvoke(
        [{"role": "system", "content": sys}, *state.messages]
    )
    return {"messages": [msg]}


async def _run_tool_call(
    tc: dict,
    *,
    user_id: str,
    store: BaseStore,
) -> str:
    if tc["name"] == "upsert_memory":
        return await tools.upsert_memory(
            **tc["args"],
            user_id=user_id,
            store=store,
        )
    if tc["name"] == "tavily_search":
        result = await tools.get_tavily_search().ainvoke(tc["args"])
        return result if isinstance(result, str) else str(result)
    raise ValueError(f"Unknown tool: {tc['name']}")


async def execute_tools(state: State, runtime: Runtime[Context]):
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    outputs = await asyncio.gather(
        *(
            _run_tool_call(
                tc,
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
            )
            for tc in tool_calls
        )
    )

    return {
        "messages": [
            {
                "role": "tool",
                "content": output,
                "tool_call_id": tc["id"],
            }
            for tc, output in zip(tool_calls, outputs)
        ]
    }


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        return "execute_tools"
    return END


# Create the graph + all nodes
builder = StateGraph(State, context_schema=Context)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(execute_tools)
builder.add_conditional_edges("call_model", route_message, ["execute_tools", END])
builder.add_edge("execute_tools", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]
