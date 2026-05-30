"""Test scaffolding for Agno-style scenario coverage.

The engine drives an LLM through three stages per run: ``meta_planner`` (one
``ainvoke``), then per-iteration ``planner`` (one ``_agenerate`` from a chat
model) and ``joiner`` (one ``agenerate_prompt``). All three eventually land on
the chat model's ``_generate`` / ``_agenerate``, so a stub that overrides
those is enough to script a full run.

The ``ScriptedChatModel`` below picks responses by inspecting the prompt for
stage markers (``Meta Plan:`` for meta-planner, ``Action: Finish/Replan`` is
what the joiner produces, etc.). Tests just provide an ordered list of
responses; the model returns them in order, panicking if the script runs out.
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence

# Ensure the repo root is on sys.path so ``from LLMEngine import ...`` works
# without an editable install.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, ChatGeneration, ChatResult
from pydantic import BaseModel, ConfigDict, Field

from LLMEngine.base import StructuredTool


# ---------------------------------------------------------------------------
# Scripted chat model
# ---------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """Returns canned responses in order, regardless of prompt content.

    We avoid pattern-matching on the prompt because a single run produces
    several distinct prompts (meta-plan, plan, plan-replan, join, join-final),
    and the test author already knows the order. Out-of-script calls raise so
    the failure is loud, not silent.
    """

    responses: List[str] = Field(default_factory=list)
    transcript: List[str] = Field(default_factory=list)
    cursor: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, responses: Sequence[str]) -> None:
        super().__init__(responses=list(responses))

    @property
    def _llm_type(self) -> str:
        return "scripted-chat"

    def _next(self, prompt_excerpt: str) -> str:
        if self.cursor >= len(self.responses):
            raise AssertionError(
                f"ScriptedChatModel ran out of responses after {self.cursor}; "
                f"prompt excerpt was: {prompt_excerpt[:200]!r}"
            )
        out = self.responses[self.cursor]
        self.cursor += 1
        self.transcript.append(prompt_excerpt[:400])
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(getattr(m, "content", str(m)) for m in messages)
        text = self._next(prompt)
        msg = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=msg, text=text)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ---------------------------------------------------------------------------
# Fake tools
# ---------------------------------------------------------------------------


class _SearchInput(BaseModel):
    query: str


class _WeatherInput(BaseModel):
    city: str


class _StockInput(BaseModel):
    ticker: str


class _BrokenInput(BaseModel):
    payload: str


def _make_tool(name: str, description: str, schema, returns: str):
    # The engine's TaskFetchingUnit invokes tools with positional args parsed
    # out of the planner output (e.g. ``search("foo")`` → call with one arg).
    # Test tools must therefore accept positional args, not just kwargs.
    async def _coro(*args, **kwargs):
        return returns
    return StructuredTool(
        name=name,
        description=description,
        args_schema=schema,
        func=None,
        coroutine=_coro,
    )


def search_tool(returns: str) -> StructuredTool:
    return _make_tool(
        "search",
        "search(query: str) -> str:\n - Web search; returns a short summary.",
        _SearchInput,
        returns,
    )


def weather_tool(returns: str) -> StructuredTool:
    return _make_tool(
        "weather",
        "weather(city: str) -> str:\n - Returns current weather for a city.",
        _WeatherInput,
        returns,
    )


def stock_tool(returns: str) -> StructuredTool:
    return _make_tool(
        "stock",
        "stock(ticker: str) -> str:\n - Returns current price summary for a ticker.",
        _StockInput,
        returns,
    )


def flaky_tool(first_returns: str, then_returns: str) -> StructuredTool:
    """A tool that fails once (returns an error string) then succeeds.

    The engine doesn't replan on tool *exceptions* — it replans when the
    *joiner* asks. So this tool returns an error-shaped string and the
    joiner script asks for a replan; the second iteration's plan calls a
    different tool that succeeds.
    """
    state = {"calls": 0}

    async def _coro(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return first_returns
        return then_returns

    return StructuredTool(
        name="brittle",
        description="brittle(payload: str) -> str:\n - Sometimes errors out.",
        args_schema=_BrokenInput,
        func=None,
        coroutine=_coro,
    )
