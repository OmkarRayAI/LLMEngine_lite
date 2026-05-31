"""Real-LLM smoke against the four scenarios validated by the stub suite.

Picks an OpenRouter-hosted model and runs:

  1. single-tool agent
  2. multi-tool parallel DAG
  3. RAG over a temporary LLMWiki
  4. tool-error → replan

Each run reports answer / replans / wall-clock / tool tasks, so divergences
from the stub-test expectations are obvious. This is **not** a unit test —
it costs tokens and may flake on model-side variance. Run by hand:

    OPENROUTER_API_KEY=... python examples/real_llm_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import textwrap
import time
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from LLMEngine import KnowledgeTool, LLMEngine, LLMWikiRetriever
from LLMEngine.base import StructuredTool


# ---- helpers ----------------------------------------------------------------


def make_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running.")
    return ChatOpenAI(
        model=os.environ.get("LLMENGINE_MODEL", "moonshotai/kimi-k2"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=1,
        timeout=60,
    )


class _Q(BaseModel):
    query: str


class _C(BaseModel):
    city: str


class _T(BaseModel):
    ticker: str


class _P(BaseModel):
    payload: str


def fake_tool(name: str, schema, returns: str, description: str) -> StructuredTool:
    """Tool whose body is canned but is invoked by a *real* LLM's plan."""

    async def _coro(*args, **kwargs):
        return returns

    return StructuredTool(
        name=name,
        description=description,
        args_schema=schema,
        func=None,
        coroutine=_coro,
    )


def flaky(then: str) -> StructuredTool:
    """First call returns an error string, second call succeeds."""
    state = {"calls": 0}

    async def _coro(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return "ERROR: upstream 503"
        return then

    return StructuredTool(
        name="brittle",
        description=("brittle(payload: str) -> str:\n"
                     " - Sometimes returns 'ERROR: ...'. If it does, abandon it\n"
                     "   and use `search` for the same query in the next plan."),
        args_schema=_P,
        func=None,
        coroutine=_coro,
    )


def _make_wiki(td: str) -> None:
    os.makedirs(os.path.join(td, "wiki"), exist_ok=True)
    open(os.path.join(td, "index.md"), "w").write(
        "# Index\n- [[wiki/banking-fy25]] — Indian banking FY25 summary\n"
    )
    open(os.path.join(td, "wiki", "banking-fy25.md"), "w").write(textwrap.dedent("""\
        ---
        title: Indian Banking FY25
        type: summary
        sources: [raw/icra-fy25.pdf]
        updated: 2026-04-01
        ---

        Indian private sector banks reported strong NIMs in FY25.
        Provisions ticked up in Q4 amid retail unsecured stress.
        Public sector banks continued asset quality recovery, with GNPA
        ratios at multi-year lows. Credit growth moderated to ~12%.
    """))


def banner(title: str) -> None:
    print(f"\n{'='*72}\n {title}\n{'='*72}")


def report(label: str, result, started: float) -> None:
    elapsed = time.monotonic() - started
    tasks = [t.name for t in result.tasks.values() if not t.is_join]
    print(
        f"[{label}] elapsed={elapsed:.2f}s  replans={result.replans}  "
        f"tools_run={tasks}"
    )
    print("answer:")
    print("  " + (result.answer or "<empty>").replace("\n", "\n  "))


# ---- scenarios --------------------------------------------------------------


async def s1_single_tool(llm) -> None:
    banner("Scenario 1 — single agent + 1 tool")
    engine = LLMEngine(llm=llm, max_replan=1)
    tool = fake_tool(
        "search", _Q,
        returns="Paris is the capital and largest city of France.",
        description="search(query: str) -> str:\n - Web search; returns one-line summary.",
    )
    t0 = time.monotonic()
    result = await engine.run(
        question="What is the capital of France?",
        tools=[tool],
    )
    report("single", result, t0)


async def s2_parallel_dag(llm) -> None:
    banner("Scenario 2 — multi-tool parallel DAG")
    engine = LLMEngine(llm=llm, max_replan=1)
    weather = fake_tool(
        "weather", _C, "Paris: 18°C, partly cloudy.",
        "weather(city: str) -> str:\n - Returns current weather for a city.",
    )
    stock = fake_tool(
        "stock", _T, "AAPL last $190.42 (+0.6% intraday).",
        "stock(ticker: str) -> str:\n - Returns latest price for a ticker.",
    )
    t0 = time.monotonic()
    result = await engine.run(
        question="Tell me the weather in Paris and the latest AAPL price.",
        tools=[weather, stock],
    )
    report("parallel", result, t0)


async def s3_rag(llm) -> None:
    banner("Scenario 3 — RAG over LLMWiki")
    with tempfile.TemporaryDirectory() as td:
        _make_wiki(td)
        kt = KnowledgeTool(
            LLMWikiRetriever(root=td),
            name="wiki",
            description=(
                "wiki(query: str, k: int = 5) -> str:\n"
                " - Searches the team wiki and returns top-k snippets with `source=`."
                " ALWAYS query before answering. Cite sources in the final answer."
            ),
        )
        engine = LLMEngine(llm=llm, max_replan=1)
        t0 = time.monotonic()
        result = await engine.run(
            question="What did Indian private banks report in FY25 — NIMs and provisions?",
            tools=[kt.get_tool()],
        )
        report("rag", result, t0)


async def s4_replan(llm) -> None:
    banner("Scenario 4 — tool error → replan")
    engine = LLMEngine(llm=llm, max_replan=3)
    brittle = flaky(then="(unreachable)")
    search = fake_tool(
        "search", _Q,
        returns="hello world (via search fallback)",
        description="search(query: str) -> str:\n - Reliable search fallback.",
    )
    t0 = time.monotonic()
    result = await engine.run(
        question="Greet me with 'hello world'.",
        tools=[brittle, search],
        instructions=(
            "Try `brittle` first. If its observation starts with 'ERROR:',"
            " replan and use `search` for the same query."
        ),
    )
    report("replan", result, t0)


async def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    llm = make_llm()
    print(f"model = {llm.model_name}")
    for fn in (s1_single_tool, s2_parallel_dag, s3_rag, s4_replan):
        try:
            await fn(llm)
        except Exception as exc:
            print(f"[{fn.__name__}] ERRORED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
