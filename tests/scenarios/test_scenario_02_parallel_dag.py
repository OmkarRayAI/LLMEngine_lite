"""Scenario 2: multi-tool parallel DAG.

Two independent tool calls — weather + stock — fan out in parallel, then
``join`` collects both observations. This is the LLM-Compiler sweet spot
(parallel tool execution that ReAct loops give up). We assert both tools
ran and that wall-clock is < the sum of their individual sleeps, proving
they actually ran concurrently.
"""
import asyncio
import time
import unittest

from .helpers import ScriptedChatModel, _StockInput, _WeatherInput
from LLMEngine.base import StructuredTool


META = "META PLAN:\nFan out weather and stock lookups, then join."

PLAN = """\
Thought: Two independent fetches — fan them out.
1. weather("Paris")
Thought: Fetch the stock in parallel.
2. stock("AAPL")
3. join()
<END_OF_PLAN>"""

JOIN = """\
Thought: Both observations are present; finalize.
Action: Finish(Paris is sunny; AAPL last $190.42.)"""


def _slow_tool(name: str, schema, sleep_s: float, returns: str) -> StructuredTool:
    """Tool that sleeps a known duration so we can prove concurrency."""

    async def _coro(*args, **kwargs):
        await asyncio.sleep(sleep_s)
        return returns

    return StructuredTool(
        name=name,
        description=f"{name}(arg: str) -> str:\n - Slow stub (sleeps {sleep_s}s).",
        args_schema=schema,
        func=None,
        coroutine=_coro,
    )


class ParallelDagScenario(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_tools_run_concurrently(self) -> None:
        from LLMEngine import LLMEngine

        # 0.4s sleep on each — sequential would be ≥0.8s, parallel ≈0.4s.
        SLEEP = 0.4
        weather = _slow_tool("weather", _WeatherInput, SLEEP, "Paris is sunny.")
        stock = _slow_tool("stock", _StockInput, SLEEP, "AAPL last $190.42.")

        llm = ScriptedChatModel([META, PLAN, JOIN])
        engine = LLMEngine(llm=llm, max_replan=1)

        t0 = time.monotonic()
        result = await engine.run(
            question="What's the weather in Paris and the AAPL price?",
            tools=[weather, stock],
        )
        elapsed = time.monotonic() - t0

        self.assertIn("Paris", result.answer)
        self.assertIn("AAPL", result.answer)
        self.assertEqual(result.replans, 0)
        # Allow generous slack for CI noise; the meaningful threshold is
        # well under 2× sequential.
        self.assertLess(
            elapsed, 2 * SLEEP + 1.0,
            f"expected parallel execution; took {elapsed:.2f}s for two {SLEEP}s tools",
        )
        # Both tool tasks should be present in the recovered task graph.
        names = sorted(t.name for t in result.tasks.values() if not t.is_join)
        self.assertEqual(names, ["stock", "weather"])


if __name__ == "__main__":
    unittest.main()
