"""Scenario 1: single agent, single tool — Agno's "first agent" baseline.

Mirrors Agno's hello-world Agent example: one tool wired in, one user
question, the agent calls the tool once and returns. This validates the
full path with the simplest topology so deviations from the contract are
obvious.
"""
import asyncio
import unittest

from .helpers import ScriptedChatModel, search_tool


META = (
    "META PLAN:\n"
    "Use the search tool to retrieve a one-line summary, then finish."
)

PLAN = """\
Thought: A single search call answers this.
1. search("capital of france")
2. join()
<END_OF_PLAN>"""

JOIN = """\
Thought: The search returned the answer.
Action: Finish(Paris is the capital of France.)"""


class SingleToolScenario(unittest.IsolatedAsyncioTestCase):
    async def test_single_tool_one_shot(self) -> None:
        from LLMEngine import LLMEngine

        llm = ScriptedChatModel([META, PLAN, JOIN])
        engine = LLMEngine(llm=llm, max_replan=1)
        tool = search_tool(returns="Paris is the capital of France.")

        result = await engine.run(question="What is the capital of France?", tools=[tool])

        self.assertIn("Paris", result.answer)
        self.assertEqual(result.replans, 0)
        # Stats are always-on. Today, the planner/joiner callback chain is
        # what gets recorded; the meta-planner uses llm.ainvoke() which
        # bypasses the stats handler. Either way, we must see at least one
        # call counted. Tightening this is tracked as a follow-up.
        total = result.stats.get("total") or {}
        self.assertGreaterEqual(total.get("calls", 0), 1,
                                f"expected ≥1 LLM call record, got {total}")
        # Event log should contain the canonical lifecycle markers.
        types = [e.type for e in result.events]
        for marker in ("run_start", "meta_plan", "plan", "join", "run_end"):
            self.assertIn(marker, types, f"missing event {marker!r} in {types}")


if __name__ == "__main__":
    unittest.main()
