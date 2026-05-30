"""Scenario 4: tool failure surfaced via the joiner triggers a replan.

The engine doesn't replan on tool *exceptions* — exceptions abort. It
replans when the *joiner* says ``Action: Replan(...)``. So this scenario
chains:

  iter 1: plan → call brittle("...")  (returns an error string)
          join → ``Replan(brittle errored, try search instead)``
  iter 2: plan → call search("...")
          join → ``Finish(...)``

We assert the run records exactly one replan and the final answer comes
from the second iteration's tool. This exercises the §1 fix that resolved
the ``planner_example_prompt_replan`` shadow bug — without that fix, this
scenario would hit the wrong prompt or NameError.
"""
import unittest

from .helpers import ScriptedChatModel, flaky_tool, search_tool


META = (
    "META PLAN:\n"
    "Try the brittle tool first; if it errors, replan with search."
)

PLAN_1 = """\
Thought: Try the primary tool.
1. brittle("hello")
2. join()
<END_OF_PLAN>"""

JOIN_1 = """\
Thought: Brittle returned an error string; try search instead.
Action: Replan(brittle returned an error; fall back to search)"""

PLAN_2 = """\
Thought: Use search as a fallback this time.
1. search("hello world")
2. join()
<END_OF_PLAN>"""

JOIN_2 = """\
Thought: Search succeeded.
Action: Finish(hello world via search)"""


class ReplanScenario(unittest.IsolatedAsyncioTestCase):
    async def test_replan_after_tool_error_string(self) -> None:
        from LLMEngine import LLMEngine

        # The flaky tool returns an error string the first call; we won't
        # actually call it a second time (replan switches to search), so the
        # second-arm value is unused.
        brittle = flaky_tool(
            first_returns="ERROR: upstream timeout",
            then_returns="(unreachable)",
        )
        search = search_tool(returns="hello world")

        llm = ScriptedChatModel([META, PLAN_1, JOIN_1, PLAN_2, JOIN_2])
        engine = LLMEngine(llm=llm, max_replan=3)

        result = await engine.run(
            question="Greet me",
            tools=[brittle, search],
        )

        self.assertEqual(result.replans, 1, f"expected 1 replan, got {result.replans}")
        self.assertIn("hello world via search", result.answer)
        # The events log should contain a replan event.
        types = [e.type for e in result.events]
        self.assertEqual(types.count("plan"), 2)
        self.assertEqual(types.count("replan"), 1)


if __name__ == "__main__":
    unittest.main()
