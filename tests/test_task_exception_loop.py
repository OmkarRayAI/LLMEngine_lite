"""Regression test for the infinite-reschedule loop on tool exceptions.

Before the fix, a tool that raised left ``tasks_done[idx]`` un-set, so
``TaskFetchingUnit.schedule()`` looped forever — the real-LLM smoke
exposed this with multiple parallel wiki calls all failing.

After the fix:
- tasks_done is set in a ``finally`` block so the scheduler can exit.
- The exception is captured as an error-string observation so the joiner
  can read it and decide to replan.
"""
import asyncio
import os
import sys
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TaskExceptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_raising_tool_does_not_hang_scheduler(self) -> None:
        from LLMEngine.task_fetching_unit import Task, TaskFetchingUnit

        async def boom(*a, **kw):
            raise RuntimeError("upstream 503")

        unit = TaskFetchingUnit()
        unit.set_tasks({
            1: Task(idx=1, name="boom", tool=boom, args=(), dependencies=[]),
            2: Task(idx=2, name="join", tool=lambda x: None, args=(),
                    dependencies=[1], is_join=True),
        })
        # Without the fix, this would hang forever.
        await asyncio.wait_for(unit.schedule(), timeout=2.0)

        # The error must be visible as an observation, not a silent crash.
        self.assertIn("ERROR", unit.tasks[1].observation or "")
        self.assertIn("upstream 503", unit.tasks[1].observation or "")

    async def test_kwargs_are_forwarded_to_tool(self) -> None:
        # Pin the wiring: planner-emitted kwargs must reach the tool.
        from LLMEngine.task_fetching_unit import Task, TaskFetchingUnit

        seen = {}

        async def capture(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return "ok"

        unit = TaskFetchingUnit()
        unit.set_tasks({
            1: Task(idx=1, name="capture", tool=capture,
                    args=("hello",), kwargs={"k": 3}, dependencies=[]),
        })
        await asyncio.wait_for(unit.schedule(), timeout=1.0)
        self.assertEqual(seen["args"], ("hello",))
        self.assertEqual(seen["kwargs"], {"k": 3})


if __name__ == "__main__":
    unittest.main()
