"""Smoke tests for ``RunConfig`` / ``RunResult`` and ``EventLog``."""
import os
import sys
import unittest

# Allow ``import LLMEngine`` from the repo root without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ConfigTests(unittest.TestCase):
    def test_run_result_is_iterable_for_legacy_callers(self) -> None:
        from LLMEngine.config import RunResult

        rr = RunResult(answer="hi", thinking_process="t")
        answer, sources, thinking = rr  # legacy tuple unpacking
        self.assertEqual(answer, "hi")
        self.assertEqual(sources, [])
        self.assertEqual(thinking, "t")

    def test_run_config_defaults(self) -> None:
        from LLMEngine.config import RunConfig

        cfg = RunConfig(question="q")
        self.assertEqual(cfg.tools, [])
        self.assertIsNone(cfg.tool_path)
        self.assertEqual(cfg.purpose, "")

    def test_event_log_emit_and_filter(self) -> None:
        from LLMEngine.events import EventLog

        log = EventLog()
        log.emit("run_start", question="q")
        log.emit("plan", num_tasks=3)
        log.emit("run_end", answer="ok")

        self.assertEqual(len(log.events), 3)
        plan_events = log.by_type("plan")
        self.assertEqual(len(plan_events), 1)
        self.assertEqual(plan_events[0].payload["num_tasks"], 3)


if __name__ == "__main__":
    unittest.main()
