"""Unit tests for the Budget tripwire."""
import os
import sys
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class BudgetTests(unittest.TestCase):
    def test_no_caps_never_exceeds(self) -> None:
        from LLMEngine.budget import Budget

        b = Budget()
        b.start()
        self.assertFalse(b.exceeded({}))
        self.assertFalse(b.time_exceeded())
        self.assertFalse(b.tokens_exceeded({}))

    def test_time_cap(self) -> None:
        from LLMEngine.budget import Budget

        b = Budget(max_seconds=0.05)
        b.start()
        self.assertFalse(b.time_exceeded())
        time.sleep(0.06)
        self.assertTrue(b.time_exceeded())
        self.assertIn("max_seconds", b.reason({}))

    def test_token_cap(self) -> None:
        from LLMEngine.budget import Budget

        b = Budget(max_total_tokens=100)
        b.start()
        stats_under = {"total": {"input_tokens": 30, "output_tokens": 50}}
        stats_over = {"total": {"input_tokens": 60, "output_tokens": 50}}
        self.assertFalse(b.tokens_exceeded(stats_under))
        self.assertTrue(b.tokens_exceeded(stats_over))
        self.assertIn("max_total_tokens", b.reason(stats_over))


if __name__ == "__main__":
    unittest.main()
