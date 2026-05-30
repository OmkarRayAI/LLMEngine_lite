"""Unit tests for the run journal TSV writer."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class JournalTests(unittest.TestCase):
    def test_append_creates_header_then_row(self) -> None:
        from LLMEngine.journal import HEADER, append_run

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "runs.tsv")
            stats = {"total": {"input_tokens": 100, "output_tokens": 50}}
            append_run(
                path,
                duration_s=1.234,
                replans=2,
                stats=stats,
                status="ok",
                answer="hello world",
                tag="exp1",
            )
            content = open(path).read()
            self.assertTrue(content.startswith(HEADER))
            lines = content.strip().splitlines()
            self.assertEqual(len(lines), 2)
            cols = lines[1].split("\t")
            self.assertEqual(cols[1], "1.23")
            self.assertEqual(cols[2], "2")
            self.assertEqual(cols[3], "150")
            self.assertEqual(cols[4], "ok")
            self.assertEqual(cols[5], "exp1")
            self.assertEqual(cols[6], "hello world")

    def test_two_appends_share_header(self) -> None:
        from LLMEngine.journal import append_run

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "runs.tsv")
            append_run(path, duration_s=0.1, replans=0, stats={}, status="ok", answer="a")
            append_run(path, duration_s=0.2, replans=1, stats={}, status="ok", answer="b")
            lines = open(path).read().strip().splitlines()
            # 1 header + 2 rows
            self.assertEqual(len(lines), 3)

    def test_strips_tabs_and_newlines_from_excerpt(self) -> None:
        from LLMEngine.journal import append_run

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "runs.tsv")
            append_run(
                path,
                duration_s=0.0,
                replans=0,
                stats={},
                status="ok",
                answer="line1\nline2\twith\ttabs",
            )
            row = open(path).read().strip().splitlines()[-1]
            cols = row.split("\t")
            # Excerpt column must contain no embedded tabs/newlines.
            self.assertEqual(len(cols), 7)
            self.assertNotIn("\n", cols[6])

    def test_journal_failures_are_silent(self) -> None:
        from LLMEngine.journal import append_run

        # Non-existent parent that we cannot create (use a path under a file).
        with tempfile.NamedTemporaryFile() as tmp:
            bad_path = os.path.join(tmp.name, "subdir", "runs.tsv")
            # Should not raise even though the parent is a file, not a dir.
            append_run(bad_path, duration_s=0, replans=0, stats={}, status="ok", answer="x")


if __name__ == "__main__":
    unittest.main()
