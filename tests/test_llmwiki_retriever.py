"""Filesystem-backed tests for LLMWikiRetriever."""
import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_wiki(td: str) -> None:
    os.makedirs(os.path.join(td, "wiki"), exist_ok=True)
    os.makedirs(os.path.join(td, "raw"), exist_ok=True)
    open(os.path.join(td, "index.md"), "w").write(textwrap.dedent("""\
        # Index

        - [[wiki/banking-fy25]] — banking sector overview
        - [[wiki/llm-evaluation]] — evaluation methods
    """))
    open(os.path.join(td, "log.md"), "w").write("# Log\n\nThis should be ignored.\n")
    open(os.path.join(td, "wiki", "banking-fy25.md"), "w").write(textwrap.dedent("""\
        ---
        title: Indian Banking FY25
        type: summary
        sources: [raw/icra-fy25.pdf]
        updated: 2026-04-01
        ---

        Indian private sector banks reported strong NIMs in FY25.
        Provisions ticked up in Q4 amid retail unsecured stress.
        Compare against [[wiki/llm-evaluation]] for unrelated context.
    """))
    open(os.path.join(td, "wiki", "llm-evaluation.md"), "w").write(textwrap.dedent("""\
        ---
        title: LLM Evaluation
        type: concept
        ---

        Evaluating large language models on retrieval-augmented generation
        requires faithfulness, answer relevance, and context precision.
    """))
    open(os.path.join(td, "raw", "icra-fy25.pdf").replace(".pdf", ".md"), "w").write(
        "Raw extracted text. Indian banking margins steady. Provisions volatile."
    )


class LLMWikiRetrieverTests(unittest.TestCase):
    def test_returns_wiki_pages_above_raw(self) -> None:
        from LLMEngine import LLMWikiRetriever

        with tempfile.TemporaryDirectory() as td:
            _make_wiki(td)
            r = LLMWikiRetriever(root=td)
            docs = r.retrieve("Indian banking FY25 margins", k=5)
            self.assertTrue(docs, "expected at least one match")
            top = docs[0]
            self.assertEqual(top.source.startswith("wiki/"), True,
                             f"expected wiki/* on top, got {top.source}")
            self.assertIn("Indian", top.text)
            self.assertEqual(top.metadata.get("type"), "summary")

    def test_skips_log_md(self) -> None:
        from LLMEngine import LLMWikiRetriever

        with tempfile.TemporaryDirectory() as td:
            _make_wiki(td)
            r = LLMWikiRetriever(root=td)
            docs = r.retrieve("ignored", k=10)
            for d in docs:
                self.assertNotEqual(d.source, "log.md")

    def test_index_is_ranked(self) -> None:
        from LLMEngine import LLMWikiRetriever

        with tempfile.TemporaryDirectory() as td:
            _make_wiki(td)
            r = LLMWikiRetriever(root=td)
            # The index lists both banking + evaluation entries; querying for
            # both should pull index.md into the top results.
            docs = r.retrieve("banking evaluation", k=5)
            sources = [d.source for d in docs]
            self.assertIn("index.md", sources)

    def test_empty_query_returns_nothing(self) -> None:
        from LLMEngine import LLMWikiRetriever

        with tempfile.TemporaryDirectory() as td:
            _make_wiki(td)
            r = LLMWikiRetriever(root=td)
            self.assertEqual(r.retrieve("the and of"), [])

    def test_missing_root_is_safe(self) -> None:
        from LLMEngine import LLMWikiRetriever

        r = LLMWikiRetriever(root="/no/such/path/here")
        self.assertEqual(r.retrieve("anything"), [])


if __name__ == "__main__":
    unittest.main()
