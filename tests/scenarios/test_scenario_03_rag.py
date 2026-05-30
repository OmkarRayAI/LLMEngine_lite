"""Scenario 3: RAG over an LLMWiki knowledge base.

Builds a tiny on-disk wiki, exposes it via ``LLMWikiRetriever`` +
``KnowledgeTool``, scripts the LLM to call ``wiki(query)`` and finish with
a citation. Proves the retrieval-tool wiring works end-to-end with the
engine, not just in unit tests.
"""
import os
import tempfile
import textwrap
import unittest

from .helpers import ScriptedChatModel


META = (
    "META PLAN:\n"
    "Search the wiki for the user's question, then synthesize a cited answer."
)

PLAN = """\
Thought: Look up the wiki for an authoritative summary.
1. wiki("Indian banking FY25 NIM provisions")
2. join()
<END_OF_PLAN>"""

JOIN = """\
Thought: The wiki returned a curated summary; cite the source.
Action: Finish(Indian private banks reported strong NIMs in FY25; provisions ticked up in Q4. [source: wiki/banking-fy25.md])"""


def _make_wiki(td: str) -> None:
    os.makedirs(os.path.join(td, "wiki"), exist_ok=True)
    open(os.path.join(td, "index.md"), "w").write(
        "# Index\n- [[wiki/banking-fy25]] — banking sector overview\n"
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
    """))


class RagScenario(unittest.IsolatedAsyncioTestCase):
    async def test_rag_with_llmwiki_retriever(self) -> None:
        from LLMEngine import KnowledgeTool, LLMEngine, LLMWikiRetriever

        with tempfile.TemporaryDirectory() as td:
            _make_wiki(td)
            retriever = LLMWikiRetriever(root=td)
            kt = KnowledgeTool(
                retriever,
                name="wiki",
                description="wiki(query: str, k: int = 5) -> str:\n - Search the team wiki.",
            )

            llm = ScriptedChatModel([META, PLAN, JOIN])
            engine = LLMEngine(llm=llm, max_replan=1)

            result = await engine.run(
                question="What did Indian private banks report in FY25?",
                tools=[kt.get_tool()],
            )

        # The answer cites a wiki source.
        self.assertIn("banking-fy25.md", result.answer)
        # The plan executed exactly one tool task plus join.
        non_join = [t for t in result.tasks.values() if not t.is_join]
        self.assertEqual(len(non_join), 1)
        self.assertEqual(non_join[0].name, "wiki")
        # The wiki tool's observation contains the snippet text the
        # retriever pulled — proves the retriever ran, not just the stub.
        self.assertIn("NIMs", non_join[0].observation)
        self.assertIn("source=wiki/banking-fy25.md", non_join[0].observation)


if __name__ == "__main__":
    unittest.main()
