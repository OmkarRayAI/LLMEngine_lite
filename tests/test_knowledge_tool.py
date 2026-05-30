"""Tests for the KnowledgeTool wrapper."""
import asyncio
import os
import sys
import unittest
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FakeSyncRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query: str, k: int = 5):
        return self._docs[:k]


class FakeAsyncRetriever:
    def __init__(self, docs):
        self._docs = docs

    async def aretrieve(self, query: str, k: int = 5):
        return self._docs[:k]


class KnowledgeToolTests(unittest.TestCase):
    def _docs(self):
        from LLMEngine import Doc
        return [
            Doc(text="alpha summary", source="wiki/alpha.md", score=2.1),
            Doc(text="beta summary", source="wiki/beta.md", score=1.4),
        ]

    def test_async_retriever_path(self) -> None:
        from LLMEngine import KnowledgeTool

        kt = KnowledgeTool(FakeAsyncRetriever(self._docs()), name="kb")
        tool = kt.get_tool()
        self.assertEqual(tool.name, "kb")
        out = asyncio.run(tool.coroutine(query="anything", k=2))
        self.assertIn("wiki/alpha.md", out)
        self.assertIn("wiki/beta.md", out)

    def test_sync_retriever_path(self) -> None:
        from LLMEngine import KnowledgeTool

        kt = KnowledgeTool(FakeSyncRetriever(self._docs()), name="kb")
        tool = kt.get_tool()
        out = asyncio.run(tool.coroutine(query="anything", k=1))
        self.assertIn("wiki/alpha.md", out)
        self.assertNotIn("wiki/beta.md", out)

    def test_k_is_clamped(self) -> None:
        from LLMEngine import KnowledgeTool

        kt = KnowledgeTool(FakeSyncRetriever(self._docs()), name="kb")
        tool = kt.get_tool()
        out = asyncio.run(tool.coroutine(query="x", k=999))
        # Should not blow up; both docs returned.
        self.assertIn("wiki/alpha.md", out)


if __name__ == "__main__":
    unittest.main()
