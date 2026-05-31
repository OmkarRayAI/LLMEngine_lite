"""Wrap any ``Retriever`` as a ``StructuredTool`` for the planner.

The engine plans against tool descriptions, so this builds a clean,
self-describing tool whose call signature the planner already knows how to
emit. The observation is a numbered, source-cited block — exactly what the
joiner expects.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..base import StructuredTool
from .base import Doc, Retriever, format_docs_for_prompt


class _RetrieveInput(BaseModel):
    query: str = Field(..., description="What to look up in the knowledge base.")
    k: int = Field(5, description="How many documents to return (1-20).")


class KnowledgeTool:
    """Adapter that turns a ``Retriever`` into a ``StructuredTool``.

    Usage::

        wiki = LLMWikiRetriever(root="./my-wiki")
        kt = KnowledgeTool(wiki, name="wiki", description="Curated team wiki…")
        result = await engine.run(question="…", tools=[kt.get_tool()])
    """

    def __init__(
        self,
        retriever: Retriever,
        name: str = "knowledge",
        description: Optional[str] = None,
        max_chars: int = 4000,
    ) -> None:
        self.retriever = retriever
        self.name = name
        self.max_chars = max_chars
        self.description = description or (
            f"{name}(query: str, k: int = 5) -> str:\n"
            " - Searches a curated knowledge base and returns the top-k\n"
            "   matching documents with their source paths.\n"
            " - Use this BEFORE answering questions whose facts may live in the\n"
            "   knowledge base. Cite returned `source=` paths in the final answer."
        )

    async def _aretrieve(self, query: str, k: int = 5) -> str:
        # Clamp k inline rather than via pydantic validators so a planner
        # over-request just degrades gracefully instead of erroring out.
        # Real LLMs occasionally hand non-int values; coerce permissively
        # and fall back on the default, never raise from retrieval.
        try:
            k = max(1, min(int(k or 5), 20))
        except (TypeError, ValueError):
            k = 5
        # Support both sync and async retrievers.
        aretrieve = getattr(self.retriever, "aretrieve", None)
        if aretrieve is not None:
            docs = await aretrieve(query, k=k)
        else:
            docs = await asyncio.to_thread(self.retriever.retrieve, query, k=k)  # type: ignore[attr-defined]
        return format_docs_for_prompt(docs, max_chars=self.max_chars)

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            coroutine=self._aretrieve,
            name=self.name,
            description=self.description,
            args_schema=_RetrieveInput,
        )

    # Convenience: KnowledgeTool can be passed directly as a tool — the
    # engine calls .get_tool() if available, but most callers will pass the
    # tool explicitly. This makes both work.
    def __iter__(self):  # so list(kt) returns [tool], for ergonomic spreads
        yield self.get_tool()
