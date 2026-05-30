"""Retriever protocol and built-in implementations.

A ``Retriever`` returns ``Doc`` snippets for a query string. Anything that
implements ``retrieve(query, k)`` can plug in: vector stores, BM25 indexes,
local markdown trees, or external services. Wrap one in ``KnowledgeTool`` to
expose it to the LLMEngine planner as a regular tool.
"""
from .base import Doc, Retriever
from .knowledge_tool import KnowledgeTool
from .llmwiki import LLMWikiRetriever

__all__ = [
    "Doc",
    "Retriever",
    "KnowledgeTool",
    "LLMWikiRetriever",
]
