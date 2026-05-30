"""Retriever protocol and ``Doc`` value type.

Deliberately minimal: a query string in, a list of ``Doc`` out. No assumptions
about embeddings, chunking, or storage. The engine only ever sees strings, so
the ``Doc`` shape exists for callers who want to inspect provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class Doc:
    """A single retrieved chunk."""

    text: str
    source: str = ""              # path, URL, or other identifier
    score: float = 0.0            # retriever-specific relevance score
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cite(self) -> str:
        """Render a one-line citation for use inside an LLM prompt."""
        if not self.source:
            return self.text
        return f"[{self.source}] {self.text}"


@runtime_checkable
class Retriever(Protocol):
    """Contract for anything that can answer ``retrieve(query, k)``.

    Implementations may be sync or async — ``KnowledgeTool`` calls a
    coroutine, so async implementations are preferred. A sync implementation
    can be wrapped via ``asyncio.to_thread``.
    """

    async def aretrieve(self, query: str, k: int = 5) -> List[Doc]:  # pragma: no cover - protocol
        ...


def format_docs_for_prompt(docs: List[Doc], max_chars: int = 4000) -> str:
    """Serialize a list of ``Doc`` for inclusion in a tool observation.

    Truncates from the tail to keep the planner's scratchpad bounded —
    long retrievals would otherwise blow up subsequent prompts.
    """
    if not docs:
        return "No matching documents."
    lines: List[str] = []
    used = 0
    for i, d in enumerate(docs, 1):
        block = f"[{i}] source={d.source} score={d.score:.3f}\n{d.text}\n"
        if used + len(block) > max_chars:
            lines.append(f"… ({len(docs) - i + 1} more documents truncated)")
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)
