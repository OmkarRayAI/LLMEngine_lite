"""Retriever for an LLM-maintained wiki on disk.

Implements the pattern from
https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f — the wiki
is a directory of curated, interlinked markdown files. The retriever:

- prefers ``wiki/`` pages (curated synthesis) over ``raw/`` (raw sources)
- treats ``index.md`` as a catalog hint (boost when matched)
- parses YAML-ish frontmatter for ``title``, ``type``, ``sources``, ``updated``
- scores with a small inline BM25 implementation — no extra deps

The point is to retrieve from durable, LLM-curated knowledge, not chunks of
raw PDFs. This is the direct alternative to embedding-based RAG at the
moderate scale (~hundreds of pages) where Karpathy argues a wiki wins.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .base import Doc

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)", flags=re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Common English stopwords; keeping the list small avoids a corpus dep.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have he her his i if in is it its
    of on or our she that the their them they this to was we were what which
    who will with you your
    """.split()
)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def _parse_frontmatter(text: str) -> tuple[Dict[str, str], str]:
    """Return ``(frontmatter_dict, body)`` for a markdown file.

    Frontmatter is the YAML-ish ``---`` block at the top. We don't pull a YAML
    parser in — we only need flat ``key: value`` lines.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_meta, body = m.group(1), m.group(2)
    meta: Dict[str, str] = {}
    for line in raw_meta.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, body


@dataclass
class _Page:
    path: str           # relative to root
    title: str
    body: str
    tokens: List[str]
    section: str        # "wiki" | "raw" | "index" | "other"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMWikiRetriever:
    """Retrieve from a directory laid out per the LLMWiki schema.

    Args:
        root: Directory containing ``wiki/``, optionally ``raw/``, ``index.md``,
            ``log.md``, ``AGENTS.md``.
        include_raw: If True, ``raw/`` files are searched too (lower priority).
        index_boost: Multiplier applied to ``index.md`` matches.
        wiki_boost: Multiplier applied to ``wiki/*`` matches.
        skip_log: If True, ``log.md`` is skipped (it's append-only and noisy).
    """

    root: str
    include_raw: bool = True
    # Boosts encode the LLMWiki schema's "prefer curated wiki over raw sources"
    # rule. Tunable, but the defaults are intentional — see Karpathy's gist.
    index_boost: float = 1.5
    wiki_boost: float = 2.0
    skip_log: bool = True
    _pages: List[_Page] = field(default_factory=list, init=False)
    _df: Counter = field(default_factory=Counter, init=False)
    _avg_len: float = field(default=0.0, init=False)
    _loaded: bool = field(default=False, init=False)

    # ---- public API -------------------------------------------------------

    async def aretrieve(self, query: str, k: int = 5) -> List[Doc]:
        # Sync today; the protocol is async to leave room for an async backend
        # without breaking callers.
        return self.retrieve(query, k=k)

    def retrieve(self, query: str, k: int = 5) -> List[Doc]:
        if not self._loaded:
            self._load()
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: List[tuple[float, _Page]] = []
        for page in self._pages:
            s = self._bm25(q_tokens, page) * self._section_boost(page.section)
            if s > 0:
                scored.append((s, page))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._to_doc(score, page) for score, page in scored[:k]]

    def reload(self) -> None:
        """Re-read the wiki from disk. Useful if the wiki is mutated mid-run."""
        self._loaded = False
        self._pages.clear()
        self._df.clear()
        self._avg_len = 0.0
        self._load()

    # ---- internals --------------------------------------------------------

    def _load(self) -> None:
        root = Path(self.root)
        if not root.exists():
            self._loaded = True
            return
        for md_path in self._iter_markdown(root):
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = _parse_frontmatter(text)
            rel = str(md_path.relative_to(root))
            section = self._classify(rel)
            title = meta.get("title") or md_path.stem.replace("-", " ").title()
            tokens = _tokenize(title + "\n" + body)
            page = _Page(
                path=rel, title=title, body=body, tokens=tokens,
                section=section, metadata=meta,
            )
            self._pages.append(page)
            for t in set(tokens):
                self._df[t] += 1
        if self._pages:
            self._avg_len = sum(len(p.tokens) for p in self._pages) / len(self._pages)
        self._loaded = True

    def _iter_markdown(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*.md"):
            rel = path.relative_to(root).as_posix()
            if self.skip_log and rel == "log.md":
                continue
            if not self.include_raw and rel.startswith("raw/"):
                continue
            yield path

    def _classify(self, rel_path: str) -> str:
        if rel_path == "index.md":
            return "index"
        if rel_path.startswith("wiki/"):
            return "wiki"
        if rel_path.startswith("raw/"):
            return "raw"
        return "other"

    def _section_boost(self, section: str) -> float:
        if section == "index":
            return self.index_boost
        if section == "wiki":
            return self.wiki_boost
        return 1.0

    def _bm25(self, q_tokens: List[str], page: _Page, k1: float = 1.5, b: float = 0.75) -> float:
        if not page.tokens:
            return 0.0
        n = len(self._pages)
        tf = Counter(page.tokens)
        score = 0.0
        for q in q_tokens:
            df = self._df.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            f = tf.get(q, 0)
            denom = f + k1 * (1 - b + b * (len(page.tokens) / max(self._avg_len, 1)))
            score += idf * (f * (k1 + 1) / denom) if denom else 0.0
        return score

    def _to_doc(self, score: float, page: _Page) -> Doc:
        snippet = self._snippet(page.body)
        meta = {
            "title": page.title,
            "section": page.section,
            "links": _LINK_RE.findall(page.body),
            **page.metadata,
        }
        return Doc(text=snippet, source=page.path, score=score, metadata=meta)

    @staticmethod
    def _snippet(body: str, max_chars: int = 600) -> str:
        body = body.strip()
        if len(body) <= max_chars:
            return body
        return body[: max_chars - 1] + "…"
