"""Run configuration and result types for ``LLMEngine.run``.

Replaces a 13-positional-arg call site with a typed config object and a
structured return value so callers can introspect plan/tasks/stats without
relying on stdout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .base import StructuredTool, Tool

ToolLike = Union[Dict[str, Any], "StructuredTool", "Tool", Any]


@dataclass
class RunConfig:
    """Configuration for a single ``LLMEngine.run`` call.

    ``tools`` accepts either:

    - the legacy list-of-dict form (with ``class``/``name``/...) — still
      requires ``tool_path`` so the dynamic loader can find the classes.
    - a list of already-instantiated ``StructuredTool`` / ``Tool`` objects
      (or anything with ``.name`` and ``.description``).
    """

    question: str
    purpose: str = ""
    instructions: str = ""
    tools: Sequence[ToolLike] = field(default_factory=list)
    tool_path: Optional[str] = None

    # Optional planner/joiner overrides — sensible defaults are used when empty.
    planner_example_prompt: str = ""
    joinner_prompt: str = ""

    # Meta-planner knobs (all optional; sensible defaults in defaults.py).
    query_understanding: str = ""
    temporal_context: str = ""
    research_approach: str = ""
    dos: str = ""
    donts: str = ""
    meta_example: str = ""

    is_table_format: bool = False


@dataclass
class RunResult:
    """Structured result of a single engine run."""

    answer: str
    thinking_process: str = ""
    meta_plan: str = ""
    events: List[Any] = field(default_factory=list)  # List[RunEvent]
    stats: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[Any, Any] = field(default_factory=dict)
    replans: int = 0
    duration_s: float = 0.0

    # Backwards-compatible tuple unpacking: ``answer, _, thinking = engine.run(...)``
    def __iter__(self):
        yield self.answer
        yield []
        yield self.thinking_process
