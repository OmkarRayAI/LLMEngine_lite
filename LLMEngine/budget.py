"""Run-budget tripwires.

A ``Budget`` tracks wall-clock time and total tokens against optional caps.
The engine consults it at iteration boundaries (between replans, after the
joiner) and aborts cleanly with ``status="budget_exceeded"`` rather than being
killed mid-tool.

This is the small slice of karpathy/autoresearch that ports cleanly to a
request/response engine: a fixed time budget makes runs bounded and
comparable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Budget:
    max_seconds: Optional[float] = None
    max_total_tokens: Optional[int] = None
    _started_at: float = 0.0

    def start(self) -> None:
        self._started_at = time.time()

    def elapsed(self) -> float:
        return time.time() - self._started_at if self._started_at else 0.0

    def time_exceeded(self) -> bool:
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds

    def tokens_exceeded(self, stats: Dict[str, Any]) -> bool:
        if self.max_total_tokens is None:
            return False
        total = stats.get("total") if isinstance(stats, dict) else None
        if not isinstance(total, dict):
            return False
        used = int(total.get("input_tokens", 0)) + int(total.get("output_tokens", 0))
        return used >= self.max_total_tokens

    def exceeded(self, stats: Dict[str, Any]) -> bool:
        return self.time_exceeded() or self.tokens_exceeded(stats)

    def reason(self, stats: Dict[str, Any]) -> str:
        if self.time_exceeded():
            return f"max_seconds={self.max_seconds} reached (elapsed={self.elapsed():.1f}s)"
        if self.tokens_exceeded(stats):
            return f"max_total_tokens={self.max_total_tokens} reached"
        return ""
