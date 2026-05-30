"""Structured run events emitted by the LLMEngine.

Events let callers stream progress (UIs, websockets, log shippers) and inspect
runs after the fact via ``RunResult.events`` without parsing stdout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

EventType = Literal[
    "run_start",
    "meta_plan",
    "plan",
    "task_start",
    "task_end",
    "task_error",
    "join",
    "replan",
    "run_end",
]


@dataclass
class RunEvent:
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "ts": self.ts, "payload": self.payload}


class EventLog:
    """Append-only event buffer with optional async fan-out subscribers."""

    def __init__(self) -> None:
        self._events: List[RunEvent] = []
        self._subscribers: List[Any] = []  # asyncio.Queue-like

    def emit(self, type: EventType, **payload: Any) -> RunEvent:
        evt = RunEvent(type=type, payload=payload)
        self._events.append(evt)
        for q in self._subscribers:
            try:
                q.put_nowait(evt)
            except Exception:
                pass
        return evt

    def subscribe(self, queue: Any) -> None:
        self._subscribers.append(queue)

    @property
    def events(self) -> List[RunEvent]:
        return list(self._events)

    def by_type(self, type: EventType) -> List[RunEvent]:
        return [e for e in self._events if e.type == type]
