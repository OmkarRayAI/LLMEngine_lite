"""Append-only TSV journal for engine runs.

Inspired by karpathy/autoresearch's ``results.tsv`` ledger: one row per run,
human-readable, grep-able, no database. Schema is fixed so existing files keep
working as the engine evolves.

Columns (tab-separated):

    timestamp   duration_s   replans   tokens   status   tag   answer_excerpt
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

HEADER = "timestamp\tduration_s\treplans\ttokens\tstatus\ttag\tanswer_excerpt\n"
_MAX_EXCERPT = 240


def _excerpt(answer: str) -> str:
    if not answer:
        return ""
    one_line = " ".join(answer.split())
    if len(one_line) > _MAX_EXCERPT:
        one_line = one_line[: _MAX_EXCERPT - 1] + "…"
    # TSV: tabs and newlines are illegal inside fields.
    return one_line.replace("\t", " ")


def _total_tokens(stats: Dict[str, Any]) -> int:
    total = stats.get("total") if isinstance(stats, dict) else None
    if not isinstance(total, dict):
        return 0
    return int(total.get("input_tokens", 0)) + int(total.get("output_tokens", 0))


def append_run(
    path: str,
    *,
    duration_s: float,
    replans: int,
    stats: Dict[str, Any],
    status: str,
    answer: str,
    tag: str = "",
    timestamp: Optional[float] = None,
) -> None:
    """Append one row to the TSV at ``path``. Creates the file with a header
    on first write. All exceptions are swallowed — journalling is best-effort
    and must not crash the run.
    """
    try:
        ts = timestamp if timestamp is not None else time.time()
        new_file = not os.path.exists(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if new_file:
                f.write(HEADER)
            row = "\t".join(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                    f"{duration_s:.2f}",
                    str(replans),
                    str(_total_tokens(stats)),
                    status,
                    tag.replace("\t", " "),
                    _excerpt(answer),
                ]
            )
            f.write(row + "\n")
    except Exception:
        # Journal failures are silent by design.
        return
