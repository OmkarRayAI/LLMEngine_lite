"""Public entrypoint to the LLM-Compiler-style engine.

Two ways to call it:

1. ``await engine.run(question="...", tools=[my_tool, ...])`` — keyword form,
   uses sensible defaults from ``default_prompts.py``.
2. ``await engine.run_config(RunConfig(...))`` — typed config object for
   callers that want explicit control of every knob.

Both return a :class:`~LLMEngine.config.RunResult`. Tuple unpacking is still
supported for backwards compatibility with the previous
``answer, [], thinking_process`` return shape.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Union

from .config import RunConfig, RunResult, ToolLike
from .default_prompts import DEFAULT_JOINNER_PROMPT, DEFAULT_PLANNER_PROMPT
from .events import EventLog
from .llm_compiler import LLMCompiler
from .logger_utils import logger
from .tools.tool_generator import ToolGenerator


class LLMEngine:
    def __init__(
        self,
        llm: Any,
        message_manager: Optional[Any] = None,
        memory: Optional[List[Any]] = None,
        max_replan: int = 3,
        verbose: bool = False,
    ) -> None:
        self.message_manager = message_manager
        self.llm = llm
        self.memory = memory if memory is not None else []
        self.max_replan = max_replan
        self.verbose = verbose
        self.is_tables = False
        self.last_run: Optional[RunResult] = None
        self._build_compiler()

    def _build_compiler(self, event_log: Optional[EventLog] = None) -> None:
        self.event_log = event_log or EventLog()
        self.agent = LLMCompiler(
            name="LLMCompiler",
            llm=self.llm,
            max_replans=self.max_replan,
            benchmark=False,
            message_manager=self.message_manager,
            event_log=self.event_log,
            collect_stats=True,
        )

    @staticmethod
    def _resolve_tools(
        tools: Sequence[ToolLike], tool_path: Optional[str]
    ) -> List[Any]:
        """Accept either pre-built tool instances or the legacy dict form."""
        if not tools:
            return []
        # If every entry is a dict, route through the dynamic generator.
        if all(isinstance(t, dict) for t in tools):
            return ToolGenerator(list(tools), tool_path)
        # If any entry is already a tool-like object (has ``name`` and
        # ``description``), accept the list verbatim. Mixed lists are also
        # supported by splitting and concatenating.
        instances: List[Any] = []
        dict_specs: List[dict] = []
        for t in tools:
            if isinstance(t, dict):
                dict_specs.append(t)
            else:
                instances.append(t)
        if dict_specs:
            instances.extend(ToolGenerator(dict_specs, tool_path))
        return instances

    async def arun_and_time(self, func, *args, **kwargs):
        """Run ``func`` and return ``(result, elapsed_seconds)``.

        Exceptions are logged via the namespaced logger and re-raised so the
        caller can react. Previously this used ``print`` which made errors
        invisible to log shippers.
        """
        start = time.time()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            logger.exception("LLMEngine run failed")
            raise
        return result, time.time() - start

    async def run_config(self, cfg: RunConfig) -> RunResult:
        """Run with a typed :class:`RunConfig`. Returns :class:`RunResult`."""
        # Each run gets a fresh event log so stale events don't leak across
        # calls when the engine is reused.
        self._build_compiler()

        resolved_tools = self._resolve_tools(cfg.tools, cfg.tool_path)
        if not resolved_tools:
            logger.warning("LLMEngine.run was called with no tools resolved.")
        logger.info("LLMEngine using %d tool(s)", len(resolved_tools))

        input_dict = {
            "input": cfg.question,
            "is_table_format": cfg.is_table_format,
            "planner_example_prompt": cfg.planner_example_prompt or DEFAULT_PLANNER_PROMPT,
            "planner_example_prompt_replan": None,
            "joinner_prompt": cfg.joinner_prompt or DEFAULT_JOINNER_PROMPT,
            "joinner_prompt_final": None,
            "purpose": cfg.purpose,
            "instructions": cfg.instructions,
            "query_understanding": cfg.query_understanding,
            "temporal_context": cfg.temporal_context,
            "research_approach": cfg.research_approach,
            "dos": cfg.dos,
            "donts": cfg.donts,
            "meta_example": cfg.meta_example,
            "tools": resolved_tools,
        }

        result, elapsed = await self.arun_and_time(self.agent.acall, input_dict, callbacks=None)

        if isinstance(result, dict):
            answer = result.get(self.agent.output_key, "")
            thinking = result.get("thinking_process", "")
            stats = result.get("stats", {}) or self.agent.get_all_stats()
            tasks = result.get("tasks", {})
            replans = int(result.get("replans", 0))
            events = result.get("events", self.event_log.events)
        else:
            answer = str(result)
            thinking = ""
            stats = self.agent.get_all_stats()
            tasks = {}
            replans = 0
            events = self.event_log.events

        meta_plan_evt = next(
            (e for e in events if getattr(e, "type", None) == "meta_plan"), None
        )
        meta_plan = meta_plan_evt.payload.get("meta_plan", "") if meta_plan_evt else ""

        rr = RunResult(
            answer=answer,
            thinking_process=thinking,
            meta_plan=meta_plan,
            events=events,
            stats=stats,
            tasks=tasks,
            replans=replans,
            duration_s=elapsed,
        )
        self.last_run = rr
        return rr

    async def run(
        self,
        question: str = "",
        purpose: str = "",
        tools: Sequence[ToolLike] = (),
        instructions: str = "",
        query_understanding: str = "",
        temporal_context: str = "",
        research_approach: str = "",
        dos: str = "",
        donts: str = "",
        meta_example: str = "",
        planner_example_prompt: str = "",
        joinner_prompt: str = "",
        tool_path: Optional[str] = None,
    ) -> RunResult:
        """Keyword-arg entrypoint. Returns :class:`RunResult`.

        Backwards compatible: ``answer, sources, thinking = await engine.run(...)``
        still works because ``RunResult`` is iterable.
        """
        cfg = RunConfig(
            question=question,
            purpose=purpose,
            instructions=instructions,
            tools=list(tools),
            tool_path=tool_path,
            planner_example_prompt=planner_example_prompt,
            joinner_prompt=joinner_prompt,
            query_understanding=query_understanding,
            temporal_context=temporal_context,
            research_approach=research_approach,
            dos=dos,
            donts=donts,
            meta_example=meta_example,
            is_table_format=self.is_tables,
        )
        return await self.run_config(cfg)
