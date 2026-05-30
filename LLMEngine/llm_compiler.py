import asyncio
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.llms.base import BaseLLM
from langchain.prompts.base import StringPromptValue

from .budget import Budget
from .callbacks import AsyncStatsCallbackHandler
from .chain import Chain
from .constants import JOINNER_REPLAN, JOINNER_FINISH
from .events import EventLog
from .planner import Planner
from .metaplanner import MetaPlanner
from .task_fetching_unit import Task, TaskFetchingUnit
from .base import StructuredTool, Tool
from .logger_utils import log, log_task_execution, logger
from .prompts import NO_ANWER_REPLY, TABLE_OUTPUT_PROMPT
from .constants import END_OF_PLAN

class LLMCompilerAgent:
    """Self defined agent for LLM Compiler."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def arun(self, prompt: str, callbacks=None) -> str:
        response = await self.llm.agenerate_prompt(
            prompts=[StringPromptValue(text=prompt)],
            stop=["<END_OF_RESPONSE>"],
            callbacks=callbacks,
        )
        if isinstance(self.llm, BaseChatModel):
            return response.generations[0][0].message.content

        if isinstance(self.llm, BaseLLM):
            return response.generations[0][0].text

        raise ValueError("LLM must be either BaseChatModel or BaseLLM")


class LLMCompiler(Chain, extra="allow"):
    """LLMCompiler Engine."""

    """The step container to use."""
    input_key: str = "input"
    output_key: str = "output"

    def __init__(
        self,
        llm: BaseLLM,
        max_replans: int,
        benchmark: bool,
        message_manager,
        event_log: Optional[EventLog] = None,
        collect_stats: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            max_replans: Maximum number of replans to do.
            benchmark: Whether to expose stats via ``get_all_stats``.
            event_log: Optional shared ``EventLog`` for structured progress events.
            collect_stats: Always wire up token/timing callbacks (cheap; previously
                gated behind ``benchmark`` and silently disabled).
        """
        super().__init__(**kwargs)


        self.meta_planner = MetaPlanner()
        self.planner = Planner()
        self.llm = llm
        self.agent = LLMCompilerAgent(llm)

        self.planner_stream = False
        self.max_replans = max_replans
        self.message_manager = message_manager
        self.event_log = event_log or EventLog()

        # Callbacks: wire stats handlers by default so observability is on
        # without requiring a benchmark flag at the call site.
        self.benchmark = bool(benchmark)
        if collect_stats or self.benchmark:
            self.planner_callback = AsyncStatsCallbackHandler(stream=False)
            self.executor_callback = AsyncStatsCallbackHandler(stream=False)
        else:
            self.planner_callback = None
            self.executor_callback = None
            
    def get(self, key):
        return getattr(self, key, None)

    def get_all_stats(self):
        stats: Dict[str, Any] = {}
        if self.planner_callback is None or self.executor_callback is None:
            return stats
        stats["planner"] = self.planner_callback.get_stats()
        stats["executor"] = self.executor_callback.get_stats()
        # Sum only numeric scalar fields so we don't try to add the
        # ``all_times`` lists element-wise.
        total: Dict[str, Any] = {}
        for k, v in stats["planner"].items():
            other = stats["executor"].get(k)
            if isinstance(v, (int, float)) and isinstance(other, (int, float)):
                total[k] = v + other
        stats["total"] = total
        return stats

    def reset_all_stats(self):
        if self.planner_callback:
            self.planner_callback.reset()
        if self.executor_callback:
            self.executor_callback.reset()

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    # TODO(sk): move all join related functions to a separate class

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Models occasionally wrap the joiner output in ``` fences. Strip them."""
        m = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\s*([\s\S]*?)```", text)
        return m.group(1) if m else text

    def _parse_joinner_output(self, raw_answer: str) -> tuple[str, str, bool]:
        """Parse the joiner output.

        Expected shape::

            Thought: xxx
            Action: Finish(yyy)        # or Replan(yyy)

        Hardening over the original:

        - Case-insensitive ``Finish`` / ``Replan`` matching.
        - Tolerates code fences around the response.
        - Tolerates missing closing parenthesis at end of stream.
        - If neither action is found at all, the joiner is asked to retry
          (caller decides). Here we return ``("", "", False)``.
        """
        if not raw_answer:
            return "", "", False

        text = self._strip_code_fences(raw_answer).strip()

        thought_match = re.search(
            r"Thought:\s*([\s\S]*?)(?=\s*Action:|$)", text, flags=re.IGNORECASE
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # ``Action: Finish(...)`` — accept any case, allow open paren without
        # close (truncated streams), and capture everything until the last
        # close paren or end of string.
        action_match = re.search(
            r"Action:\s*(Finish|Replan)\s*\(([\s\S]*?)(?:\)\s*$|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not action_match:
            logger.warning(
                "joiner output did not match Finish|Replan; raw=%r", raw_answer[:300]
            )
            return thought, "", False

        action_type = action_match.group(1).lower()
        answer = action_match.group(2).strip()
        is_replan = action_type == "replan"
        return thought, answer, is_replan

    def _generate_context_for_replanner(
        self, tasks: Mapping[int, Task], joinner_thought: str
    ) -> str:
        """Formatted like this:
        ```
        1. action 1
        Observation: xxx
        2. action 2
        Observation: yyy
        ...
        Thought: joinner_thought
        ```
        """
        previous_plan_and_observations = "\n".join(
            [
                task.get_though_action_observation(
                    include_action=True, include_action_idx=True
                )
                for task in tasks.values()
                if not task.is_join
            ]
        )
        joinner_thought = f"Thought: {joinner_thought}"
        context = "\n\n".join([previous_plan_and_observations, joinner_thought])
        return context

    def _format_contexts(self, contexts: Sequence[str]) -> str:
        """contexts is a list of context
        each context is formatted as the description of _generate_context_for_replanner
        """
        formatted_contexts = ""
        for context in contexts:
            formatted_contexts += f"Previous Plan:\n\n{context}\n\n"
        formatted_contexts += "Current Plan:\n\n"
        return formatted_contexts

    async def join(
        self, inputs: Dict[str, Any], agent_scratchpad: str, is_final: bool , 
        joinner_prompt_final_in , joinner_prompt_in
    ) -> str:

        input_query = inputs["input"]
        is_table_format = inputs["is_table_format"]

        if is_final:
            joinner_prompt = joinner_prompt_final_in
        else:
            joinner_prompt = joinner_prompt_in

        if is_table_format:
            joinner_prompt = TABLE_OUTPUT_PROMPT

        prompt = (
            f"{joinner_prompt}\n"  # Instructions and examples
            f"Question: {input_query}\n\n"  # User input query
            f"{agent_scratchpad}\n"
        )
        # log("Joining prompt:\n", prompt, block=True)
        response = await self.agent.arun(
            prompt, callbacks=[self.executor_callback] if self.benchmark else None
        )
        raw_answer = cast(str, response)
        log("Question: \n", input_query, block=True)
        log("Raw Answer: \n", raw_answer, block=True)
        thought, answer, is_replan = self._parse_joinner_output(raw_answer)
        if is_final:
            # If final, we don't need to replan
            if is_replan:
                answer = NO_ANWER_REPLY
            is_replan = False
        return thought, answer, is_replan

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        raise NotImplementedError("LLMCompiler is async only.")

    async def _acall(
        self,
        inputs: Dict[str, Any]

    ) -> Dict[str, Any]:
        """
        inputs = dict
        keys:
        planner_example_prompt
        planner_example_prompt_replan
        joinner_prompt
        joinner_prompt_final
        input
        purpose
        tools
        """
        # Always resolve the replan prompt to a concrete value. Previously this
        # was only assigned on the truthy-falsy branch (and to a *local* that
        # wasn't used downstream), so the replan path could read the wrong
        # prompt or NameError.
        planner_example_prompt = inputs['planner_example_prompt']
        planner_example_prompt_replan = (
            inputs.get('planner_example_prompt_replan') or planner_example_prompt
        )
        if not inputs.get('planner_example_prompt_replan'):
            logger.debug(
                "Replan example prompt not specified, falling back to the planner prompt."
            )
        joinner_prompt = inputs['joinner_prompt']
        joinner_prompt_final = inputs.get('joinner_prompt_final') or joinner_prompt

        events = self.event_log
        budget: Budget = inputs.get("budget") or Budget()
        budget.start()
        events.emit("run_start", question=inputs.get("input"))

        contexts = []
        joinner_thought = ""
        agent_scratchpad = ""
        answer = ""  # Initialize answer to prevent UnboundLocalError
        is_final_iter = False  # Initialize is_final_iter to prevent UnboundLocalError
        replans = 0
        status = "ok"

        # Get meta data with thinking process
        meta_data_result = await self.meta_planner.retrieve_meta_data(inputs['input'],inputs['purpose'],inputs['instructions'],inputs["tools"],self.llm,None,inputs.get("query_understanding", ""),inputs.get("temporal_context", ""),inputs.get("research_approach", ""),inputs.get("dos", ""),inputs.get("donts", ""),inputs.get("meta_example", ""))
        #question: str, purpose : str, instructions: str , tools: list[str], llm, message_manager
        thinking_process = meta_data_result.get("thinking_process", "")
        meta_data = meta_data_result.get("meta_plan", "")
        events.emit("meta_plan", meta_plan=meta_data)

        for i in range(self.max_replans):
            is_first_iter = i == 0
            is_final_iter = i == self.max_replans - 1

            # Budget tripwire — checked at iteration boundaries so we never
            # interrupt a tool call mid-flight.
            if budget.exceeded(self.get_all_stats()):
                status = "budget_exceeded"
                logger.info("Aborting run: %s", budget.reason(self.get_all_stats()))
                events.emit(
                    "run_end",
                    answer=answer,
                    replans=replans,
                    status=status,
                    reason=budget.reason(self.get_all_stats()),
                )
                break

            task_fetching_unit = TaskFetchingUnit()
            #llm,example_prompt,example_prompt_replan,tools,stop,inputs,meta_data,is_replan,callbacks
            tasks = await self.planner.plan(
                llm=self.llm,
                example_prompt=planner_example_prompt,
                example_prompt_replan=planner_example_prompt_replan,   
                inputs=inputs,
                meta_data=meta_data,
                tools=inputs["tools"],
                stop=[END_OF_PLAN],
                is_replan=not is_first_iter,
                # callbacks=run_manager.get_child() if run_manager else None,
                callbacks=[self.planner_callback]
                if self.planner_callback
                else None,
            )
            log("Graph of tasks: ", tasks, block=True)
            events.emit(
                "plan",
                iteration=i,
                num_tasks=len(tasks),
                tasks=[
                    {"idx": t.idx, "name": t.name, "deps": list(t.dependencies)}
                    for t in tasks.values()
                ],
            )
            if self.planner_callback is not None:
                self.planner_callback.additional_fields["num_tasks"] = len(tasks)
            task_fetching_unit.set_tasks(tasks)
            if self.message_manager is not None:
                await self.message_manager.send_message("System is processing data")
            sched_start = time.time()
            await task_fetching_unit.schedule()
            events.emit(
                "task_end",
                iteration=i,
                duration_s=round(time.time() - sched_start, 3),
                num_tasks=len(tasks),
            )
            tasks = task_fetching_unit.tasks
            # collect thought-action-observation
            agent_scratchpad += "\n\n"
            agent_scratchpad += "".join(
                [
                    task.get_though_action_observation(
                        include_action=True, include_thought=True
                    )
                    for task in tasks.values()
                    if not task.is_join
                ]
            )
            agent_scratchpad = agent_scratchpad.strip()

            log("Agent scratchpad:\n", agent_scratchpad, block=True)
            if self.message_manager is not None:
                await self.message_manager.send_message("preparing answer")
            joinner_thought, answer, is_replan = await self.join(
                inputs,
                agent_scratchpad=agent_scratchpad,
                is_final=is_final_iter,
                joinner_prompt_final_in=joinner_prompt_final,
                joinner_prompt_in=joinner_prompt,
            )
            # Retry the joiner once if its output failed to parse — the most
            # common failure mode is a stray code fence or capitalisation that
            # the hardened regex above already tolerates, but a totally absent
            # ``Action:`` line still results in (thought, "", False).
            if answer == "" and not is_replan:
                logger.info("Joiner produced no Action — retrying once.")
                joinner_thought, answer, is_replan = await self.join(
                    inputs,
                    agent_scratchpad=agent_scratchpad,
                    is_final=is_final_iter,
                    joinner_prompt_final_in=joinner_prompt_final,
                    joinner_prompt_in=joinner_prompt,
                )
            if not is_replan:
                log("Break out of replan loop.")
                events.emit("join", iteration=i, is_replan=False, answer=answer)
                if answer != NO_ANWER_REPLY:
                    log("Formatted Answer: \n", answer, block=True)
                    try:
                        log_task_execution(tasks=tasks, final_answer=answer)
                    except Exception as exc:  # pragma: no cover - best-effort log
                        logger.debug("log_task_execution failed: %s", exc)
                break

            replans += 1
            events.emit("replan", iteration=i, reason=joinner_thought)

            # Collect contexts for the subsequent replanner
            context = self._generate_context_for_replanner(
                tasks=tasks, joinner_thought=joinner_thought
            )
            contexts.append(context)
            formatted_contexts = self._format_contexts(contexts)
            log("Contexts:\n", formatted_contexts, block=True)
            inputs["context"] = formatted_contexts

        if is_final_iter and status == "ok":
            log("Reached max replan limit.")
        # Only emit run_end here if the budget path didn't already emit it.
        if status != "budget_exceeded":
            events.emit("run_end", answer=answer, replans=replans, status=status)
        return {
            self.output_key: answer,
            "thinking_process": thinking_process,
            "events": events.events,
            "stats": self.get_all_stats(),
            "tasks": dict(getattr(task_fetching_unit, "tasks", {})),
            "replans": replans,
            "status": status,
        }