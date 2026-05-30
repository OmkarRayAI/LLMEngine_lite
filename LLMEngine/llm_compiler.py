import asyncio
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.llms.base import BaseLLM
from langchain.prompts.base import StringPromptValue

from .callbacks import AsyncStatsCallbackHandler
from .chain import Chain
from .constants import JOINNER_REPLAN, JOINNER_FINISH
from .planner import Planner
from .metaplanner import MetaPlanner
from .task_fetching_unit import Task, TaskFetchingUnit
from .base import StructuredTool, Tool
from .logger_utils import log, log_task_execution
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
        **kwargs,
    ) -> None:
        """
        Args:
            max_replans: Maximum number of replans to do.
            benchmark: Whether to collect benchmark stats.

        """
        super().__init__(**kwargs)

        
        self.meta_planner = MetaPlanner()
        self.planner = Planner()
        self.llm = llm
        self.agent = LLMCompilerAgent(llm)
        
        self.planner_stream = False
        self.max_replans = max_replans
        self.message_manager = message_manager

        # callbacks
        self.benchmark = False
        if benchmark:
            self.planner_callback = AsyncStatsCallbackHandler(stream=False)
            self.executor_callback = AsyncStatsCallbackHandler(stream=False)
        else:
            self.planner_callback = None
            self.executor_callback = None
            
    def get(self, key):
        return getattr(self, key, None)

    def get_all_stats(self):
        stats = {}
        if self.benchmark:
            stats["planner"] = self.planner_callback.get_stats()
            stats["executor"] = self.executor_callback.get_stats()
            stats["total"] = {
                k: v + stats["executor"].get(k, 0) for k, v in stats["planner"].items()
            }

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

    def _parse_joinner_output(self, raw_answer: str) -> tuple[str, str, bool]:
        """
        Parse the joinner output format which is expected to be:
        ```
        Thought: xxx
        Action: Finish/Replan(yyy)
        ```

        Returns:
            tuple containing:
                thought (str): The thought content
                answer (str): The answer content inside Finish() or Replan()
                is_replan (bool): Whether this is a replan action
        """
        # Extract thought
        thought_pattern = r"Thought:\s*([\s\S]*?)(?=\s*Action:|$)"
        thought_match = re.search(thought_pattern, raw_answer)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Modified action pattern to handle missing closing parenthesis
        action_pattern = r"Action:\s*(Finish|Replan)\(([\s\S]*?)(?:\s*\)\s*$|$)"# Made ) optional
        
        action_match = re.search(action_pattern, raw_answer)
        answer = ""
        is_replan = False
        
        if action_match:
            action_type = action_match.group(1)
            answer = action_match.group(2).strip()
            is_replan = (action_type == "Replan")
        
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
        if not inputs['planner_example_prompt_replan']:
            log(
                "Replan example prompt not specified, using the same prompt as the planner."
            )
            planner_example_prompt_replan = inputs['planner_example_prompt']
        planner_example_prompt = inputs['planner_example_prompt']
        joinner_prompt = inputs['joinner_prompt']
        joinner_prompt_final = inputs['joinner_prompt_final'] or joinner_prompt
            
        contexts = []
        joinner_thought = ""
        agent_scratchpad = ""
        answer = ""  # Initialize answer to prevent UnboundLocalError
        is_final_iter = False  # Initialize is_final_iter to prevent UnboundLocalError


        # Get meta data with thinking process
        meta_data_result = await self.meta_planner.retrieve_meta_data(inputs['input'],inputs['purpose'],inputs['instructions'],inputs["tools"],self.llm,None,inputs.get("query_understanding", ""),inputs.get("temporal_context", ""),inputs.get("research_approach", ""),inputs.get("dos", ""),inputs.get("donts", ""),inputs.get("meta_example", ""))
        #question: str, purpose : str, instructions: str , tools: list[str], llm, message_manager
        thinking_process = meta_data_result.get("thinking_process", "")
        meta_data = meta_data_result.get("meta_plan", "")

        for i in range(self.max_replans):
            is_first_iter = i == 0
            is_final_iter = i == self.max_replans - 1

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
            if self.benchmark:
                self.planner_callback.additional_fields["num_tasks"] = len(tasks)
            task_fetching_unit.set_tasks(tasks)
            if self.message_manager is not None:
                await self.message_manager.send_message("System is processing data")
            await task_fetching_unit.schedule()
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
                joinner_prompt_in = joinner_prompt

            )
            if not is_replan:
                log("Break out of replan loop.")
                if answer != NO_ANWER_REPLY:
                    #processor = DataProcessor(agent=self.agent, executor_callback=[self.executor_callback], benchmark=self.benchmark)
                    
                    #answer = await processor.convert_to_markdown(answer)
                    #TODO: remove this graph generator
                    #answer = processor.replace_graph_placeholders(answer, tasks)
                    log("Formatted Answer: \n", answer, block=True)
                    log_task_execution(tasks=tasks, final_answer=answer)
                break

            # Collect contexts for the subsequent replanner
            context = self._generate_context_for_replanner(
                tasks=tasks, joinner_thought=joinner_thought
            )
            contexts.append(context)
            formatted_contexts = self._format_contexts(contexts)
            log("Contexts:\n", formatted_contexts, block=True)
            inputs["context"] = formatted_contexts

        if is_final_iter:
            log("Reached max replan limit.")
        return {self.output_key: answer, 
            "thinking_process": thinking_process
        }