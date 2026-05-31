from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Dict, List, Optional

from .logger_utils import log

SCHEDULING_INTERVAL = 0.01  # seconds


def _default_stringify_rule_for_arguments(args):
    def stringify(arg):
        # Handle strings by adding quotes around them
        if isinstance(arg, str):
            return f'"{arg}"'
        return str(arg)  # Default conversion for other types

    # If there's only one argument, wrap it in parentheses
    if len(args) == 1:
        return f"({stringify(args[0])})"
    else:
        # Handle multiple arguments as a tuple
        return "(" + ", ".join(stringify(arg) for arg in args) + ")"



def _replace_arg_mask_with_real_value(
    args, dependencies: List[int], tasks: Dict[str, Task]
):
    if isinstance(args, (list, tuple)):
        return type(args)(
            _replace_arg_mask_with_real_value(item, dependencies, tasks)
            for item in args
        )
    elif isinstance(args, str):
        for dependency in sorted(dependencies, reverse=True):
            # consider both ${1} and $1 (in case planner makes a mistake)
            for arg_mask in ["${" + str(dependency) + "}", "$" + str(dependency)]:
                if arg_mask in args:
                    if tasks[dependency].observation is not None:
                        args = args.replace(
                            arg_mask, str(tasks[dependency].observation)
                        )
        return args
    else:
        return args


@dataclass
class Task:
    idx: int
    name: str
    tool: Callable
    args: Collection[Any]
    dependencies: Collection[int]
    stringify_rule: Optional[Callable] = None
    thought: Optional[str] = None
    observation: Optional[str] = None
    is_join: bool = False
    # Keyword arguments emitted by the planner as ``name(arg, key=value)``.
    # Default-empty so existing code paths that never set this still work.
    kwargs: Dict[str, Any] = field(default_factory=dict)

    async def __call__(self) -> Any:
        log("running task")
        x = await self.tool(*self.args, **self.kwargs)
        log("done task")
        return x

    def get_though_action_observation(
        self, include_action=True, include_thought=True, include_action_idx=False
    ) -> str:
        thought_action_observation = ""
        if self.thought and include_thought:
            thought_action_observation = f"Thought: {self.thought}\n"
        if include_action:
            idx = f"{self.idx}. " if include_action_idx else ""
            if self.stringify_rule:
                # If the user has specified a custom stringify rule for the
                # function argument, use it
                thought_action_observation += f"{idx}{self.stringify_rule(self.args)}\n"
            else:
                # Otherwise, we have a default stringify rule
                thought_action_observation += (
                    f"{idx}{self.name}"
                    f"{_default_stringify_rule_for_arguments(self.args)}\n"
                )
        if self.observation is not None:
            thought_action_observation += f"Observation: {self.observation}\n"
        return thought_action_observation


class TaskFetchingUnit:
    tasks: Dict[str, Task]
    tasks_done: Dict[str, asyncio.Event]
    remaining_tasks: set[str]

    def __init__(self):
        self.tasks = {}
        self.tasks_done = {}
        self.remaining_tasks = set()

    def set_tasks(self, tasks: dict[str, Any]):
        self.tasks.update(tasks)
        self.tasks_done.update({task_idx: asyncio.Event() for task_idx in tasks})
        self.remaining_tasks.update(set(tasks.keys()))

    def _all_tasks_done(self):
        return all(self.tasks_done[d].is_set() for d in self.tasks_done)

    def _get_all_executable_tasks(self):
        return [
            task_name
            for task_name in self.remaining_tasks
            if all(
                self.tasks_done[d].is_set() for d in self.tasks[task_name].dependencies
            )
        ]

    def _preprocess_args(self, task: Task):
        """Replace dependency placeholders, i.e. ${1}, in task.args with the actual observation."""
        args = []
        for arg in task.args:
            arg = _replace_arg_mask_with_real_value(arg, task.dependencies, self.tasks)
            args.append(arg)
        task.args = args

    async def _run_task(self, task: Task):
        self._preprocess_args(task)
        try:
            if not task.is_join:
                try:
                    task.observation = await task()
                except Exception as exc:
                    # A tool that raises must not leave the task "running"
                    # forever — the scheduler waits on tasks_done.is_set().
                    # Surface the error as an observation so the joiner can
                    # see it and decide to replan.
                    log(f"task {task.idx} ({task.name}) raised: {exc!r}")
                    task.observation = (
                        f"ERROR: tool {task.name!r} raised "
                        f"{type(exc).__name__}: {exc}"
                    )
        finally:
            self.tasks_done[task.idx].set()

    async def schedule(self):
        """Run all tasks in self.tasks in parallel, respecting dependencies."""
        # run until all tasks are done
        while not self._all_tasks_done():
            # Find tasks with no dependencies or with all dependencies met
            executable_tasks = self._get_all_executable_tasks()

            for task_name in executable_tasks:
                asyncio.create_task(self._run_task(self.tasks[task_name]))
                self.remaining_tasks.remove(task_name)

            await asyncio.sleep(SCHEDULING_INTERVAL)

    async def aschedule(self, task_queue: asyncio.Queue[Optional[Task]], func):
        """Asynchronously listen to task_queue and schedule tasks as they arrive."""
        no_more_tasks = False  # Flag to check if all tasks are received

        while True:
            if not no_more_tasks:
                # Wait for a new task to be added to the queue
                task = await task_queue.get()

                # Check for sentinel value indicating end of tasks
                if task is None:
                    no_more_tasks = True
                else:
                    # Parse and set the new tasks
                    self.set_tasks({task.idx: task})

            # Schedule and run executable tasks
            executable_tasks = self._get_all_executable_tasks()

            if executable_tasks:
                for task_name in executable_tasks:
                    asyncio.create_task(self._run_task(self.tasks[task_name]))
                    self.remaining_tasks.remove(task_name)
            elif no_more_tasks and self._all_tasks_done():
                # Exit the loop if no more tasks are expected and all tasks are done
                break
            else:
                # If no executable tasks are found, sleep for the SCHEDULING_INTERVAL
                await asyncio.sleep(SCHEDULING_INTERVAL)