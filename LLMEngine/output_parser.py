from __future__ import annotations

import ast
import csv
from io import StringIO
import re
from typing import Any, Optional, Sequence, Tuple, Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import OutputParserException

from .task_fetching_unit import Task
from .base import StructuredTool, Tool

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"

END_OF_PLAN = "<END_OF_PLAN>"


def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


class LLMCompilerPlanParser(AgentOutputParser, extra="allow"):
    """Planning output parser."""

    def __init__(self, tools: Sequence[Union[Tool, StructuredTool]], **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def parse(self, text: str) -> list[str]:
        # 1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
        # pattern = r"(\d+)\. (\w+)\(([^)]+)\)"
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        matches = re.findall(pattern, text)

        graph_dict = {}

        for match in matches:
            # idx = 1, function = "search", args = "Ronaldo number of kids"
            # thought will be the preceding thought, if any, otherwise an empty string
            thought, idx, tool_name, args, _ = match
            idx = int(idx)

            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )

            graph_dict[idx] = task
            if task.is_join:
                break

        return graph_dict


### Helper functions


def _parse_llm_compiler_action_args(args: str) -> Tuple[tuple, dict]:
    """Parse arguments from the planner's ``name(arg1, arg2, key=value)`` form.

    Returns ``(positional_args, keyword_args)``. Real LLMs (Kimi, GPT, Claude)
    routinely emit ``tool("query", k=3)``-style kwargs even when not asked to;
    treating them as positional strings produces nonsense like ``"k=3"`` being
    coerced to int. We split here once, at the parser, so downstream code
    sees a clean call signature.
    """
    if args == "":
        return (), {}

    # Use csv.reader to split by commas, preserving quoted strings
    csv_reader = csv.reader(StringIO(args), skipinitialspace=True)
    raw_pieces = next(csv_reader)

    positional: list[Any] = []
    kwargs: dict[str, Any] = {}
    for piece in raw_pieces:
        # Detect ``name=value`` only when ``name`` is a bare Python identifier
        # before the first ``=``. This avoids treating ``"a=b"`` (inside a
        # string literal) as a kwarg, since that is one csv field already.
        keyword = _split_kwarg(piece)
        if keyword is not None:
            name, value = keyword
            kwargs[name] = _coerce_literal(value)
        else:
            positional.append(_coerce_literal(piece))

    return tuple(positional), kwargs


_KWARG_HEAD = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")


def _split_kwarg(piece: str) -> Optional[Tuple[str, str]]:
    """If ``piece`` starts with ``identifier=…``, return ``(identifier, rest)``.

    Returns ``None`` for plain positional values (including string literals
    that happen to contain an ``=`` inside them, since those are already a
    single csv field that does not start with a bare identifier).
    """
    m = _KWARG_HEAD.match(piece)
    if not m:
        return None
    name = m.group(1)
    value = piece[m.end():]
    return name, value


def _coerce_literal(raw: str) -> Any:
    """Best-effort: parse Python literals (numbers, strings, lists), else str."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _find_tool(
    tool_name: str, tools: Sequence[Union[Tool, StructuredTool]]
) -> Union[Tool, StructuredTool]:
    """Find a tool by name.

    Args:
        tool_name: Name of the tool to find.

    Returns:
        Tool or StructuredTool.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise OutputParserException(f"Tool {tool_name} not found.")


def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Sequence[Any]
) -> dict[str, list[str]]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        # depends on the previous step
        dependencies = list(range(1, idx))
    else:
        # define dependencies based on the dependency rule in tool_definitions.py
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, args)]

    return dependencies


def instantiate_task(
    tools: Sequence[Union[Tool, StructuredTool]],
    idx: int,
    tool_name: str,
    args: str,
    thought: str,
) -> Task:
    dependencies = _get_dependencies_from_graph(idx, tool_name, args)
    parsed_args, parsed_kwargs = _parse_llm_compiler_action_args(args)
    if tool_name == "join":
        # join does not have a tool
        tool_func = lambda x: None
        stringify_rule = None
    else:
        tool = _find_tool(tool_name, tools)
        if hasattr(tool, 'coroutine') and tool.coroutine:
            tool_func = tool.coroutine
        else:
            tool_func = tool.func
        stringify_rule = tool.stringify_rule
    return Task(
        idx=idx,
        name=tool_name,
        tool=tool_func,
        args=parsed_args,
        kwargs=parsed_kwargs,
        dependencies=dependencies,
        stringify_rule=stringify_rule,
        thought=thought,
        is_join=tool_name == "join",
    )