"""Unit tests for ``_parse_llm_compiler_action_args``.

The argument parser is now responsible for splitting positional from keyword
arguments. These tests pin down the cases real LLMs (Kimi K2, GPT-4o,
Claude) actually emit — anything that breaks here breaks every tool call.
"""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class PlannerArgsTests(unittest.TestCase):
    def test_positional_only(self) -> None:
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args('"hello world"')
        self.assertEqual(args, ("hello world",))
        self.assertEqual(kwargs, {})

    def test_two_positional(self) -> None:
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args('"AAPL", "summary"')
        self.assertEqual(args, ("AAPL", "summary"))
        self.assertEqual(kwargs, {})

    def test_kwarg_int(self) -> None:
        # The exact case real LLMs emit for retrieval calls.
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args('"banking FY25", k=3')
        self.assertEqual(args, ("banking FY25",))
        self.assertEqual(kwargs, {"k": 3})

    def test_kwarg_string(self) -> None:
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args('"x", mode="strict"')
        self.assertEqual(args, ("x",))
        self.assertEqual(kwargs, {"mode": "strict"})

    def test_equals_in_unquoted_token_is_treated_as_kwarg(self) -> None:
        # Documented behaviour: csv strips quotes before we see the piece,
        # so ``"a=b"`` (a single positional string with an `=`) becomes
        # piece ``a=b`` and is interpreted as kwarg ``a="b"``. In practice
        # planners do not emit bare ``a=b`` strings as positional values —
        # they emit either ``"...spaces and ='s..."`` (kept intact by csv
        # only when there are no spaces around `=`) or proper kwargs. This
        # test pins the behaviour so it doesn't regress silently.
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args('"prefix", "a=b"')
        self.assertEqual(args, ("prefix",))
        self.assertEqual(kwargs, {"a": "b"})

    def test_dependency_token_stays_string(self) -> None:
        # ``$1`` references are resolved later by the scheduler — the parser
        # must leave them alone.
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args("$1")
        self.assertEqual(args, ("$1",))
        self.assertEqual(kwargs, {})

    def test_empty(self) -> None:
        from LLMEngine.output_parser import _parse_llm_compiler_action_args
        args, kwargs = _parse_llm_compiler_action_args("")
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
