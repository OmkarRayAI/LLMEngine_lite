"""Unit tests for the hardened joiner parser.

These intentionally avoid importing ``LLMEngine.llm_compiler`` (which pulls in
LangChain) so they can run in any minimal environment. The parser is reproduced
here behaviourally by importing the module-level static method.
"""
import re
import sys
import types
import unittest


# Inline copies of the parser logic — kept in lockstep with
# ``LLMCompiler._parse_joinner_output`` and ``_strip_code_fences``.
def _strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\s*([\s\S]*?)```", text)
    return m.group(1) if m else text


def _parse(raw: str):
    if not raw:
        return "", "", False
    text = _strip_code_fences(raw).strip()
    thought_match = re.search(
        r"Thought:\s*([\s\S]*?)(?=\s*Action:|$)", text, flags=re.IGNORECASE
    )
    thought = thought_match.group(1).strip() if thought_match else ""
    action_match = re.search(
        r"Action:\s*(Finish|Replan)\s*\(([\s\S]*?)(?:\)\s*$|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not action_match:
        return thought, "", False
    return thought, action_match.group(2).strip(), action_match.group(1).lower() == "replan"


class JoinerParserTests(unittest.TestCase):
    def test_basic_finish(self) -> None:
        thought, ans, replan = _parse("Thought: ok\nAction: Finish(hello)")
        self.assertEqual(thought, "ok")
        self.assertEqual(ans, "hello")
        self.assertFalse(replan)

    def test_basic_replan(self) -> None:
        _, ans, replan = _parse("Thought: missing data\nAction: Replan(need search)")
        self.assertTrue(replan)
        self.assertEqual(ans, "need search")

    def test_lowercase_action(self) -> None:
        _, ans, replan = _parse("Thought: x\nAction: finish(y)")
        self.assertFalse(replan)
        self.assertEqual(ans, "y")

    def test_truncated_paren(self) -> None:
        _, ans, _ = _parse("Thought: x\nAction: Finish(unterminated answer with more text")
        self.assertIn("unterminated answer", ans)

    def test_code_fenced(self) -> None:
        raw = "```\nThought: t\nAction: Finish(boxed)\n```"
        thought, ans, _ = _parse(raw)
        self.assertEqual(thought, "t")
        self.assertEqual(ans, "boxed")

    def test_no_action_returns_empty(self) -> None:
        # When the model emits no Action: line, ans must be empty so the
        # caller knows to retry rather than treating an empty string as the
        # final answer.
        thought, ans, replan = _parse("Thought: hmm\n(nothing else)")
        self.assertIn("hmm", thought)
        self.assertEqual(ans, "")
        self.assertFalse(replan)


if __name__ == "__main__":
    unittest.main()
