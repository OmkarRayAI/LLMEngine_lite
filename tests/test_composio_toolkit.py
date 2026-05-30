"""Tests for the ComposioToolkit adapter using a stub Composio client.

We don't import the real composio SDK here. Instead we hand the toolkit a
fake client object that mimics the documented surface
(``composio.create(user_id).tools()`` and ``composio.tools.execute(...)``).
"""
import asyncio
import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _gmail_send_spec():
    return {
        "type": "function",
        "function": {
            "name": "GMAIL_SEND_EMAIL",
            "description": "Send an email via Gmail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient address"},
                    "subject": {"type": "string", "description": "Subject line"},
                    "body": {"type": "string", "description": "HTML or plain body"},
                    "cc": {"type": "array", "description": "CC list"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    }


def _calendar_create_spec():
    return {
        "type": "function",
        "function": {
            "name": "GOOGLECALENDAR_CREATE_EVENT",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "attendees": {"type": "array"},
                },
                "required": ["summary", "start", "end"],
            },
        },
    }


class FakeSession:
    def __init__(self, tool_specs):
        self._specs = tool_specs
        self.last_kwargs = None

    def tools(self, **kwargs):
        self.last_kwargs = kwargs
        return list(self._specs)


class FakeToolsAPI:
    def __init__(self):
        self.calls = []

    def execute(self, slug, user_id=None, arguments=None):
        self.calls.append({"slug": slug, "user_id": user_id, "arguments": arguments})
        return {"ok": True, "slug": slug, "echo": arguments}


class FakeComposio:
    def __init__(self, specs):
        self._specs = specs
        self.tools = FakeToolsAPI()

    def create(self, user_id):
        return FakeSession(self._specs)


class ComposioToolkitTests(unittest.TestCase):
    def test_loads_and_normalizes_tools(self) -> None:
        from LLMEngine.integrations import ComposioToolkit

        composio = FakeComposio([_gmail_send_spec(), _calendar_create_spec()])
        kit = ComposioToolkit(
            user_id="omkar@example.com",
            toolkits=["GMAIL", "GOOGLECALENDAR"],
            composio=composio,
        )
        tools = kit.get_tools()
        names = sorted(t.name for t in tools)
        self.assertEqual(names, ["gmail_send_email", "googlecalendar_create_event"])

    def test_required_args_propagate_to_schema(self) -> None:
        from LLMEngine.integrations import ComposioToolkit

        composio = FakeComposio([_gmail_send_spec()])
        kit = ComposioToolkit(user_id="u", toolkits=["GMAIL"], composio=composio)
        tool = kit.get_tools()[0]
        schema = tool.args_schema.model_json_schema()
        self.assertIn("to", schema["properties"])
        self.assertIn("subject", schema["properties"])
        self.assertIn("body", schema["properties"])
        # cc is optional
        self.assertEqual(set(schema.get("required", [])), {"to", "subject", "body"})

    def test_execute_dispatches_to_composio(self) -> None:
        from LLMEngine.integrations import ComposioToolkit

        composio = FakeComposio([_gmail_send_spec()])
        kit = ComposioToolkit(user_id="omkar@example.com", composio=composio)
        tool = kit.get_tools()[0]
        result = asyncio.run(
            tool.coroutine(to="alex@example.com", subject="hi", body="hello")
        )
        payload = json.loads(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["slug"], "GMAIL_SEND_EMAIL")
        self.assertEqual(composio.tools.calls[0]["user_id"], "omkar@example.com")
        self.assertEqual(
            composio.tools.calls[0]["arguments"],
            {"to": "alex@example.com", "subject": "hi", "body": "hello"},
        )

    def test_errors_are_returned_as_json_string(self) -> None:
        from LLMEngine.integrations import ComposioToolkit

        class BoomTools:
            def execute(self, *a, **kw):
                raise RuntimeError("upstream 503")

        class BoomComposio:
            def __init__(self):
                self.tools = BoomTools()
            def create(self, user_id):
                return FakeSession([_gmail_send_spec()])

        kit = ComposioToolkit(user_id="u", composio=BoomComposio())
        tool = kit.get_tools()[0]
        result = asyncio.run(tool.coroutine(to="x", subject="y", body="z"))
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertEqual(payload["tool"], "GMAIL_SEND_EMAIL")


if __name__ == "__main__":
    unittest.main()
