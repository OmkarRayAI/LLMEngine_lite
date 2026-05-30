"""Example: LLMEngine planning a Gmail + Calendar task via Composio.

Auth (OAuth, refresh, token storage) lives inside Composio. LLMEngine plans a
DAG of tool calls and dispatches each one through the Composio SDK using the
configured ``user_id``.

Setup::

    pip install composio
    export COMPOSIO_API_KEY=...
    # one-time: connect Gmail + Google Calendar to the user_id below
    # via the Composio dashboard or CLI

Run::

    OPENAI_API_KEY=... python examples/composio_gmail_calendar.py
"""
from __future__ import annotations

import asyncio
import sys

from LLMEngine import LLMEngine
from LLMEngine.integrations import ComposioToolkit


async def main(user_id: str, question: str) -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    toolkit = ComposioToolkit(
        user_id=user_id,
        toolkits=["GMAIL", "GOOGLECALENDAR"],
    )

    engine = LLMEngine(llm=llm, max_replan=2)
    result = await engine.run(
        question=question,
        purpose=(
            "You are a personal assistant for the user. Use the available\n"
            "Gmail and Google Calendar tools to satisfy the request."
        ),
        instructions=(
            "Plan tool calls in parallel where they are independent.\n"
            "Always finish with a one-paragraph summary of what you did."
        ),
        tools=toolkit.get_tools(),
        max_seconds=120,  # belt + suspenders against runaway tool loops
    )
    print(result.answer)
    print("\nstats:", result.stats.get("total"))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    user = sys.argv[1] if len(sys.argv) > 1 else "demo@example.com"
    question = sys.argv[2] if len(sys.argv) > 2 else (
        "Find any unread email from Alex this week, then schedule a 30-minute "
        "follow-up with them tomorrow afternoon."
    )
    asyncio.run(main(user, question))
