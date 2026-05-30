"""Domain-neutral defaults for the planner and joiner prompts.

The prompts shipped in ``prompts.py`` are finance-flavoured (Tesla/Apple).
These minimal defaults work for any toolset and are used by ``RunConfig`` when
the caller does not supply their own.
"""
from .constants import END_OF_PLAN, JOINNER_FINISH, JOINNER_REPLAN

DEFAULT_PLANNER_PROMPT = (
    "Question: Find the latest open issues for repository OWNER/REPO and email a summary to the maintainer.\n"
    "Thought: First, fetch the open issues from the source of truth.\n"
    '1. github_agent("List the latest open issues for OWNER/REPO with their titles and authors")\n'
    "Thought: Next, send the summary using the previous step's output as input.\n"
    '2. email_agent("Send a summary email", $1)\n'
    "Thought: All required data is gathered, finalize.\n"
    f"3. join() {END_OF_PLAN}\n"
)

DEFAULT_JOINNER_PROMPT = (
    "Solve the user's question using the Assistant Scratchpad below.\n"
    "Guidelines:\n"
    "- Reason briefly (1-2 sentences) about whether the observations contain enough\n"
    "  information to answer the question.\n"
    "- Ignore irrelevant action results.\n"
    "- If the information is sufficient, finish with the most complete answer you can.\n"
    "- If a tool failed or the answer is missing, replan with a short reason.\n"
    "\n"
    "Respond strictly in this format:\n"
    "Thought: <your reasoning>\n"
    "Action: <action>\n"
    "Available actions:\n"
    f"(1) {JOINNER_FINISH}(<final answer>): finishes the task.\n"
    f"(2) {JOINNER_REPLAN}(<reason>): asks for another planning iteration.\n"
)
