"""Example: LLMEngine answering questions over an LLM-maintained wiki.

The wiki layout follows
https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f::

    my-wiki/
        index.md
        log.md
        AGENTS.md
        wiki/
            indian-banking-fy25.md
            llm-evaluation.md
        raw/
            icra-fy25.md

Run::

    OPENAI_API_KEY=... python examples/rag_llmwiki.py
"""
from __future__ import annotations

import asyncio
import sys

from LLMEngine import KnowledgeTool, LLMEngine, LLMWikiRetriever


async def main(wiki_root: str, question: str) -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Stand up the retriever and wrap it as a tool.
    retriever = LLMWikiRetriever(root=wiki_root)
    kb = KnowledgeTool(
        retriever,
        name="wiki",
        description=(
            "wiki(query: str, k: int = 5) -> str:\n"
            " - Searches the LLM-maintained wiki and returns top-k snippets\n"
            "   with their `wiki/` or `raw/` source paths.\n"
            " - ALWAYS query the wiki before answering domain questions; cite\n"
            "   the returned `source=` paths in your final answer."
        ),
    )

    engine = LLMEngine(llm=llm, max_replan=2)
    result = await engine.run(
        question=question,
        purpose="Answer questions from a curated knowledge wiki.",
        instructions=(
            "Always call the `wiki` tool first. Cite returned source paths in\n"
            "the final answer. If the wiki is thin, say so explicitly."
        ),
        tools=[kb.get_tool()],
    )
    print("Answer:\n", result.answer)
    print("\nReplans:", result.replans, "Duration:", round(result.duration_s, 2), "s")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    wiki_root = sys.argv[1] if len(sys.argv) > 1 else "./my-wiki"
    question = sys.argv[2] if len(sys.argv) > 2 else "What did Indian private banks report in FY25?"
    asyncio.run(main(wiki_root, question))
