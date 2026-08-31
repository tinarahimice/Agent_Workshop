"""Tool-selecting LlamaIndex workflow agent."""
import logging
import re
from collections.abc import Sequence
from typing import TypedDict

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from src.config import Settings, get_settings
from src.llm import create_llm
from src.tools import calculate_discount, calculate_final_price, knowledge_search_tool

log=logging.getLogger("AGENT")

_SEARCH_CALL = re.compile(
    r"<search_knowledge_base\b[^>]*\bquestion\s*=\s*([\"'])(?P<question>.*?)\1[^>]*/?>",
    re.IGNORECASE | re.DOTALL,
)


class ConversationMessage(TypedDict):
    role: str
    content: str


def build_agent(settings: Settings | None=None) -> FunctionAgent:
    settings=settings or get_settings()
    system_prompt=settings.prompt_path.read_text(encoding="utf-8").strip()
    return FunctionAgent(
        tools=[knowledge_search_tool(settings), calculate_discount, calculate_final_price],
        llm=create_llm(settings),
        system_prompt=system_prompt,
    )


async def run_agent(
    question: str,
    settings: Settings | None = None,
    chat_history: Sequence[ConversationMessage] | None = None,
) -> str:
    """Answer a question with enough prior conversation to resolve follow-ups."""
    history = [ChatMessage(**message) for message in (chat_history or [])]
    log.info("USER QUERY %s",question)
    response = await build_agent(settings).run(
        user_msg=question,
        chat_history=history,
    )
    answer = str(response).strip()

    # Small local models occasionally print an XML-like function call instead of
    # handing it back to LlamaIndex.  That syntax is an implementation detail and
    # must never become the assistant's user-facing answer.  Recover the intended
    # search here and return its natural-language result instead.
    search_call = _SEARCH_CALL.fullmatch(answer)
    if search_call:
        search_question = search_call.group("question").strip() or question
        log.warning("Recovering unexecuted search tool call")
        answer = await knowledge_search_tool(settings or get_settings())(search_question)

    log.info("FINAL ANSWER %s",answer); return answer
