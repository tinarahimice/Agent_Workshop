from types import SimpleNamespace

import pytest

from src.agent import run_agent


@pytest.mark.asyncio
async def test_run_agent_passes_conversation_history(monkeypatch) -> None:
    class FakeAgent:
        async def run(self, **kwargs):
            assert kwargs["user_msg"] == "Yep"
            assert [(message.role.value, message.content) for message in kwargs["chat_history"]] == [
                ("user", "I want NovaMonitor Ultra"),
                ("assistant", "Would you like its product details?"),
            ]
            return "The NovaMonitor Ultra costs $850."

    monkeypatch.setattr("src.agent.build_agent", lambda settings: FakeAgent())

    answer = await run_agent(
        "Yep",
        SimpleNamespace(),
        [
            {"role": "user", "content": "I want NovaMonitor Ultra"},
            {"role": "assistant", "content": "Would you like its product details?"},
        ],
    )

    assert answer == "The NovaMonitor Ultra costs $850."


@pytest.mark.asyncio
async def test_run_agent_executes_tool_call_markup_instead_of_showing_it(monkeypatch) -> None:
    class FakeAgent:
        async def run(self, **kwargs):
            return '<search_knowledge_base question="SPECIAL WARRANTY NOTICE"/>'

    searched_questions = []

    async def fake_search(question: str) -> str:
        searched_questions.append(question)
        return "The special warranty notice applies to refurbished products."

    monkeypatch.setattr("src.agent.build_agent", lambda settings: FakeAgent())
    monkeypatch.setattr(
        "src.agent.knowledge_search_tool",
        lambda settings: fake_search,
    )

    answer = await run_agent("BitTeck SPECIAL WARRANTY NOTICE", SimpleNamespace())

    assert searched_questions == ["SPECIAL WARRANTY NOTICE"]
    assert answer == "The special warranty notice applies to refurbished products."
