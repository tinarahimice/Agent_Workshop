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
