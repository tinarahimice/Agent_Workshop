import logging

import pytest

from src.config import Settings
from src.rag import RagResult
from src.tools import knowledge_search_tool, search_knowledge_base


@pytest.mark.asyncio
async def test_search_returns_answer_without_exposing_sources(monkeypatch, caplog) -> None:
    async def fake_query(question: str) -> RagResult:
        assert question == "Tell me about Nova Monitor Ultra"
        return RagResult(answer="A 32-inch color-accurate display.", sources=["products.txt"])

    monkeypatch.setattr("src.tools.query_knowledge_base", fake_query)

    with caplog.at_level(logging.INFO, logger="TOOL"):
        answer = await search_knowledge_base("Tell me about Nova Monitor Ultra")

    assert answer == "A 32-inch color-accurate display."
    assert "products.txt" not in answer
    assert "sources=products.txt" in caplog.text


@pytest.mark.asyncio
async def test_settings_bound_search_does_not_expose_sources(monkeypatch) -> None:
    settings = Settings(_env_file=None)

    async def fake_query(question: str, settings: Settings) -> RagResult:
        return RagResult(answer="$850", sources=["product_catalog.txt"])

    monkeypatch.setattr("src.tools.query_knowledge_base", fake_query)

    answer = await knowledge_search_tool(settings)("What does it cost?")

    assert answer == "$850"
