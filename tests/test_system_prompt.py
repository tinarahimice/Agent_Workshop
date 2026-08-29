from pathlib import Path


def test_system_prompt_explicitly_routes_payment_questions_to_search() -> None:
    prompt = (Path(__file__).parents[1] / "prompts/system_prompt.txt").read_text(
        encoding="utf-8"
    )

    assert "payment methods" in prompt
    assert "MUST call it before answering every question about BitTeck" in prompt
    assert "without calling it first" in prompt
