import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.rerank import OllamaRerank


def test_every_retrieved_node_is_reranked_before_top_n(monkeypatch) -> None:
    reranker = OllamaRerank(model="qwen3:0.6b", top_n=2)
    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=0.9),
        NodeWithScore(node=TextNode(text="second"), score=0.8),
        NodeWithScore(node=TextNode(text="third"), score=0.7),
    ]
    scores = {"first": 10.0, "second": 95.0, "third": 60.0}
    judged: list[str] = []

    def score(_self: OllamaRerank, _query: str, document: str) -> float:
        judged.append(document)
        return scores[document]

    monkeypatch.setattr(OllamaRerank, "_relevance_score", score)

    result = reranker.postprocess_nodes(nodes, QueryBundle("question"))

    assert judged == ["first", "second", "third"]
    assert [node.node.get_content() for node in result] == ["second", "third"]


def test_reranking_failure_is_not_silently_ignored(monkeypatch) -> None:
    reranker = OllamaRerank(model="qwen3:0.6b")
    nodes = [NodeWithScore(node=TextNode(text="document"), score=0.9)]

    def fail(_self: OllamaRerank, _query: str, _document: str) -> float:
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(OllamaRerank, "_relevance_score", fail)

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        reranker.postprocess_nodes(nodes, QueryBundle("question"))
