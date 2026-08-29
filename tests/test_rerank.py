import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.rerank import FastEmbedRerank


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return self.scores


def test_every_retrieved_node_is_batch_reranked_before_top_n() -> None:
    reranker = FastEmbedRerank(top_n=2)
    encoder = FakeCrossEncoder([0.1, 0.95, 0.6])
    reranker._cross_encoder = encoder
    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=0.9),
        NodeWithScore(node=TextNode(text="second"), score=0.8),
        NodeWithScore(node=TextNode(text="third"), score=0.7),
    ]

    result = reranker.postprocess_nodes(nodes, QueryBundle("question"))

    assert encoder.calls == [("question", ["first", "second", "third"])]
    assert [node.node.get_content() for node in result] == ["second", "third"]
    assert [node.score for node in result] == [0.95, 0.6]


def test_reranking_failure_is_not_silently_ignored() -> None:
    class FailedCrossEncoder:
        def rerank(self, _query: str, _documents: list[str]) -> list[float]:
            raise OSError("model unavailable")

    reranker = FastEmbedRerank()
    reranker._cross_encoder = FailedCrossEncoder()
    nodes = [NodeWithScore(node=TextNode(text="document"), score=0.9)]

    with pytest.raises(RuntimeError, match="Dedicated local reranking failed"):
        reranker.postprocess_nodes(nodes, QueryBundle("question"))
