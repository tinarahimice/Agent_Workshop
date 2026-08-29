"""Local reranking with a dedicated FastEmbed cross-encoder model."""

from typing import Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field, PrivateAttr


class FastEmbedRerank(BaseNodePostprocessor):
    """Rerank every retrieved node with a dedicated cross encoder."""

    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(3, ge=1)
    _cross_encoder: Any = PrivateAttr(default=None)

    def _get_cross_encoder(self) -> Any:
        if self._cross_encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._cross_encoder = TextCrossEncoder(model_name=self.model)
        return self._cross_encoder

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("FastEmbedRerank requires a query")
        if not nodes:
            return []

        documents = [node.node.get_content() for node in nodes]
        try:
            scores = list(
                self._get_cross_encoder().rerank(query_bundle.query_str, documents)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Dedicated local reranking failed with model {self.model!r}: "
                f"{type(exc).__name__}: {exc}. Confirm the model can be downloaded "
                "and the FastEmbed cache is writable."
            ) from exc
        if len(scores) != len(nodes):
            raise RuntimeError(
                "Dedicated reranker returned a different number of scores than documents"
            )

        for node, score in zip(nodes, scores, strict=True):
            node.score = float(score)
        return sorted(
            nodes,
            key=lambda node: node.score if node.score is not None else float("-inf"),
            reverse=True,
        )[: self.top_n]
