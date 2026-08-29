"""Node reranking through an Ollama-hosted relevance judge."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field


class OllamaRerank(BaseNodePostprocessor):
    """Rank retrieved nodes with a local Qwen3 model served by Ollama.

    Every retrieved query/document pair receives a structured relevance score.
    Nodes are then sorted by that score before only ``top_n`` are returned. A
    reranking failure is deliberately propagated: RAG must not silently answer
    from the original retrieval order.
    """

    model: str
    base_url: str = "http://localhost:11434"
    request_timeout: float = Field(120.0, gt=0)
    top_n: int = Field(3, ge=1)

    def _relevance_score(self, query: str, document: str) -> float:
        prompt = (
            "Score how relevant the document is to the query from 0 to 100. "
            "Return only the requested JSON object.\n"
            f"Query: {query}\nDocument: {document}"
        )
        request = Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": {
                        "type": "object",
                        "properties": {
                            "relevance_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                            }
                        },
                        "required": ["relevance_score"],
                    },
                    "options": {"temperature": 0},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Ollama reranking failed at {self.base_url!r} with model "
                f"{self.model!r}: {type(exc).__name__}: {exc}. Pull the model "
                "and confirm Ollama is reachable."
            ) from exc
        answer = str(payload.get("response", "")).strip()
        try:
            score = float(json.loads(answer)["relevance_score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Ollama reranker returned an invalid relevance score: {answer!r}"
            ) from exc
        if not 0 <= score <= 100:
            raise RuntimeError(f"Ollama reranker score is outside 0..100: {score}")
        return score

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("OllamaRerank requires a query")
        scored_nodes = []
        for node in nodes:
            retrieval_score = node.score if node.score is not None else float("-inf")
            rerank_score = self._relevance_score(
                query_bundle.query_str, node.node.get_content()
            )
            scored_nodes.append((rerank_score, retrieval_score, node))
        ranked = [
            node
            for _, _, node in sorted(scored_nodes, key=lambda item: item[:2], reverse=True)
        ]
        return ranked[: self.top_n]
