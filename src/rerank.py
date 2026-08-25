"""Node reranking through an Ollama-hosted yes/no reranker model."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field


class OllamaRerank(BaseNodePostprocessor):
    """Rank retrieved nodes with a local Qwen3 reranker served by Ollama.

    Qwen3 rerankers answer ``yes`` or ``no`` for a query/document pair. Relevant
    nodes are placed first, while the hybrid retrieval score provides stable
    ordering within each group.
    """

    model: str
    base_url: str = "http://localhost:11434"
    request_timeout: float = Field(120.0, gt=0)
    top_n: int = Field(3, ge=1)

    def _is_relevant(self, query: str, document: str) -> bool:
        prompt = (
            "Judge whether the document is relevant to the query. "
            "Answer with exactly yes or no.\n"
            f"Query: {query}\nDocument: {document}"
        )
        request = Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
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
        answer = str(payload.get("response", "")).strip().lower()
        if not answer:
            raise RuntimeError("Ollama reranker returned an empty response")
        return answer.startswith("yes")

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("OllamaRerank requires a query")
        ranked = sorted(
            nodes,
            key=lambda node: (
                self._is_relevant(query_bundle.query_str, node.node.get_content()),
                node.score if node.score is not None else float("-inf"),
            ),
            reverse=True,
        )
        return ranked[: self.top_n]
