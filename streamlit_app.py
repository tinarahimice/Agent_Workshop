"""Workshop chat UI for the standalone RAG and multi-tool agent paths."""
import asyncio

import streamlit as st

from src.agent import run_agent
from src.config import get_settings
from src.logging_config import configure_logging
from src.rag import query_knowledge_base
from src.redis_client import close_redis


async def answer(question: str, mode: str) -> tuple[str, list[str], bool | None]:
    try:
        if mode == "Agent":
            return await run_agent(question), [], None
        result = await query_knowledge_base(question)
        return result.answer, result.sources, result.cached
    finally:
        # Streamlit reruns this file; do not retain a client bound to an old event loop.
        await close_redis()


settings = get_settings()
configure_logging(settings.log_level)
st.set_page_config(page_title="NovaTech AI Workshop", page_icon="🧭")
st.title("🧭 NovaTech AI Workshop")
st.caption("OCR → Jina Embeddings → Vector Search → Jina Reranker → RAG → Agent")

with st.sidebar:
    st.header("Demo settings")
    mode = st.radio("Answer with", ("RAG", "Agent"))
    st.write(f"**LLM:** `{settings.llm_provider}`")
    model = settings.ollama_model if settings.llm_provider == "ollama" else settings.llm_model
    st.write(f"**Model:** `{model}`")
    st.write(f"**Embedding:** `{settings.embedding_model}`")
    st.write(f"**Reranker:** `{settings.reranker_model}`")
    st.info("RAG shows sources and cache status. Agent can search and then call calculators.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            st.caption(message["details"])

if question := st.chat_input("Ask about a NovaTech product or policy…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(f"Running {mode} pipeline…"):
                response, sources, cached = asyncio.run(answer(question, mode))
            details = ""
            if cached is not None:
                details = f"Cache: {'HIT' if cached else 'MISS'} · Sources: {', '.join(sources) or 'none'}"
            st.markdown(response)
            if details:
                st.caption(details)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "details": details}
            )
        except Exception as exc:
            message = f"Unable to answer: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
