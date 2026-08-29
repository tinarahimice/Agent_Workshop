"""Ollama-first chat UI with automatic tool selection."""
import asyncio

import streamlit as st

from src.agent import run_agent
from src.config import Settings, get_settings
from src.logging_config import configure_logging
from src.redis_client import close_redis


async def answer(question: str, settings: Settings) -> str:
    try:
        return await run_agent(question, settings)
    finally:
        # Streamlit reruns this file; do not retain a client bound to an old event loop.
        await close_redis()


base_settings = get_settings()
configure_logging(base_settings.log_level)
st.set_page_config(page_title="AI Workshop", page_icon="🧭")
st.title("AI Workshop")
st.caption("OCR → Embeddings → Vector Search → Reranker → RAG → Agent")

with st.sidebar:
    st.header("Demo settings")
    enable_api_models = st.toggle(
        "Enable API models",
        value=False,
        help="Opt in to the configured OpenAI-compatible LLM and Jina embedding/reranking APIs.",
    )
    settings = base_settings.for_streamlit(enable_api_models)
    st.write(f"**LLM:** `{settings.llm_provider}`")
    model = (
        settings.ollama_model
        if settings.llm_provider == "ollama"
        else settings.llm_model
    )
    st.write(f"**Model:** `{model}`")
    embedding_model = (
        settings.ollama_embedding_model
        if settings.embedding_provider == "ollama"
        else settings.embedding_model
    )
    reranker_model = (
        settings.ollama_reranker_model
        if settings.reranker_provider == "ollama"
        else settings.reranker_model
    )
    st.write(f"**Embedding:** `{settings.embedding_provider}/{embedding_model}`")
    st.write(f"**Reranker:** `{settings.reranker_provider}/{reranker_model}`")
    if enable_api_models:
        st.warning("API mode uses configured keys and requires an index built with Jina embeddings.")
    else:
        st.success("Local mode: generation, embeddings, and reranking use Ollama.")
    st.info(
        "The assistant automatically decides whether to search the knowledge "
        "base or use a calculator."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            st.caption(message["details"])

if question := st.chat_input("Ask about a BitTeck product or policy…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Choosing and running the required tools…"):
                response = asyncio.run(answer(question, settings))
            details = f"Provider: {settings.llm_provider} · Tools selected automatically"
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
