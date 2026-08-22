"""Current LlamaIndex workflow agent with observable, explicit tools."""
import logging
from llama_index.core.agent.workflow import FunctionAgent
from src.config import Settings, get_settings
from src.llm import create_llm
from src.tools import calculate_discount, calculate_final_price, search_knowledge_base
log=logging.getLogger("AGENT")
def build_agent(settings: Settings | None=None) -> FunctionAgent:
    settings=settings or get_settings()
    system_prompt=settings.prompt_path.read_text(encoding="utf-8").strip()
    return FunctionAgent(
        tools=[search_knowledge_base, calculate_discount, calculate_final_price],
        llm=create_llm(settings),
        system_prompt=system_prompt,
    )
async def run_agent(question: str, settings: Settings | None=None) -> str:
    log.info("USER QUERY %s",question); response=await build_agent(settings).run(user_msg=question); answer=str(response)
    log.info("FINAL ANSWER %s",answer); return answer
