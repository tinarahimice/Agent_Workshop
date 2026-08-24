import logging
from src.rag import query_knowledge_base
log=logging.getLogger("TOOL")
def _valid(value: float,name: str) -> float:
    value=float(value)
    if value < 0: raise ValueError(f"{name} cannot be negative")
    return value
def _percent(value: float,name: str) -> float:
    value=_valid(value,name)
    if value > 100: raise ValueError(f"{name} must be between 0 and 100")
    return value
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate a product price after a percentage discount."""
    result=round(_valid(price,"price")*(1-_percent(discount_percent,"discount_percent")/100),2)
    log.info("TOOL RESULT calculate_discount=%s",result); return result
def calculate_final_price(price: float, tax_percent: float) -> float:
    """Calculate a product's final price after percentage sales tax."""
    result=round(_valid(price,"price")*(1+_percent(tax_percent,"tax_percent")/100),2)
    log.info("TOOL RESULT calculate_final_price=%s",result); return result
async def search_knowledge_base(question: str) -> str:
    """Search BitTeck products, prices, warranties, shipping, returns, FAQs, and company policies."""
    log.info("TOOL SELECTED search_knowledge_base arguments=%s",question)
    result=await query_knowledge_base(question)
    return f"{result.answer}\nSources: {', '.join(result.sources) or 'none'}"
