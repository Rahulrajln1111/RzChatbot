
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from utility import  calculate
from fractions import Fraction
from decimal import Decimal, localcontext
from search_client import search_internet

from rag import retrieve, format_context, list_sources

def _thread_id(config: RunnableConfig) -> str | None:
    return (config or {}).get("configurable", {}).get("thread_id")

@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    """Search the user's uploaded documents (PDF/DOCX/TXT) for relevant passages.
    Use this when the question is about content the user has uploaded.
    Returns passages with source + page citations."""
    tid = _thread_id(config)
    if not tid:
        return "No active conversation / no documents indexed."
    results = retrieve(tid, query)
    return format_context(results)

@tool
def list_indexed_documents(config: RunnableConfig) -> str:
    """List the documents currently indexed for this conversation."""
    tid = _thread_id(config)
    if not tid:
        return "No active conversation."
    srcs = list_sources(tid)
    return "\n".join(srcs) if srcs else "No documents indexed yet."

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions accurately."""

    try:
        result = calculate(expression)

        # Convert exact result into a Fraction
        exact = Fraction(result)

        # Calculate decimal using arbitrary precision
        with localcontext() as ctx:
            ctx.prec = 30

            decimal_result = (
                Decimal(exact.numerator) /
                Decimal(exact.denominator)
            )

        return (
            f"Exact result: {exact}\n"
            f"Decimal result: {decimal_result}"
        )

    except Exception as e:
        return f"Calculation error: {e}"

@tool
def search_web(query: str) -> str:
    """Search the internet for current information.
    Use this when the user asks for latest news, current facts,
    documentation, or information that requires web research.
    """    
    try:
        results = search_internet(query)
        
        if not results:
            return "No search results found."

        return "\n\n".join(
            f"Title: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Content: {item['content']}"
            for item in results
        )

    except Exception as e:
        return f"Search error: {e}"