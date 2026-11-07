
from langchain_core.tools import tool
from utility import  calculate
@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions accurately.
    Use this tool for arithmetic, multiplication,
    division, percentages and other calculations.
    """
    try:
        return calculate(expression)
    except Exception as e:
        return f"Calculation error: {e}"
