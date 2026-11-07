
from langchain_core.tools import tool
from utility import  calculate
from fractions import Fraction
from decimal import Decimal, localcontext

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
