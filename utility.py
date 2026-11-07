
import ast
import operator
from fractions import Fraction

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

def calculate(expression: str):
    if len(expression) > 200:
        raise ValueError("Expression too long")

    tree = ast.parse(expression, mode="eval")

    def solve(node):
        if isinstance(node, ast.Expression):
            return solve(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return Fraction(node.value)
            if type(node.value) is float:
                return Fraction(str(node.value))

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in OPS:
                raise ValueError("Unsupported operator")

            left = solve(node.left)
            right = solve(node.right)

            if op_type == ast.Pow:
                if right.denominator != 1 or abs(right) > 1000:
                    raise ValueError("Invalid exponent")

            return OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            value = solve(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            if isinstance(node.op, ast.UAdd):
                return value

        raise ValueError("Invalid mathematical expression")

    result = solve(tree)

    if isinstance(result, Fraction) and result.denominator == 1:
        return str(result.numerator)

    return str(result)