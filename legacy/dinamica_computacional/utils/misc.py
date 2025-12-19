from __future__ import annotations

import ast
import math
from typing import Any, Dict

import numpy as np


ALLOWED_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "sqrt": np.sqrt,
}

ALLOWED_NAMES = {"pi": math.pi}


class SafeEval(ast.NodeVisitor):
    def __init__(self, names: Dict[str, Any]):
        self.names = names

    def visit(self, node):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant)):
            return super().visit(node)
        raise ValueError(f"Unsafe expression node: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("Unsupported binary operator")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        func = ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_Name(self, node: ast.Name):
        if node.id in self.names:
            return self.names[node.id]
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise ValueError(f"Name not allowed: {node.id}")

    def visit_Constant(self, node: ast.Constant):
        return node.value


def safe_eval_expr(expr: str, names: Dict[str, Any]):
    tree = ast.parse(expr, mode="eval")
    visitor = SafeEval(names)
    return visitor.visit(tree)
