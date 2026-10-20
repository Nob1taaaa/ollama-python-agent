#!/usr/bin/env python3
"""A small tool-calling agent backed by a local Ollama model."""

from __future__ import annotations

import ast
import operator
import os
import sys
from datetime import datetime

from ollama import Client

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
MAX_TOOL_ROUNDS = 5

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluate a math expression using +, -, *, /, //, %, and **.

    Args:
        expression: Arithmetic expression, for example "17 * 23".
    """
    try:
        value = _eval_math(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as exc:
        return f"error: {exc}"
    return str(value)


def _eval_math(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_math(node.left), _eval_math(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


TOOLS = [current_time, calculate]
TOOL_BY_NAME = {fn.__name__: fn for fn in TOOLS}


def ensure_model(client: Client) -> None:
    try:
        names = [model.model for model in client.list().models]
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {HOST}. Start it with `ollama serve`.\n{exc}"
        ) from exc
    if MODEL not in names:
        raise SystemExit(
            f"Model {MODEL!r} is not installed. Pull it with `ollama pull {MODEL}`.\n"
            f"Installed: {', '.join(names) or '(none)'}"
        )


def run_agent(client: Client, user_text: str) -> str:
    messages: list = [
        {
            "role": "system",
            "content": (
                "You are a local assistant running on Ollama. "
"ONLY use the calculate tool if the user explicitly asks you to solve a math problem or numeric calculation. Do not use tools for general text questions."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat(model=MODEL, messages=messages, tools=TOOLS)
        message = response.message
        messages.append(message)

        calls = message.tool_calls or []
        if not calls:
            return message.content or ""

        for call in calls:
            name = call.function.name
            args = call.function.arguments or {}
            print(f"  tool {name}({args})", file=sys.stderr)
            fn = TOOL_BY_NAME.get(name)
            if fn is None:
                result = f"error: unknown tool {name}"
            else:
                try:
                    result = str(fn(**args))
                except TypeError as exc:
                    result = f"error: {exc}"
            print(f"  -> {result}", file=sys.stderr)
            messages.append({"role": "tool", "content": result, "tool_name": name})

    return "Stopped after too many tool calls."


def main() -> None:
    client = Client(host=HOST)
    ensure_model(client)
    print(f"Connected to {HOST} using {MODEL}", file=sys.stderr)

    if len(sys.argv) > 1:
        print(run_agent(client, " ".join(sys.argv[1:])))
        return

    print("Ask something, or type quit.", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            return
        print(run_agent(client, line))


if __name__ == "__main__":
    main()
