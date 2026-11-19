"""Restricted arithmetic and symbolic-calculus calculator for model tool calls."""

from __future__ import annotations

import ast
import atexit
import json
import math
import operator
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional


class CalculatorError(ValueError):
    """Raised when calculator input is invalid or cannot be evaluated."""


TIMEOUT_SECONDS = 2.0  # compute time per expression; worker startup is not counted
MAX_TIMEOUT_SECONDS = 60.0
WORKER_STARTUP_TIMEOUT_SECONDS = 60.0
MAX_EXPRESSION_LENGTH = 512
MAX_AST_NODES = 100
MAX_AST_DEPTH = 20
MAX_POWER = 10_000
MAX_OUTPUT_LENGTH = 2_000
MAX_INTEGER_BITS = 100_000
_SYMBOLIC_FUNCS = {"diff", "integrate", "limit"}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_ALLOWED_FUNCS = {
    "sqrt", "log", "pow", "max", "min", "abs", "sin", "cos", "tan",
    "exp", "ceil", "floor", "round", "sum", "prod", "mean", "diff",
    "integrate", "limit", "approx",
}


def _normalize(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("Expression must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters.")
    return (expression.strip().replace("^", "**").replace("×", "*")
            .replace("÷", "/").replace("−", "-").replace("π", "pi"))


def _check_tree(node: ast.AST, depth: int = 0) -> None:
    if depth > MAX_AST_DEPTH:
        raise CalculatorError("Expression is nested too deeply.")
    if sum(1 for _ in ast.walk(node)) > MAX_AST_NODES:
        raise CalculatorError("Expression contains too many operations.")
    for child in ast.iter_child_nodes(node):
        _check_tree(child, depth + 1)


def _sympy():
    import sympy as sp
    return sp


def _checked_power(base: Any, exponent: Any) -> Any:
    """Raise base to exponent after rejecting results that would be enormous."""
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise CalculatorError("Exponent must be a number.")
    if abs(exponent) > MAX_POWER:
        raise CalculatorError(f"Exponent magnitude must be at most {MAX_POWER}.")
    if isinstance(base, int) and not isinstance(base, bool) and isinstance(exponent, int) and exponent > 0:
        if abs(base) > 1 and (abs(base).bit_length() - 1) * exponent > MAX_INTEGER_BITS:
            raise CalculatorError("Integer result is too large.")
    try:
        return base ** exponent
    except OverflowError as exc:
        raise CalculatorError("Result is too large.") from exc


def _evaluate_node(node: ast.AST) -> Any:
    sp = _sympy()
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > 1e100:
            raise CalculatorError("Numeric literal is too large.")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", node.id):
            return sp.Symbol(node.id)
        raise CalculatorError(f"Name is not allowed: {node.id}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate_node(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _evaluate_node(node.left), _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow):
            return _checked_power(left, right)
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_INTEGER_BITS:
            raise CalculatorError("Integer result is too large.")
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name not in _ALLOWED_FUNCS:
            raise CalculatorError(f"Function is not allowed: {name}")
        args = [_evaluate_node(arg) for arg in node.args]
        return _call_function(name, args)
    raise CalculatorError(f"Unsupported syntax: {type(node).__name__}")


def _call_function(name: str, args: list[Any]) -> Any:
    sp = _sympy()
    if name in {"sum", "prod", "mean", "max", "min"}:
        values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        if not values:
            raise CalculatorError(f"{name} requires at least one value.")
        if name == "sum":
            return sp.Add(*values) if any(getattr(v, "free_symbols", set()) for v in values) else math.fsum(values)
        if name == "prod":
            return sp.Mul(*values) if any(getattr(v, "free_symbols", set()) for v in values) else math.prod(values)
        if name == "mean":
            return sp.Add(*values) / len(values) if any(getattr(v, "free_symbols", set()) for v in values) else math.fsum(values) / len(values)
        return (sp.Max if name == "max" else sp.Min)(*values) if any(getattr(v, "free_symbols", set()) for v in values) else (max(values) if name == "max" else min(values))
    if name in {"diff", "integrate", "limit"}:
        expected = {"diff": (2, 3), "integrate": (2, 4), "limit": (3, 3)}[name]
        if not expected[0] <= len(args) <= expected[1]:
            raise CalculatorError(f"{name} received the wrong number of arguments.")
        expr, variable = args[:2]
        if not isinstance(variable, sp.Symbol):
            raise CalculatorError(f"Second argument to {name} must be a variable name.")
        if name == "diff":
            order = int(args[2]) if len(args) == 3 else 1
            if order < 1 or order > 20:
                raise CalculatorError("Derivative order must be between 1 and 20.")
            return sp.diff(expr, variable, order)
        if name == "integrate":
            if len(args) == 2:
                return sp.integrate(expr, variable)
            if len(args) == 4:
                return sp.integrate(expr, (variable, args[2], args[3]))
            raise CalculatorError("integrate takes expr,var or expr,var,lower,upper.")
        return sp.limit(expr, variable, args[2])
    if name == "approx":
        if len(args) not in (1, 2):
            raise CalculatorError("approx takes expr and optional decimal precision.")
        precision = int(args[1]) if len(args) == 2 else 15
        if not 1 <= precision <= 50:
            raise CalculatorError("Approximation precision must be between 1 and 50 digits.")
        return sp.N(args[0], precision)

    if any(getattr(arg, "free_symbols", set()) for arg in args):
        funcs = {"sqrt": sp.sqrt, "log": sp.log, "pow": _checked_power,
                 "abs": sp.Abs, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                 "exp": sp.exp, "ceil": sp.ceiling, "floor": sp.floor}
        if name == "round":
            raise CalculatorError("round is only available for numeric values.")
        func = funcs.get(name)
        if func is None:
            raise CalculatorError(f"{name} is only available for numeric values.")
        return func(*args)

    funcs = {
        "sqrt": lambda x: math.sqrt(x[0]),
        "log": lambda x: math.log(x[0], x[1]) if len(x) > 1 else math.log(x[0]),
        "pow": lambda x: _checked_power(x[0], x[1]),
        "abs": lambda x: abs(x[0]),
        "sin": lambda x: math.sin(x[0]), "cos": lambda x: math.cos(x[0]),
        "tan": lambda x: math.tan(x[0]), "exp": lambda x: math.exp(x[0]),
        "ceil": lambda x: math.ceil(x[0]), "floor": lambda x: math.floor(x[0]),
        "round": lambda x: round(x[0], int(x[1])) if len(x) > 1 else round(x[0]),
    }
    func = funcs.get(name)
    if func is None:
        raise CalculatorError(f"Function is not available: {name}")
    if name == "pow" and len(args) != 2:
        raise CalculatorError("pow takes exactly two arguments.")
    return func(args)


def _evaluate_expression(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    _check_tree(tree)
    result = _evaluate_node(tree)
    uses_calculus = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SYMBOLIC_FUNCS
        for node in ast.walk(tree)
    )
    free_symbols = getattr(result, "free_symbols", set())
    if free_symbols and not uses_calculus:
        names = ", ".join(sorted(str(symbol) for symbol in free_symbols))
        raise CalculatorError(
            f"Unknown name(s): {names}. The calculator evaluates numeric expressions; "
            "variables are only allowed inside diff, integrate, or limit."
        )
    if isinstance(result, float) and math.isfinite(result):
        return f"{result:.12g}"  # hide binary floating-point noise such as 64.80000000000001
    return str(result)


class _CalculatorWorker:
    """One long-lived worker process, started once and reused for every call.

    A fresh process cannot reuse modules the main app already loaded, so the
    startup cost (Python + SymPy imports) is paid once here instead of per call.
    If a calculation exceeds its time limit, the worker is killed and a
    replacement starts warming up immediately.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._responses: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._ready = False

    def _start(self) -> None:
        self._responses = queue.Queue()
        self._ready = False
        self._process = subprocess.Popen(
            [sys.executable, "-m", "tools.calculator_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=Path(__file__).resolve().parents[1],
        )
        threading.Thread(
            target=self._read_responses, args=(self._process, self._responses), daemon=True
        ).start()

    @staticmethod
    def _read_responses(process: subprocess.Popen, responses: "queue.Queue[Optional[dict]]") -> None:
        for line in process.stdout:
            try:
                responses.put(json.loads(line))
            except json.JSONDecodeError:
                responses.put({"ok": False, "error": "Calculator worker returned an invalid response."})
        responses.put(None)  # the process exited

    def _stop(self) -> None:
        if self._process is not None:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._process = None
        self._ready = False

    def _ensure_ready(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._start()
        if self._ready:
            return
        try:
            message = self._responses.get(timeout=WORKER_STARTUP_TIMEOUT_SECONDS)
        except queue.Empty:
            self._stop()
            raise CalculatorError("Calculator worker did not start in time.")
        if not message or not message.get("ready"):
            self._stop()
            raise CalculatorError("Calculator worker failed to start.")
        self._ready = True

    def warm_up(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()

    def evaluate(self, expression: str, timeout: float) -> dict:
        with self._lock:
            self._ensure_ready()
            try:
                self._process.stdin.write(json.dumps({"expression": expression}) + "\n")
                self._process.stdin.flush()
            except OSError as exc:
                self._stop()
                raise CalculatorError("Calculator worker is not running.") from exc
            try:
                response = self._responses.get(timeout=timeout)
            except queue.Empty:
                self._stop()
                self._start()
                raise TimeoutError(f"Calculator exceeded its {timeout:.1f}s time limit.")
            if response is None:
                self._stop()
                raise CalculatorError("Calculator worker stopped unexpectedly.")
            return response

    def close(self) -> None:
        with self._lock:
            self._stop()


_worker = _CalculatorWorker()
atexit.register(_worker.close)


def warm_up() -> None:
    """Start the calculator worker in the background so the first call is fast."""
    _worker.warm_up()


def calculate(expression: str, timeout: float = TIMEOUT_SECONDS) -> str:
    """Safely evaluate arithmetic or symbolic calculus, enforcing a hard timeout.

    Expressions use radians for trigonometric functions. Examples include
    ``sqrt(9) + 2^3``, ``mean([1, 2, 6])``, ``diff(x^3, x)``,
    ``integrate(sin(x), x)``, and ``limit(sin(x)/x, x, 0)``.
    """
    normalized = _normalize(expression)
    if not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Timeout must be greater than 0 and no more than {MAX_TIMEOUT_SECONDS:g} seconds.")
    response = _worker.evaluate(normalized, timeout)
    if not response.get("ok"):
        raise CalculatorError(response.get("error", "Calculation failed."))
    result = response["result"]
    if len(result) > MAX_OUTPUT_LENGTH:
        raise CalculatorError("Result is too large to return safely.")
    return result
