"""Persistent calculator worker process.

Started once by ``tools.calculator`` and reused for every calculation. It reads
one JSON request per line from stdin and writes one JSON response per line to
stdout. SymPy is imported up front so the first real calculation is fast.
"""

import json
import sys

from tools.calculator import CalculatorError, _evaluate_expression, _sympy


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    _sympy()  # warm import
    _send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = {"ok": True, "result": _evaluate_expression(request["expression"])}
        except CalculatorError as exc:
            response = {"ok": False, "error": str(exc)}
        except Exception as exc:  # report every evaluation failure back to the parent
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _send(response)


if __name__ == "__main__":
    main()
