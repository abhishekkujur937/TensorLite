"""Independent NumPy interpreter for .tl files.

Shares no code with the C++ compiler: it parses the file with regular
expressions and computes every value with NumPy (float32). The test runner
compares tlc's generated C against these results.

Limitation: expects one statement per line (all our .tl files do).
"""

import re
from pathlib import Path

import numpy as np

NAME = r"[A-Za-z_][A-Za-z0-9_]*"
DIMS = r"\[\s*(\d+)\s*(?:,\s*(\d+)\s*)?\]"

DECL_RE = re.compile(rf"^(input|weight)\s+({NAME})\s*{DIMS}$")
CONST_RE = re.compile(rf"^const\s+({NAME})\s*{DIMS}\s*=\s*(-?\d+(?:\.\d+)?)$")
ASSIGN_RE = re.compile(rf"^({NAME})\s*=\s*(.+)$")
OUTPUT_RE = re.compile(rf"^output\s+({NAME})$")

# Splits "relu(add(x, b))" into ["relu", "(", "add", "(", "x", ",", "b", ")", ")"]
TOKEN_RE = re.compile(rf"{NAME}|[(),]")


def dims_from_match(first, second):
    """Regex groups → shape tuple: ("64", "128") → (64, 128), ("256", None) → (256,)."""
    if second is None:
        return (int(first),)
    return (int(first), int(second))


def parse_tl(tl_path):
    """Returns a list of statements, each a dict with a "kind" key:
    decl   → kind, decl_kind, name, shape
    const  → kind, name, shape, value
    assign → kind, name, expr
    output → kind, name
    """
    statements = []
    text = Path(tl_path).read_text(encoding="utf-8")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#")[0].strip()
        if line == "":
            continue

        match = DECL_RE.match(line)
        if match:
            statements.append({
                "kind": "decl",
                "decl_kind": match.group(1),
                "name": match.group(2),
                "shape": dims_from_match(match.group(3), match.group(4)),
            })
            continue

        match = CONST_RE.match(line)
        if match:
            statements.append({
                "kind": "const",
                "name": match.group(1),
                "shape": dims_from_match(match.group(2), match.group(3)),
                "value": float(match.group(4)),
            })
            continue

        match = OUTPUT_RE.match(line)
        if match:
            statements.append({"kind": "output", "name": match.group(1)})
            continue

        match = ASSIGN_RE.match(line)
        if match:
            statements.append({"kind": "assign", "name": match.group(1), "expr": match.group(2)})
            continue

        raise ValueError(f"{tl_path}:{line_number}: reference.py cannot parse: {raw_line}")

    return statements


def op_matmul(a, b):
    return a @ b


def op_add(a, b):
    return a + b


def op_mul(a, b):
    return a * b


def op_relu(a):
    return np.maximum(a, np.float32(0.0))


OPS = {
    "matmul": op_matmul,
    "add": op_add,
    "mul": op_mul,
    "relu": op_relu,
}


def eval_tokens(tokens, pos, env):
    """Evaluates the expression starting at tokens[pos].
    Returns (value, position after the expression)."""
    name = tokens[pos]
    pos += 1

    # Plain variable.
    if pos >= len(tokens) or tokens[pos] != "(":
        return env[name], pos

    # Call: name ( expr , expr ... )
    pos += 1   # the "("
    args = []
    while True:
        value, pos = eval_tokens(tokens, pos, env)
        args.append(value)
        if tokens[pos] == ",":
            pos += 1
            continue
        pos += 1   # the ")"
        break

    result = OPS[name](*args)
    return result.astype(np.float32), pos


def run_reference(tl_path, data_dir):
    """Loads <name>.bin inputs/weights from data_dir, runs the program with
    NumPy, and returns {output_name: array}."""
    data_dir = Path(data_dir)
    env = {}
    outputs = {}

    for stmt in parse_tl(tl_path):
        if stmt["kind"] == "decl":
            path = data_dir / f"{stmt['name']}.bin"
            env[stmt["name"]] = np.fromfile(path, dtype=np.float32).reshape(stmt["shape"])
        elif stmt["kind"] == "const":
            env[stmt["name"]] = np.full(stmt["shape"], stmt["value"], dtype=np.float32)
        elif stmt["kind"] == "assign":
            tokens = TOKEN_RE.findall(stmt["expr"])
            value, _ = eval_tokens(tokens, 0, env)
            env[stmt["name"]] = value
        else:
            outputs[stmt["name"]] = env[stmt["name"]]

    return outputs
