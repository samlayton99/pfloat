"""Static no-leakage audit of the emulator build.

clang parses csrc/emul.c (which includes kernels.h, lapack_gelss.h and lapack_solve.h, macros
expanded) and this walks the syntax tree of every function defined in those files, every branch
included. Outside the rounding layer (the functions implementing the correctly rounded +, -, *,
/, sqrt) it reports:

- floating-point +, -, *, / (including compound assignments),
- calls to functions that are neither defined in these files nor exact (fabs, copysign,
  nearbyint, ldexp -- ldexp only ever feeds the format rounding, through LDEXP) nor thread
  plumbing (pthread mutexes, create/join),
- conversions from integers to floating types and between floating types,
- floating literals not passed through FROMD (which checks that they are format values).

Comparisons and unary minus are exact and allowed. ``audit()`` returns the findings; the test
suite requires them to equal ``ALLOWED`` exactly. Run ``python -m pfloat.audit`` to print them.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

CSRC = Path(__file__).resolve().parent / "csrc"
SOURCES = {"emul.c", "kernels.h", "lapack_gelss.h", "lapack_solve.h"}
ROUNDING_LAYER = {"to_bits", "from_bits", "sgn", "shift_of", "at_midpoint", "rnd", "round_fast", "add_slow",
                  "mul_slow", "div_slow", "sqrt_slow", "e_add", "e_sub", "e_mul", "e_div", "e_sqrt", "e_fromd",
                  "e_i2t", "e_setfmt", "flush_counters", "pb_round", "pb_events"}
EXACT_CALLS = {"fabs", "copysign", "nearbyint", "ldexp", "malloc", "free", "memcpy",
               "pthread_mutex_lock", "pthread_mutex_unlock", "pthread_create", "pthread_join"}
FLOAT_TYPES = {"double", "float", "long double", "_Float16", "__bf16"}
ARITH = {"+", "-", "*", "/"}
# The one allowed exception: DGELSS's crossover MNTHR = ILAENV(6) = INT(REAL(MIN(M,N))*1.6E0), an
# integer computed in single precision exactly as reference LAPACK does; it only selects whether
# to factor A (QR or LQ) before bidiagonalizing.
ALLOWED = [{"function": "lp_gelss", "what": "floating '*'", "type": "float",
            "operands": ["CStyleCastExpr", "FloatingLiteral:1.60000002"]},
           {"function": "lp_gelss", "what": "cast IntegralToFloating to float", "operands": ["DeclRefExpr"]},
           {"function": "lp_gelss", "what": "literal 1.60000002"}]


def _is_float(t: dict | None) -> bool:
    if not t:
        return False
    return t.get("desugaredQualType", t.get("qualType")) in FLOAT_TYPES or t.get("qualType") in FLOAT_TYPES


def _dump(src: Path) -> dict:
    out = subprocess.run(["clang", "-fsyntax-only", "-Xclang", "-ast-dump=json", "-std=c11", "-I", str(src),
                          str(src / "emul.c")], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _callee(call: dict) -> tuple[str | None, str | None]:
    """(name, declaration kind) of the called entity; a function pointer is a parameter or variable."""
    stack = list(call.get("inner", [])[:1])
    while stack:
        n = stack.pop()
        if n.get("kind") == "DeclRefExpr":
            ref = n.get("referencedDecl", {})
            return ref.get("name"), ref.get("kind")
        stack.extend(n.get("inner", []))
    return None, None


def _describe(node: dict) -> str:
    kind = node.get("kind")
    if kind in ("FloatingLiteral", "IntegerLiteral"):
        return f"{kind}:{node.get('value')}"
    if kind in ("ImplicitCastExpr", "ParenExpr") and node.get("inner"):
        return _describe(node["inner"][0])
    return kind or "?"


def audit(src: Path = CSRC) -> list[dict]:
    """Findings for the emulator build of the sources in ``src`` (default: this package)."""
    ast = _dump(Path(src))
    current = {"file": None}

    def track(node):
        locs = [node.get("loc", {}), node.get("range", {}).get("begin", {}), node.get("range", {}).get("end", {})]
        for loc in locs:
            for sub in (loc, loc.get("spellingLoc", {}), loc.get("expansionLoc", {})):
                if "file" in sub:
                    current["file"] = Path(sub["file"]).name

    functions = []
    for decl in ast.get("inner", []):
        stack = [decl]
        while stack:  # walk in document order so the file tracker stays current
            node = stack.pop()
            track(node)
            stack.extend(reversed(node.get("inner", [])))
        if decl.get("kind") == "FunctionDecl" and any(c.get("kind") == "CompoundStmt" for c in decl.get("inner", [])):
            if current["file"] in SOURCES:
                functions.append(decl)
    ours = {d["name"] for d in functions}
    findings = []

    def walk(node, fn, parent_call=None):
        kind = node.get("kind")
        if kind in ("BinaryOperator", "CompoundAssignOperator"):
            op = node.get("opcode", "").rstrip("=") if kind == "CompoundAssignOperator" else node.get("opcode")
            ftype = node.get("computationResultType") if kind == "CompoundAssignOperator" else node.get("type")
            if op in ARITH and _is_float(ftype):
                findings.append({"function": fn, "what": f"floating '{node['opcode']}'", "type": ftype.get("qualType"),
                                 "operands": [_describe(c) for c in node.get("inner", [])]})
        elif kind == "CallExpr":
            name, decl_kind = _callee(node)
            # calls through function pointers (the parallel-for running part functions) dispatch to
            # functions defined here, which are audited themselves
            if decl_kind == "FunctionDecl" and name not in ours and name not in EXACT_CALLS:
                findings.append({"function": fn, "what": f"call {name}"})
            parent_call = name
        elif kind in ("ImplicitCastExpr", "CStyleCastExpr") and node.get("castKind") in ("IntegralToFloating", "FloatingCast"):
            findings.append({"function": fn, "what": f"cast {node['castKind']} to {node['type'].get('qualType')}",
                             "operands": [_describe(c) for c in node.get("inner", [])]})
        elif kind == "FloatingLiteral" and parent_call != "e_fromd":
            findings.append({"function": fn, "what": f"literal {node.get('value')}"})
        for c in node.get("inner", []):
            walk(c, fn, parent_call)

    for decl in functions:
        if decl["name"] in ROUNDING_LAYER:
            continue
        for body in (c for c in decl.get("inner", []) if c.get("kind") == "CompoundStmt"):
            walk(body, decl["name"])
    return findings


if __name__ == "__main__":
    for f in audit():
        print(json.dumps(f))
