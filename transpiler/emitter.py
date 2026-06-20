"""AST -> Typst math string.

Adjacent atoms are joined with spaces. Typst collapses inter-atom spacing
automatically, so this is layout-safe and crucially keeps letter runs separate
(``R K`` rather than the single identifier ``RK``), matching LaTeX semantics for
the space-tokenized im2latex data.

Unsupported commands raise :class:`TranspileError`, which the batch harness
counts toward coverage rather than silently dropping.
"""

from __future__ import annotations

from transpiler import mapping
from transpiler.mapping import Func, Ignore, Sym
from transpiler.parser import (
    Char,
    Cmd,
    Delim,
    Env,
    Group,
    Node,
    Script,
    Seq,
    Sqrt,
)


class TranspileError(Exception):
    pass


def emit(node: Node) -> str:
    if isinstance(node, Seq):
        return _emit_seq(node)
    if isinstance(node, Char):
        return mapping.CHAR_MAP.get(node.c, node.c)
    if isinstance(node, Group):
        return _emit_seq(node.body)
    if isinstance(node, Cmd):
        return _emit_cmd(node)
    if isinstance(node, Script):
        return _emit_script(node)
    if isinstance(node, Sqrt):
        return _emit_sqrt(node)
    if isinstance(node, Delim):
        return _emit_delim(node)
    if isinstance(node, Env):
        return _emit_env(node)
    raise TranspileError(f"unknown AST node {node!r}")


def _emit_seq(seq: Seq) -> str:
    parts = (emit(item) for item in seq.items)
    return " ".join(p for p in parts if p != "")


def _emit_cmd(node: Cmd) -> str:
    name = node.name
    if name in mapping.DROP_ARG:
        return ""
    if name in mapping.ATTACH_TOP:
        return f"attach({_emit_seq(node.args[1])}, t: {_emit_seq(node.args[0])})"
    if name in mapping.ATTACH_BOTTOM:
        return f"attach({_emit_seq(node.args[1])}, b: {_emit_seq(node.args[0])})"
    entry = mapping.COMMANDS.get(name)
    if entry is None:
        raise TranspileError(f"unsupported command \\{name}")
    if isinstance(entry, Sym):
        return entry.typst
    if isinstance(entry, Ignore):
        return entry.typst
    if isinstance(entry, Func):
        args = ", ".join(_emit_seq(arg) for arg in node.args)
        return f"{entry.typst}({args})"
    raise TranspileError(f"unsupported command \\{name}")


def _grouped(node: Node | None) -> str:
    """Emit a node for use as a script argument, parenthesising if needed."""
    if node is None:
        return ""
    s = emit(node)
    if " " in s and not (s.startswith("(") and s.endswith(")")):
        return f"({s})"
    return s


def _emit_script(node: Script) -> str:
    if node.base is None:
        # Degenerate prescript-like form, e.g. ``{}^{\prime}`` or ``_{\mathrm M}``.
        # Emitting the bare script content avoids invalid Typst like ``^^`` / ``__``.
        parts = [emit(s) for s in (node.sup, node.sub) if s is not None]
        return " ".join(p for p in parts if p)
    out = _grouped(node.base)
    if node.sup is not None:
        out += "^" + _grouped(node.sup)
    if node.sub is not None:
        out += "_" + _grouped(node.sub)
    return out


def _emit_sqrt(node: Sqrt) -> str:
    radicand = _emit_seq(node.radicand)
    if node.index is not None:
        return f"root({_emit_seq(node.index)}, {radicand})"
    return f"sqrt({radicand})"


def _emit_delim(node: Delim) -> str:
    left = _delim_token(node.left)
    right = _delim_token(node.right)
    body = _emit_seq(node.body)
    inner = " ".join(part for part in (left, body, right) if part)
    return f"lr({inner})"


def _delim_token(tok: str) -> str:
    mapped = mapping.delimiter(tok)
    if mapped is None:
        raise TranspileError(f"unsupported delimiter {tok!r}")
    return mapped


def _emit_env(node: Env) -> str:
    name = node.name
    if name == "cases":
        branches = ", ".join(_emit_row_spaced(row) for row in node.rows)
        return f"cases({branches})"
    if name in mapping.MATRIX_DELIMS:
        delim = mapping.MATRIX_DELIMS[name]
        rows = "; ".join(
            ", ".join(_emit_seq(cell) for cell in row) for row in node.rows
        )
        prefix = "" if delim == "#none" else f"delim: {delim}, "
        return f"mat({prefix}{rows})"
    # alignment-style environments: flatten, dropping alignment markers
    return " ".join(_emit_row_spaced(row) for row in node.rows)


def _emit_row_spaced(row: list[Seq]) -> str:
    return " ".join(_emit_seq(cell) for cell in row)
