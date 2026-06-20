"""Top-level orchestration: math-mode LaTeX string -> Typst math string."""

from __future__ import annotations

from transpiler.emitter import TranspileError, emit
from transpiler.parser import ParseError, parse

__all__ = ["transpile", "TranspileError"]


def transpile(latex: str) -> str:
    """Convert a math-mode LaTeX fragment to a Typst math fragment.

    Raises :class:`TranspileError` on unsupported commands or malformed input.
    The returned string is the *inner* math content (no surrounding ``$``).
    """
    try:
        ast = parse(latex)
    except ParseError as exc:
        raise TranspileError(f"parse error: {exc}") from exc
    return emit(ast).strip()
