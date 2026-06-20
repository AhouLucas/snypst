"""Deterministic math-mode LaTeX to Typst transpiler.

Pipeline: tokenizer -> parser -> emitter, with mapping.py as the per-command
source of truth and validator.py as the typst-compile gate.
"""

from transpiler.transpile import TranspileError, transpile

__all__ = ["transpile", "TranspileError"]
