"""Typst-compiler validation gate.

Shells out to ``typst compile`` on the emitted Typst wrapped in display math.
A formula is only admissible to the training set if it compiles. Designed to be
batchable and CI-runnable (the ``typst`` binary is assumed on PATH).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from transpiler.emitter import TranspileError
from transpiler.transpile import transpile

TYPST_BIN = "typst"


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    stderr: str


@dataclass(frozen=True)
class ValidateResult:
    ok: bool
    stage: str  # "transpile" | "compile" | "ok"
    typst: str | None
    error: str


def typst_available() -> bool:
    return shutil.which(TYPST_BIN) is not None


def compile_typst(typst_math: str, timeout: float = 20.0) -> CompileResult:
    """Compile a Typst math fragment to PDF in a temp dir, discard output."""
    doc = f"$ {typst_math} $\n"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "eq.typ"
        out = Path(tmp) / "eq.pdf"
        src.write_text(doc, encoding="utf-8")
        try:
            proc = subprocess.run(
                [TYPST_BIN, "compile", str(src), str(out)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(False, "timeout")
        except FileNotFoundError:
            return CompileResult(False, f"{TYPST_BIN} not found on PATH")
    return CompileResult(proc.returncode == 0, proc.stderr.strip())


def validate(latex: str, timeout: float = 20.0) -> ValidateResult:
    """Transpile then compile, reporting which stage (if any) failed."""
    try:
        typst = transpile(latex)
    except TranspileError as exc:
        return ValidateResult(False, "transpile", None, str(exc))
    result = compile_typst(typst, timeout=timeout)
    if not result.ok:
        return ValidateResult(False, "compile", typst, result.stderr)
    return ValidateResult(True, "ok", typst, "")
