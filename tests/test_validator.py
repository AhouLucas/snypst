"""Integration tests that invoke the real ``typst`` compiler.

Skipped automatically when the binary is unavailable so the unit suite still
runs in environments without typst.
"""

import pytest

from transpiler.validator import compile_typst, typst_available, validate

pytestmark = pytest.mark.skipif(
    not typst_available(), reason="typst binary not on PATH"
)

REAL_FORMULAS = [
    r"R _ { 1 2 } K _ { 1 } R _ { 2 1 } d K _ { 2 }",
    r"E _ { n } - E _ { m } = \frac { \lambda ^ { \prime } ( n ^ { 2 } ) } { y ^ { 2 } }",
    r"\sigma ^ { 1 } + i \sigma ^ { 2 } = f ( \sigma ^ { 1 } + i \sigma ^ { 2 } )",
    r"\sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 }",
    r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
    r"\left( \frac { a } { b } \right)",
    r"\sqrt [ 3 ] { x + 1 }",
]


@pytest.mark.parametrize("latex", REAL_FORMULAS)
def test_real_formulas_compile(latex: str):
    result = validate(latex)
    assert result.ok, f"{result.stage}: {result.error} (typst={result.typst})"


def test_broken_typst_is_rejected():
    # Unbalanced delimiter should fail compilation, not crash.
    result = compile_typst("frac(a,")
    assert not result.ok
    assert result.stderr
