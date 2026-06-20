import pytest

from transpiler import transpile
from transpiler.emitter import TranspileError


def test_subscript_grouping():
    assert transpile(r"R _ { 1 2 }") == "R_(1 2)"


def test_single_atom_script_not_parenthesised():
    assert transpile("x_i") == "x_i"


def test_frac():
    assert transpile(r"\frac{a}{b}") == "frac(a, b)"


def test_sqrt_and_root():
    assert transpile(r"\sqrt{x}") == "sqrt(x)"
    assert transpile(r"\sqrt[3]{x}") == "root(3, x)"


def test_greek_and_relation_symbols():
    assert transpile(r"\alpha \leq \beta") == "alpha <= beta"


def test_letter_runs_stay_separate():
    # Must not collapse into the single Typst identifier "RK".
    assert transpile("R K") == "R K"


def test_delimiter_pair():
    assert transpile(r"\left( a \right)") == "lr(( a ))"


def test_matrix():
    out = transpile(r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}")
    assert out == 'mat(delim: "(", a, b; c, d)'


def test_accent_and_font():
    assert transpile(r"\hat{x}") == "hat(x)"
    assert transpile(r"\mathbf{v}") == "bold(v)"


def test_special_char_escaped():
    assert transpile(r"\%") == '"%"'


def test_brace_delimiters():
    # Braces must be escaped in Typst math or they open an invisible group.
    assert transpile(r"\left\{ x \right.") == "lr(\\{ x)"


def test_empty_base_script_flattens():
    # Nested/prescript forms must not produce invalid Typst (^^ or __).
    assert transpile(r"\rho ^ { ^ { \prime } }") == "rho^prime"
    assert transpile(r"L _ { _ \mathrm{M} }") == "L_upright(M)"


def test_script_after_open_bracket_is_prescript():
    assert transpile(r"( ^ { * } F )") == "( * F )"


def test_stackrel_uses_attach():
    assert transpile(r"\stackrel{def}{=}") == "attach(=, t: d e f)"


def test_phantom_drops_content():
    assert transpile(r"\phantom{xy} z") == "z"


def test_unsupported_command_raises():
    with pytest.raises(TranspileError):
        transpile(r"\someunknownmacro{x}")
