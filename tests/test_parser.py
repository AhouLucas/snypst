import pytest

from transpiler.parser import (
    Char,
    Cmd,
    Delim,
    Env,
    Group,
    ParseError,
    Script,
    Sqrt,
    parse,
)


def test_command_arity():
    seq = parse(r"\frac{a}{b}")
    (node,) = seq.items
    assert isinstance(node, Cmd)
    assert node.name == "frac"
    assert len(node.args) == 2


def test_script_attaches_to_previous_atom():
    seq = parse("x^2")
    (node,) = seq.items
    assert isinstance(node, Script)
    assert isinstance(node.base, Char) and node.base.c == "x"
    assert isinstance(node.sup, Char) and node.sup.c == "2"


def test_sub_and_sup_combined():
    (node,) = parse("x_i^2").items
    assert isinstance(node, Script)
    assert node.sub is not None and node.sup is not None


def test_unknown_command_parses_as_arity_zero():
    # Parser stays permissive; the emitter is what rejects unknown commands.
    seq = parse(r"\fakecmd x")
    assert isinstance(seq.items[0], Cmd)
    assert seq.items[0].name == "fakecmd"
    assert seq.items[0].args == []


def test_sqrt_optional_index():
    (node,) = parse(r"\sqrt[3]{x}").items
    assert isinstance(node, Sqrt)
    assert node.index is not None


def test_sqrt_plain():
    (node,) = parse(r"\sqrt{x}").items
    assert isinstance(node, Sqrt)
    assert node.index is None


def test_leftright_delimiters():
    (node,) = parse(r"\left( a \right)").items
    assert isinstance(node, Delim)
    assert node.left == "(" and node.right == ")"


def test_environment_rows_and_cells():
    (node,) = parse(r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}").items
    assert isinstance(node, Env)
    assert node.name == "pmatrix"
    assert len(node.rows) == 2
    assert len(node.rows[0]) == 2


def test_group_node():
    (node,) = parse("{ab}").items
    assert isinstance(node, Group)


def test_stray_right_raises():
    with pytest.raises(ParseError):
        parse(r"a \right)")
