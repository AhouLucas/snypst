from transpiler.tokenizer import Token, tokenize


def test_space_tokenized_form():
    toks = tokenize(r"R _ { 1 2 }")
    assert toks == [
        Token("char", "R"),
        Token("sub", "_"),
        Token("lbrace", "{"),
        Token("char", "1"),
        Token("char", "2"),
        Token("rbrace", "}"),
    ]


def test_compact_form_matches_spaced():
    assert tokenize(r"\frac{a}{b}") == tokenize(r"\frac { a } { b }")


def test_control_word_vs_control_symbol():
    toks = tokenize(r"\alpha \, \!")
    assert toks == [Token("cmd", "alpha"), Token("cmd", ","), Token("cmd", "!")]


def test_rowsep_and_amp():
    toks = tokenize(r"a & b \\ c")
    kinds = [t.kind for t in toks]
    assert kinds == ["char", "amp", "char", "rowsep", "char"]


def test_scripts():
    assert tokenize("x^2_n") == [
        Token("char", "x"),
        Token("sup", "^"),
        Token("char", "2"),
        Token("sub", "_"),
        Token("char", "n"),
    ]
