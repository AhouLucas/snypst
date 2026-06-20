"""Recursive-descent parser: token stream -> AST.

Handles explicit groups, sub/superscripts (attached to the preceding atom),
``\\left``/``\\right`` delimiter pairs, ``\\sqrt`` with an optional index, and
``\\begin``/``\\end`` environments (matrices, cases, alignment) split into rows
and cells.

Arity for ordinary commands comes from :mod:`transpiler.mapping`; unknown
commands are parsed as arity-0 and flagged later by the emitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from transpiler import mapping
from transpiler.tokenizer import Token, tokenize


class ParseError(Exception):
    pass


# --- AST nodes ---------------------------------------------------------------
@dataclass
class Seq:
    """An ordered run of nodes (whole expression, group body, or a cell)."""

    items: list["Node"]


@dataclass
class Char:
    c: str


@dataclass
class Cmd:
    name: str
    args: list[Seq]


@dataclass
class Group:
    body: Seq


@dataclass
class Script:
    base: "Node | None"
    sub: "Node | None"
    sup: "Node | None"


@dataclass
class Sqrt:
    index: Seq | None
    radicand: Seq


@dataclass
class Delim:
    left: str  # raw LaTeX delimiter token
    right: str
    body: Seq


@dataclass
class Env:
    name: str
    rows: list[list[Seq]]


Node = Char | Cmd | Group | Script | Sqrt | Delim | Env | Seq


_OPENERS = frozenset("([")


def _is_opener(node: "Node") -> bool:
    """Open delimiters can't carry scripts; ``(^{*}x)`` is a prescript, not ``(^``."""
    return isinstance(node, Char) and node.c in _OPENERS


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.toks = tokens
        self.i = 0

    # -- low-level cursor --
    def _peek(self) -> Token | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self) -> Token:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    # -- entry point --
    def parse(self) -> Seq:
        seq = self._parse_seq(lambda t: False)
        if self._peek() is not None:
            raise ParseError(f"unexpected token {self._peek()!r}")
        return seq

    # -- sequences --
    def _parse_seq(self, is_stop: Callable[[Token], bool]) -> Seq:
        items: list[Node] = []
        while True:
            tok = self._peek()
            if tok is None or is_stop(tok):
                break
            if tok.kind in ("sub", "sup"):
                base = items.pop() if (items and not _is_opener(items[-1])) else None
                items.append(self._parse_scripts(base))
                continue
            atom = self._parse_atom()
            nxt = self._peek()
            if nxt is not None and nxt.kind in ("sub", "sup") and not _is_opener(atom):
                atom = self._parse_scripts(atom)
            items.append(atom)
        return Seq(items)

    def _parse_scripts(self, base: Node | None) -> Script:
        sub: Node | None = None
        sup: Node | None = None
        while True:
            tok = self._peek()
            if tok is None or tok.kind not in ("sub", "sup"):
                break
            kind = self._next().kind
            arg = self._parse_atom()
            if kind == "sub":
                sub = arg
            else:
                sup = arg
        return Script(base, sub, sup)

    # -- atoms --
    def _parse_atom(self) -> Node:
        tok = self._next()
        if tok.kind == "char":
            return Char(tok.value)
        if tok.kind == "lbrace":
            body = self._parse_seq(lambda t: t.kind == "rbrace")
            self._expect("rbrace")
            return Group(body)
        if tok.kind == "cmd":
            return self._parse_command(tok.value)
        raise ParseError(f"unexpected token {tok!r}")

    def _parse_command(self, name: str) -> Node:
        if name == "left":
            return self._parse_leftright()
        if name == "begin":
            return self._parse_env()
        if name == "sqrt":
            return self._parse_sqrt()
        if name in ("right", "end"):
            raise ParseError(f"stray \\{name}")
        n_args = mapping.arity(name)
        args = [self._parse_required_arg() for _ in range(n_args)]
        return Cmd(name, args)

    def _parse_required_arg(self) -> Seq:
        tok = self._peek()
        if tok is None:
            raise ParseError("missing argument")
        if tok.kind == "lbrace":
            self._next()
            body = self._parse_seq(lambda t: t.kind == "rbrace")
            self._expect("rbrace")
            return body
        return Seq([self._parse_atom()])

    def _parse_sqrt(self) -> Sqrt:
        index: Seq | None = None
        tok = self._peek()
        if tok is not None and tok.kind == "char" and tok.value == "[":
            self._next()
            index = self._parse_seq(
                lambda t: t.kind == "char" and t.value == "]"
            )
            self._expect_char("]")
        radicand = self._parse_required_arg()
        return Sqrt(index, radicand)

    def _parse_leftright(self) -> Delim:
        left = self._parse_delim_token()
        body = self._parse_seq(lambda t: t.kind == "cmd" and t.value == "right")
        self._expect_cmd("right")
        right = self._parse_delim_token()
        return Delim(left, right, body)

    def _parse_delim_token(self) -> str:
        tok = self._peek()
        if tok is None:
            raise ParseError("missing delimiter")
        self._next()
        if tok.kind == "cmd":
            return tok.value
        return tok.value

    def _parse_env(self) -> Env:
        name = self._read_group_text()
        if name == "array":
            # discard the column-spec argument, e.g. {ccc}
            if self._peek() is not None and self._peek().kind == "lbrace":
                self._read_group_text()

        def is_stop(t: Token) -> bool:
            return t.kind in ("amp", "rowsep") or (t.kind == "cmd" and t.value == "end")

        rows: list[list[Seq]] = []
        row: list[Seq] = []
        while True:
            cell = self._parse_seq(is_stop)
            row.append(cell)
            tok = self._peek()
            if tok is None:
                raise ParseError(f"unterminated environment {name!r}")
            if tok.kind == "amp":
                self._next()
                continue
            if tok.kind == "rowsep":
                self._next()
                rows.append(row)
                row = []
                continue
            # cmd 'end'
            self._next()
            self._read_group_text()  # consume {name}
            break
        if row and (len(row) > 1 or row[0].items):
            rows.append(row)
        return Env(name, rows)

    def _read_group_text(self) -> str:
        self._expect("lbrace")
        chars: list[str] = []
        while True:
            tok = self._peek()
            if tok is None:
                raise ParseError("unterminated group")
            if tok.kind == "rbrace":
                self._next()
                break
            self._next()
            chars.append(tok.value)
        return "".join(chars)

    # -- expectation helpers --
    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            raise ParseError(f"expected {kind}, got {tok!r}")
        return self._next()

    def _expect_cmd(self, name: str) -> None:
        tok = self._peek()
        if tok is None or tok.kind != "cmd" or tok.value != name:
            raise ParseError(f"expected \\{name}, got {tok!r}")
        self._next()

    def _expect_char(self, value: str) -> None:
        tok = self._peek()
        if tok is None or tok.kind != "char" or tok.value != value:
            raise ParseError(f"expected {value!r}, got {tok!r}")
        self._next()


def parse(latex: str) -> Seq:
    return Parser(tokenize(latex)).parse()
