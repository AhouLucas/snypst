"""LaTeX (math-mode) string -> flat token stream.

A small state machine. Whitespace is *not* significant in math mode beyond
terminating a control word, so it is dropped. This means both the
space-tokenized im2latex form (``R _ { 1 2 }``) and the compact MathWriting form
(``\\frac{dy}{dt}``) produce the same token stream.

Token kinds:

- ``cmd``    : a control sequence, value is the name without the backslash
               (``frac``, ``alpha``); control symbols carry their single char
               (``\\,`` -> ``cmd ','``).
- ``rowsep`` : ``\\\\`` (environment row separator).
- ``lbrace`` / ``rbrace`` : ``{`` / ``}``.
- ``sub`` / ``sup`` : ``_`` / ``^``.
- ``amp``    : ``&`` (environment cell separator).
- ``char``   : any other single character.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


_SINGLE = {"{": "lbrace", "}": "rbrace", "_": "sub", "^": "sup", "&": "amp"}


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "\\":
            j = i + 1
            if j < n and src[j] == "\\":
                tokens.append(Token("rowsep", "\\\\"))
                i = j + 1
                continue
            if j < n and src[j].isalpha():
                k = j
                while k < n and src[k].isalpha():
                    k += 1
                tokens.append(Token("cmd", src[j:k]))
                i = k
                continue
            if j < n:
                # control symbol: backslash + single non-letter char
                tokens.append(Token("cmd", src[j]))
                i = j + 1
                continue
            # trailing lone backslash: ignore
            i = j
            continue
        kind = _SINGLE.get(c)
        if kind is not None:
            tokens.append(Token(kind, c))
            i += 1
            continue
        tokens.append(Token("char", c))
        i += 1
    return tokens
