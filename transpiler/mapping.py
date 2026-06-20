"""Per-command mapping table: the source of truth for LaTeX -> Typst.

Each LaTeX command maps to one of a few emission strategies:

- ``Sym``    : arity-0 symbol, emitted as a fixed Typst token (e.g. ``\\alpha`` -> ``alpha``).
- ``Func``   : takes N mandatory ``{}`` args, emitted as ``name(arg1, arg2, ...)``
               (covers fractions, accents, font commands, ``\\overline`` ...).
- ``Ignore`` : spacing/no-op commands, emitted as a (possibly empty) literal token.

``\\sqrt`` (optional index arg), ``\\left``/``\\right`` delimiter pairs and
``\\begin``/``\\end`` environments are handled structurally in the parser and do
not appear in ``COMMANDS``.

Unknown commands are intentionally absent: the parser treats them as arity-0 and
the emitter raises ``TranspileError`` so the batch harness can report coverage
gaps instead of silently dropping content.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sym:
    """Arity-0 symbol emitted verbatim as a Typst token."""

    typst: str


@dataclass(frozen=True)
class Func:
    """Command emitted as a Typst function call ``typst(arg, ...)``."""

    typst: str
    arity: int


@dataclass(frozen=True)
class Ignore:
    """Spacing / no-op command emitted as a literal token (often empty)."""

    typst: str = ""


# --- Greek letters -----------------------------------------------------------
_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon.alt", "zeta": "zeta",
    "eta": "eta", "theta": "theta", "vartheta": "theta.alt", "iota": "iota",
    "kappa": "kappa", "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "xi",
    "omicron": "omicron", "pi": "pi", "varpi": "pi.alt", "rho": "rho",
    "varrho": "rho.alt", "sigma": "sigma", "varsigma": "sigma.alt",
    "tau": "tau", "upsilon": "upsilon", "phi": "phi", "varphi": "phi.alt",
    "chi": "chi", "psi": "psi", "omega": "omega",
    "Gamma": "Gamma", "Delta": "Delta", "Theta": "Theta", "Lambda": "Lambda",
    "Xi": "Xi", "Pi": "Pi", "Sigma": "Sigma", "Upsilon": "Upsilon",
    "Phi": "Phi", "Psi": "Psi", "Omega": "Omega",
}

# --- Binary operators / relations -------------------------------------------
_OPS = {
    "times": "times", "div": "div", "pm": "plus.minus", "mp": "minus.plus",
    "cdot": "dot.op", "ast": "ast", "star": "star.op", "bullet": "bullet",
    "oplus": "plus.circle", "ominus": "minus.circle", "otimes": "times.circle",
    "oslash": "div.circle", "odot": "dot.circle", "wedge": "and", "vee": "or",
    "cap": "sect", "cup": "union", "sqcap": "sect.sq", "sqcup": "union.sq",
    "setminus": "without", "uplus": "union.plus", "amalg": "product.co",
    "circ": "compose", "diamond": "diamond", "triangleleft": "triangle.l",
    "triangleright": "triangle.r", "dagger": "dagger", "ddagger": "dagger.double",
    "leq": "<=", "le": "<=", "geq": ">=", "ge": ">=", "neq": "!=", "ne": "!=",
    "equiv": "equiv", "approx": "approx", "cong": "tilde.equiv", "sim": "tilde.op",
    "simeq": "tilde.eq", "propto": "prop", "ll": "lt.double", "gg": "gt.double",
    "subset": "subset", "supset": "supset", "subseteq": "subset.eq",
    "supseteq": "supset.eq", "sqsubseteq": "subset.eq.sq", "in": "in",
    "ni": "in.rev", "notin": "in.not", "perp": "perp", "mid": "divides",
    "parallel": "parallel", "asymp": "asymp", "doteq": "eq.dot",
    "models": "models", "vdash": "tack.r", "dashv": "tack.l",
    "prec": "prec", "succ": "succ", "preceq": "prec.eq", "succeq": "succ.eq",
    "leqslant": "lt.eq.slant", "geqslant": "gt.eq.slant", "cdotp": "dot.op",
}

# --- Arrows ------------------------------------------------------------------
_ARROWS = {
    "rightarrow": "arrow.r", "to": "arrow.r", "leftarrow": "arrow.l",
    "gets": "arrow.l", "leftrightarrow": "arrow.l.r", "Rightarrow": "arrow.r.double",
    "Leftarrow": "arrow.l.double", "Leftrightarrow": "arrow.l.r.double",
    "mapsto": "arrow.r.bar", "longrightarrow": "arrow.r.long",
    "longleftarrow": "arrow.l.long", "longmapsto": "arrow.r.long.bar",
    "uparrow": "arrow.t", "downarrow": "arrow.b", "updownarrow": "arrow.t.b",
    "hookrightarrow": "arrow.r.hook", "hookleftarrow": "arrow.l.hook",
    "nearrow": "arrow.tr", "searrow": "arrow.br", "swarrow": "arrow.bl",
    "nwarrow": "arrow.tl", "rightharpoonup": "harpoon.rt",
    "leftharpoonup": "harpoon.lt", "implies": "arrow.r.double",
    "iff": "arrow.l.r.double", "Longrightarrow": "arrow.r.double.long",
    "Longleftarrow": "arrow.l.double.long",
    "Longleftrightarrow": "arrow.l.r.double.long",
    "longleftrightarrow": "arrow.l.r.long",
}

# --- Misc symbols ------------------------------------------------------------
_MISC = {
    "infty": "infinity", "partial": "diff", "nabla": "nabla", "forall": "forall",
    "exists": "exists", "nexists": "exists.not", "neg": "not", "lnot": "not",
    "emptyset": "emptyset", "varnothing": "emptyset", "aleph": "aleph",
    "hbar": "planck.reduce", "ell": "ell", "Re": "Re", "Im": "Im", "wp": "wp",
    "prime": "prime", "angle": "angle", "triangle": "triangle.stroked.t",
    "backslash": "backslash", "surd": "root.bottom", "flat": "flat",
    "sharp": "sharp", "natural": "natural", "clubsuit": "suit.club",
    "diamondsuit": "suit.diamond", "heartsuit": "suit.heart",
    "spadesuit": "suit.spade", "top": "top", "bot": "bot",
    "ldots": "dots.h", "cdots": "dots.c", "vdots": "dots.v", "ddots": "dots.down",
    "dots": "dots.h", "dotsc": "dots.h", "lozenge": "lozenge.stroked",
    "Box": "ballot", "checkmark": "checkmark", "complement": "complement",
    "land": "and", "lor": "or", "neg ": "not",
    "imath": "dotless.i", "jmath": "dotless.j", "Re ": "Re",
    "square": "square.stroked", "bigtriangledown": "triangle.stroked.b",
    "bigtriangleup": "triangle.stroked.t", "varkappa": "kappa.alt",
    "S": "section", "wp": "℘",
}

# --- Delimiter symbols usable bare (also see DELIMITERS) ----------------------
_DELIM_SYMS = {
    "langle": "chevron.l", "rangle": "chevron.r", "lfloor": "floor.l",
    "rfloor": "floor.r", "lceil": "ceil.l", "rceil": "ceil.r",
    "vert": "|", "Vert": "parallel", "lbrace": "brace.l", "rbrace": "brace.r",
    "lbrack": "[", "rbrack": "]", "|": "parallel",
}

# --- Big operators (limits attach as ordinary scripts) -----------------------
_BIG = {
    "sum": "sum", "prod": "product", "coprod": "product.co", "int": "integral",
    "iint": "integral.double", "iiint": "integral.triple", "oint": "integral.cont",
    "bigcup": "union.big", "bigcap": "sect.big", "bigsqcup": "union.sq.big",
    "bigvee": "or.big", "bigwedge": "and.big", "bigoplus": "plus.circle.big",
    "bigotimes": "times.circle.big", "bigodot": "dot.circle.big",
}

# --- Named operators (Typst recognises these identifiers) --------------------
_NAMED_OPS = [
    "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth",
    "arcsin", "arccos", "arctan", "log", "ln", "lg", "exp", "lim", "limsup",
    "liminf", "max", "min", "sup", "inf", "det", "deg", "dim", "ker", "hom",
    "arg", "gcd", "Pr", "mod", "bmod", "pmod",
]

# --- Functions: fractions, roots-as-func, accents, fonts, over/under ----------
_FUNCS = {
    "frac": Func("frac", 2), "dfrac": Func("frac", 2), "tfrac": Func("frac", 2),
    "binom": Func("binom", 2),
    "overline": Func("overline", 1), "underline": Func("underline", 1),
    "overbrace": Func("overbrace", 1), "underbrace": Func("underbrace", 1),
    "overrightarrow": Func("arrow", 1), "vec": Func("arrow", 1),
    # accents
    "hat": Func("hat", 1), "widehat": Func("hat", 1), "tilde": Func("tilde", 1),
    "widetilde": Func("tilde", 1), "bar": Func("macron", 1), "dot": Func("dot", 1),
    "ddot": Func("dot.double", 1), "acute": Func("acute", 1),
    "grave": Func("grave", 1), "breve": Func("breve", 1), "check": Func("caron", 1),
    "mathring": Func("circle", 1),
    # font / style commands
    "mathrm": Func("upright", 1), "mathbf": Func("bold", 1),
    "boldsymbol": Func("bold", 1), "bm": Func("bold", 1),
    "mathbb": Func("bb", 1), "mathcal": Func("cal", 1), "mathfrak": Func("frak", 1),
    "mathsf": Func("sans", 1), "mathtt": Func("mono", 1), "mathit": Func("italic", 1),
    "operatorname": Func("upright", 1), "text": Func("upright", 1),
    "textrm": Func("upright", 1), "textbf": Func("bold", 1), "mbox": Func("upright", 1),
    "mathscr": Func("cal", 1), "mathop": Func("upright", 1), "fbox": Func("upright", 1),
    # arg consumed but emitted as nothing (reserves space in LaTeX; we only
    # care about compilability) -- see DROP_ARG / ATTACH_* in the emitter
    "phantom": Func("hide", 1), "vphantom": Func("hide", 1), "hphantom": Func("hide", 1),
    # script-stacking: top/bottom over a base, emitted via Typst ``attach``
    "stackrel": Func("attach", 2), "overset": Func("attach", 2),
    "underset": Func("attach", 2),
}

# Commands whose mandatory argument is consumed but emitted as empty.
DROP_ARG: frozenset[str] = frozenset({"phantom", "vphantom", "hphantom"})

# Script-stacking commands -> ``attach(base, <side>: script)``.
# LaTeX order is {script}{base}; ``underset`` places the script below.
ATTACH_TOP: frozenset[str] = frozenset({"stackrel", "overset"})
ATTACH_BOTTOM: frozenset[str] = frozenset({"underset"})

# --- Spacing / no-op commands ------------------------------------------------
_IGNORE = {
    ",": Ignore("thin"), ":": Ignore("med"), ";": Ignore("med"),
    "!": Ignore(""), " ": Ignore("space"), "quad": Ignore("quad"),
    "qquad": Ignore("wide"), "thinspace": Ignore("thin"), "nonumber": Ignore(""),
    "limits": Ignore(""), "nolimits": Ignore(""), "displaystyle": Ignore(""),
    "textstyle": Ignore(""), "scriptstyle": Ignore(""), "left.": Ignore(""),
    "scriptscriptstyle": Ignore(""), "enspace": Ignore("space"),
    "thickspace": Ignore("thick"), "negthinspace": Ignore(""), "mit": Ignore(""),
    # old-style font switches: drop the switch, keep the content
    "rm": Ignore(""), "bf": Ignore(""), "it": Ignore(""), "sf": Ignore(""),
    "tt": Ignore(""), "cal": Ignore(""), "sc": Ignore(""), "sl": Ignore(""),
    "bm ": Ignore(""),
    # font-size switches
    "tiny": Ignore(""), "scriptsize": Ignore(""), "footnotesize": Ignore(""),
    "small": Ignore(""), "normalsize": Ignore(""), "large": Ignore(""),
    "Large": Ignore(""), "LARGE": Ignore(""), "huge": Ignore(""), "Huge": Ignore(""),
    # manual delimiter sizing: drop, the delimiter still renders
    "big": Ignore(""), "Big": Ignore(""), "bigg": Ignore(""), "Bigg": Ignore(""),
    "bigl": Ignore(""), "Bigl": Ignore(""), "bigr": Ignore(""), "Bigr": Ignore(""),
    "biggl": Ignore(""), "biggr": Ignore(""), "Biggl": Ignore(""), "Biggr": Ignore(""),
    "bigm": Ignore(""), "Bigm": Ignore(""), "biggm": Ignore(""),
    # misc no-ops in math
    "hline": Ignore(""), "tag": Ignore(""), "notag": Ignore(""), "not": Ignore(""),
    "protect": Ignore(""), "vphantom": Ignore(""),
}


# --- Backslash-escaped literals (control symbols) ----------------------------
_ESCAPED = {
    "%": '"%"', "#": '"#"', "&": '"&"', "$": '"$"', "_": "_",
    "{": "\\{", "}": "\\}",
}


def _build() -> dict[str, Sym | Func | Ignore]:
    table: dict[str, Sym | Func | Ignore] = {}
    for group in (_GREEK, _OPS, _ARROWS, _MISC, _DELIM_SYMS, _BIG, _ESCAPED):
        for name, typ in group.items():
            table[name] = Sym(typ)
    for name in _NAMED_OPS:
        table[name] = Sym(name)
    table.update(_FUNCS)
    table.update(_IGNORE)
    return table


COMMANDS: dict[str, Sym | Func | Ignore] = _build()


# Delimiter token (char or command name) -> Typst delimiter token.
# "." is the LaTeX null delimiter and maps to an empty string.
DELIMITERS: dict[str, str] = {
    "(": "(", ")": ")", "[": "[", "]": "]", "/": "\\/", "|": "|",
    "{": "\\{", "}": "\\}", ".": "", "<": "chevron.l", ">": "chevron.r",
    "langle": "chevron.l", "rangle": "chevron.r", "lfloor": "floor.l",
    "rfloor": "floor.r", "lceil": "ceil.l", "rceil": "ceil.r",
    "lbrace": "brace.l", "rbrace": "brace.r", "lbrack": "[", "rbrack": "]",
    "vert": "|", "Vert": "parallel", "|_": "floor.r", "backslash": "\\\\",
}

# Matrix-style environments -> Typst ``mat`` delim option (None = no delimiter).
MATRIX_DELIMS: dict[str, str | None] = {
    "matrix": "#none", "smallmatrix": "#none", "array": "#none",
    "pmatrix": '"("', "bmatrix": '"["', "Bmatrix": '"{"',
    "vmatrix": '"|"', "Vmatrix": '"||"',
}

# Characters that are special in Typst math and must be emitted as strings.
CHAR_MAP: dict[str, str] = {
    "%": '"%"', "#": '"#"', "&": '"&"', '"': '"\\""',
}


def arity(name: str) -> int:
    """Mandatory ``{}`` arguments the parser must consume for ``\\name``.

    Unknown commands return 0 so the parser keeps going; the emitter is what
    flags them as unsupported.
    """
    entry = COMMANDS.get(name)
    if isinstance(entry, Func):
        return entry.arity
    return 0


def is_known(name: str) -> bool:
    return name in COMMANDS


def delimiter(token: str) -> str | None:
    """Typst delimiter for a ``\\left``/``\\right`` token, or None if unknown."""
    return DELIMITERS.get(token)
