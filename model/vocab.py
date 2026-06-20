"""Shared character-level vocabulary for both target formats.

Char-level is the simplest tokenization that works across Typst and LaTeX
without a hand-built token table, and the target strings are short. The vocab
spans every character seen in the manifest's ``latex`` and ``typst`` fields,
plus the specials and the two format-conditioning tokens that are prepended at
decoder start (``[LATEX]`` / ``[TYPST]``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

# Reserved ids 0..4. Format tokens get the next two ids in build().
PAD = "<P>"
BOS = "<S>"
EOS = "<E>"
UNK = "<U>"
SPECIALS = [PAD, BOS, EOS, UNK]

FORMAT_TOKEN = {"latex": "[LATEX]", "typst": "[TYPST]"}


class Vocab:
    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def format_id(self, fmt: str) -> int:
        return self.stoi[FORMAT_TOKEN[fmt]]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Characters -> ids (no specials added)."""
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Ids -> string, stopping at EOS and dropping specials/format tokens."""
        out: list[str] = []
        special = set(SPECIALS) | set(FORMAT_TOKEN.values())
        for i in ids:
            tok = self.itos[i]
            if tok == EOS:
                break
            if tok in special:
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build(texts: Iterable[str]) -> Vocab:
    """Build a vocab: specials, format tokens, then sorted unique characters."""
    chars: set[str] = set()
    for t in texts:
        chars.update(t)
    itos = list(SPECIALS)
    itos += [FORMAT_TOKEN["latex"], FORMAT_TOKEN["typst"]]
    itos += sorted(chars)
    return Vocab(itos)
