"""Manifest -> training examples for the format-conditioned model.

Each manifest row carries one image and *both* target formats, so it expands to
up to two examples: ``(image, [LATEX]->latex)`` and ``(image, [TYPST]->typst)``.
Images come from two very different sources (clean printed im2latex vs.
rasterised handwriting); both are letterboxed onto a fixed grayscale canvas so
the encoder sees a constant input size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from model.vocab import Vocab

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ImageConfig:
    height: int = 128
    width: int = 512


@dataclass
class Example:
    image: str  # repo-relative path
    fmt: str  # "latex" | "typst"
    target: str


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_examples(rows: list[dict], formats: tuple[str, ...] = ("latex", "typst")) -> list[Example]:
    examples: list[Example] = []
    for r in rows:
        for fmt in formats:
            examples.append(Example(image=r["image"], fmt=fmt, target=r[fmt]))
    return examples


def load_image(path: str | Path, cfg: ImageConfig) -> torch.Tensor:
    """Load grayscale, letterbox onto a fixed (H, W) white canvas, return (1,H,W) in [0,1]."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    img = Image.open(p).convert("L")
    w, h = img.size
    scale = min(cfg.width / w, cfg.height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (cfg.width, cfg.height), color=255)
    canvas.paste(img, ((cfg.width - new_w) // 2, (cfg.height - new_h) // 2))
    t = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float() / 255.0
    return t.view(1, cfg.height, cfg.width)


class MarkupDataset(Dataset):
    def __init__(self, examples: list[Example], vocab: Vocab, img_cfg: ImageConfig) -> None:
        self.examples = examples
        self.vocab = vocab
        self.img_cfg = img_cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        image = load_image(ex.image, self.img_cfg)
        v = self.vocab
        # Decoder sequence: [FORMAT] <S> tokens... <E>
        target_ids = [v.format_id(ex.fmt), v.bos_id, *v.encode(ex.target), v.eos_id]
        return {"image": image, "target": torch.tensor(target_ids, dtype=torch.long)}


def collate(batch: list[dict], pad_id: int) -> dict:
    images = torch.stack([b["image"] for b in batch])
    targets = [b["target"] for b in batch]
    max_len = max(t.size(0) for t in targets)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, : t.size(0)] = t
    return {"image": images, "target": padded}
