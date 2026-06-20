"""Evaluate a trained checkpoint on a manifest split.

Reports, broken down **per format** (latex/typst) and **per source**
(im2latex/mathwriting):
  - exact-match rate (decoded string == gold)
  - character-level accuracy (alignment-free, length-normalised)
  - Typst **compile pass-rate** — the hard metric — on decoded Typst.

    uv run python -m eval.evaluate --split test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from model.data import ImageConfig, load_image
from model.model import ModelConfig, Seq2Seq
from model.train import DEFAULT_CKPT, DEFAULT_MANIFEST, pick_device
from model.vocab import Vocab
from transpiler.validator import compile_typst, typst_available


def _char_acc(pred: str, gold: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    n = min(len(pred), len(gold))
    correct = sum(p == g for p, g in zip(pred[:n], gold[:n]))
    return correct / max(len(pred), len(gold))


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[Seq2Seq, Vocab]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: ModelConfig = blob["cfg"]
    vocab = Vocab(blob["vocab"])
    model = Seq2Seq(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, vocab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model, vocab = load_checkpoint(args.ckpt, device)
    img_cfg = ImageConfig()
    can_compile = typst_available()

    import json

    rows = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and json.loads(line)["split"] == args.split:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"evaluating {len(rows)} rows from split={args.split}")

    # keyed by (source, fmt)
    n = defaultdict(int)
    exact = defaultdict(int)
    char = defaultdict(float)
    compiles = defaultdict(int)
    compile_n = defaultdict(int)

    for r in rows:
        image = load_image(r["image"], img_cfg).unsqueeze(0).to(device)
        for fmt in ("latex", "typst"):
            gold = r[fmt]
            ids = model.greedy_decode(
                image, vocab.format_id(fmt), vocab.bos_id, vocab.eos_id, args.max_len
            )
            pred = vocab.decode(ids)
            key = (r["source"], fmt)
            n[key] += 1
            exact[key] += pred == gold
            char[key] += _char_acc(pred, gold)
            if fmt == "typst" and can_compile:
                compile_n[key] += 1
                compiles[key] += compile_typst(pred).ok

    print(f"\n{'source':12} {'fmt':6} {'N':>5} {'exact':>7} {'char':>7} {'compile':>9}")
    for key in sorted(n):
        source, fmt = key
        cnt = n[key]
        comp = (
            f"{compiles[key]}/{compile_n[key]}" if compile_n[key] else "-"
        )
        print(
            f"{source:12} {fmt:6} {cnt:5d} {exact[key]/cnt:7.2%} "
            f"{char[key]/cnt:7.2%} {comp:>9}"
        )


if __name__ == "__main__":
    main()
