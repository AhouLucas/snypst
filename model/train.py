"""Train the format-conditioned model; defaults to an overfit smoke test.

The smoke test trains on a few hundred examples and confirms the whole vertical
slice works: loss collapses, greedy decode round-trips *both* formats, and the
decoded Typst actually compiles (reused from the transpiler's validator). This
is the gate before any full-scale run.

    uv run python -m model.train --smoke --limit 200 --epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.data import (
    ImageConfig,
    MarkupDataset,
    collate,
    expand_examples,
    load_manifest,
)
from model.model import ModelConfig, Seq2Seq
from model.vocab import Vocab, build

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "data" / "processed" / "manifest.jsonl"
DEFAULT_CKPT = REPO / "model" / "checkpoints" / "model.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset: MarkupDataset, batch_size: int, shuffle: bool) -> DataLoader:
    pad = dataset.vocab.pad_id
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate(b, pad),
    )


def run_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        tgt_in, tgt_out = target[:, :-1], target[:, 1:]
        logits = model(image, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * image.size(0)
        count += image.size(0)
    return total / max(count, 1)


def smoke_check(model: Seq2Seq, dataset: MarkupDataset, device: torch.device, n: int = 6) -> None:
    """Greedy-decode a few examples for each format and compile the Typst."""
    from transpiler.validator import compile_typst, typst_available

    vocab = dataset.vocab
    can_compile = typst_available()
    typst_ok = typst_total = 0
    exact = 0
    print("\n=== smoke decode ===")
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        ex = dataset.examples[i]
        image = item["image"].unsqueeze(0).to(device)
        ids = model.greedy_decode(
            image, vocab.format_id(ex.fmt), vocab.bos_id, vocab.eos_id
        )
        pred = vocab.decode(ids)
        ok = pred == ex.target
        exact += ok
        mark = "OK " if ok else "XX "
        print(f"[{ex.fmt:5}] {mark} gold={ex.target!r}")
        if not ok:
            print(f"          pred={pred!r}")
        if ex.fmt == "typst" and can_compile:
            typst_total += 1
            if compile_typst(pred).ok:
                typst_ok += 1
    print(f"exact-match {exact}/{min(n, len(dataset))}", end="")
    if typst_total:
        print(f" | typst compiles {typst_ok}/{typst_total}")
    else:
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--smoke", action="store_true", help="overfit a small subset")
    ap.add_argument("--limit", type=int, default=None, help="cap number of manifest rows")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    examples = expand_examples(rows)
    print(f"{len(rows)} rows -> {len(examples)} examples")

    vocab = build([r["latex"] for r in rows] + [r["typst"] for r in rows])
    print(f"vocab size: {len(vocab)}")
    img_cfg = ImageConfig()
    dataset = MarkupDataset(examples, vocab, img_cfg)
    loader = make_loader(dataset, args.batch_size, shuffle=True)

    cfg = ModelConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        num_decoder_layers=args.layers,
        pad_id=vocab.pad_id,
    )
    model = Seq2Seq(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, loader, criterion, device, optimizer)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  loss {loss:.4f}")

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "vocab": vocab.itos}, args.ckpt)
    vocab.save(args.ckpt.with_name("vocab.json"))
    print(f"saved checkpoint -> {args.ckpt.relative_to(REPO)}")

    if args.smoke:
        smoke_check(model, dataset, device)


if __name__ == "__main__":
    main()
