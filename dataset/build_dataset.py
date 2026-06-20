"""Build (image, latex, typst) triples into a JSONL manifest.

Each candidate's LaTeX is transpiled and gated by the typst compiler; only
compilable records are written. The manifest is the single source of truth for
training, with one row per image carrying both target formats:

    {"image", "latex", "typst", "source", "split", "license"}

im2latex images are referenced in place; MathWriting inks are rasterised to
``<out>/handwritten/<id>.png``. Use ``--limit`` (or the per-source limits) to
build a subset for pipeline validation before a full-scale run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dataset.inkml import parse_inkml
from dataset.render import RenderConfig, render_to_png
from transpiler.validator import ValidateResult, typst_available, validate

REPO = Path(__file__).resolve().parent.parent
IM2LATEX = REPO / "data" / "raw" / "im2latex"
MATHWRITING = REPO / "data" / "raw" / "handwritten-raw"
DEFAULT_OUT = REPO / "data" / "processed"

# im2latex provenance is permissive; MathWriting is research/non-commercial.
LICENSE = {"im2latex": "unknown", "mathwriting": "CC-BY-NC-SA-4.0"}


@dataclass
class Record:
    image: str
    latex: str
    typst: str
    source: str
    split: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "image": self.image,
                "latex": self.latex,
                "typst": self.typst,
                "source": self.source,
                "split": self.split,
                "license": LICENSE[self.source],
            },
            ensure_ascii=False,
        )


@dataclass
class Stats:
    seen: int = 0
    passed: int = 0
    transpile_fail: int = 0
    compile_fail: int = 0
    missing_image: int = 0

    def record(self, result: ValidateResult) -> None:
        self.seen += 1
        if result.ok:
            self.passed += 1
        elif result.stage == "transpile":
            self.transpile_fail += 1
        else:
            self.compile_fail += 1


def _hash_split(key: str, ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> str:
    """Deterministic train/valid/test assignment from a stable key."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < ratios[0]:
        return "train"
    if h < ratios[0] + ratios[1]:
        return "valid"
    return "test"


# --- im2latex ---------------------------------------------------------------
def _im2latex_candidates(limit: int) -> list[tuple[str, str]]:
    formulas = IM2LATEX / "final_png_formulas.txt"
    images = IM2LATEX / "corresponding_png_images.txt"
    out: list[tuple[str, str]] = []
    with formulas.open(encoding="utf-8") as ff, images.open(encoding="utf-8") as fi:
        for latex, image_name in zip(ff, fi):
            if len(out) >= limit:
                break
            out.append((latex.strip(), image_name.strip()))
    return out


def build_im2latex(limit: int, workers: int, stats: Stats) -> list[Record]:
    img_dir = IM2LATEX / "generated_png_images"
    candidates = _im2latex_candidates(limit)

    def work(item: tuple[str, str]) -> Record | None:
        latex, image_name = item
        image_path = img_dir / image_name
        if not image_path.exists():
            stats.missing_image += 1
            return None
        result = validate(latex)
        stats.record(result)
        if not result.ok or result.typst is None:
            return None
        return Record(
            image=str(image_path.relative_to(REPO)),
            latex=latex,
            typst=result.typst,
            source="im2latex",
            split=_hash_split(image_name),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(work, candidates) if r is not None]


# --- MathWriting ------------------------------------------------------------
def _inkml_paths(split_dir: Path, limit: int) -> list[Path]:
    paths: list[Path] = []
    with os.scandir(split_dir) as it:
        for entry in it:
            if len(paths) >= limit:
                break
            if entry.name.endswith(".inkml"):
                paths.append(Path(entry.path))
    return paths


def build_mathwriting(
    limits: dict[str, int], workers: int, out_dir: Path, stats: Stats
) -> list[Record]:
    cfg = RenderConfig()
    png_dir = out_dir / "handwritten"

    def work(args: tuple[Path, str]) -> Record | None:
        path, split = args
        ink = parse_inkml(path)
        if not ink.label or not ink.strokes:
            return None
        result = validate(ink.label)
        stats.record(result)
        if not result.ok or result.typst is None:
            return None
        png = png_dir / split / f"{ink.sample_id}.png"
        render_to_png(ink.strokes, png, cfg)
        return Record(
            image=str(png.relative_to(REPO)),
            latex=ink.label,
            typst=result.typst,
            source="mathwriting",
            split=split,
        )

    tasks: list[tuple[Path, str]] = []
    for split, limit in limits.items():
        if limit <= 0:
            continue
        for path in _inkml_paths(MATHWRITING / split, limit):
            tasks.append((path, split))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(work, tasks) if r is not None]


# --- orchestration ----------------------------------------------------------
def _report(name: str, stats: Stats) -> None:
    n = stats.seen or 1
    print(f"\n  [{name}] seen={stats.seen} pass={stats.passed} "
          f"({100 * stats.passed / n:.1f}%) "
          f"transpile_fail={stats.transpile_fail} compile_fail={stats.compile_fail} "
          f"missing_image={stats.missing_image}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="convenience: sets both per-source train limits")
    ap.add_argument("--im2latex-limit", type=int, default=4000)
    ap.add_argument("--hw-train-limit", type=int, default=1500)
    ap.add_argument("--hw-eval-limit", type=int, default=200,
                    help="inks pulled from each of valid/ and test/")
    ap.add_argument("--sources", nargs="+", default=["im2latex", "mathwriting"],
                    choices=["im2latex", "mathwriting"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.limit is not None:
        args.im2latex_limit = args.limit
        args.hw_train_limit = args.limit

    if not typst_available():
        raise SystemExit("typst binary not found on PATH")
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[Record] = []
    if "im2latex" in args.sources:
        s = Stats()
        records += build_im2latex(args.im2latex_limit, args.workers, s)
        _report("im2latex", s)
    if "mathwriting" in args.sources:
        s = Stats()
        limits = {
            "train": args.hw_train_limit,
            "valid": args.hw_eval_limit,
            "test": args.hw_eval_limit,
        }
        records += build_mathwriting(limits, args.workers, args.out, s)
        _report("mathwriting", s)

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")

    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for rec in records:
        split_counts[rec.split] = split_counts.get(rec.split, 0) + 1
        source_counts[rec.source] = source_counts.get(rec.source, 0) + 1

    print(f"\n=== Wrote {len(records)} triples -> {manifest.relative_to(REPO)} ===")
    print(f"  by source: {source_counts}")
    print(f"  by split : {split_counts}")


if __name__ == "__main__":
    main()
