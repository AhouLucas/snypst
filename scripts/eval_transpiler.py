#!/usr/bin/env python
"""Batch transpiler coverage / compile pass-rate harness.

Runs a sample of real im2latex formulas through the transpiler and the typst
compiler, reporting:

  * totals: transpiled, compile-pass, compile-fail, transpile-fail
  * a histogram of the LaTeX commands implicated in failures (the prioritised
    to-do list for growing ``mapping.py``)
  * a handful of failing examples per stage

Usage:
    uv run python scripts/eval_transpiler.py [--limit N] [--workers K]
                                             [--formulas PATH] [--show M]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from transpiler.validator import typst_available, validate

DEFAULT_FORMULAS = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "im2latex" / "final_png_formulas.txt"
)

_CMD_RE = re.compile(r"\\([a-zA-Z]+)")


@dataclass
class Outcome:
    latex: str
    ok: bool
    stage: str
    typst: str | None
    error: str


def _commands_in(latex: str) -> set[str]:
    return set(_CMD_RE.findall(latex))


def run(latex: str) -> Outcome:
    res = validate(latex)
    return Outcome(latex, res.ok, res.stage, res.typst, res.error)


def _unsupported_command(error: str) -> str | None:
    m = re.search(r"unsupported command \\([a-zA-Z]+)", error)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    ap.add_argument("--show", type=int, default=5, help="examples per failure stage")
    args = ap.parse_args()

    if not typst_available():
        raise SystemExit("typst binary not found on PATH")
    if not args.formulas.exists():
        raise SystemExit(f"formulas file not found: {args.formulas}")

    with args.formulas.open(encoding="utf-8") as fh:
        formulas = [line.strip() for _, line in zip(range(args.limit), fh)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(run, formulas))

    total = len(outcomes)
    passed = [o for o in outcomes if o.ok]
    transpile_fail = [o for o in outcomes if o.stage == "transpile"]
    compile_fail = [o for o in outcomes if o.stage == "compile"]

    # Histogram: commands driving failures.
    unsupported = Counter()
    for o in transpile_fail:
        cmd = _unsupported_command(o.error)
        if cmd:
            unsupported[cmd] += 1
    compile_fail_cmds = Counter()
    for o in compile_fail:
        compile_fail_cmds.update(_commands_in(o.latex))

    pct = lambda n: f"{100 * n / total:.1f}%" if total else "n/a"
    print(f"\n=== Transpiler coverage over {total} formulas "
          f"({args.formulas.name}) ===")
    print(f"  compile-pass : {len(passed):>6}  ({pct(len(passed))})")
    print(f"  transpile-fail: {len(transpile_fail):>6}  ({pct(len(transpile_fail))})")
    print(f"  compile-fail : {len(compile_fail):>6}  ({pct(len(compile_fail))})")

    if unsupported:
        print("\n  Top unsupported commands (transpile failures):")
        for cmd, count in unsupported.most_common(20):
            print(f"    \\{cmd:<16} {count}")

    if compile_fail_cmds:
        print("\n  Commands frequent in compile failures (hints, not root cause):")
        for cmd, count in compile_fail_cmds.most_common(15):
            print(f"    \\{cmd:<16} {count}")

    for label, bucket in (("transpile", transpile_fail), ("compile", compile_fail)):
        if not bucket:
            continue
        print(f"\n  Sample {label} failures:")
        for o in bucket[: args.show]:
            print(f"    latex : {o.latex[:90]}")
            if o.typst is not None:
                print(f"    typst : {o.typst[:90]}")
            print(f"    error : {o.error[:120]}")
            print()


if __name__ == "__main__":
    main()
