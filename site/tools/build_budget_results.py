#!/usr/bin/env python3
"""Build site/assets/budget-results.js for the results section of the project page.

Reads the compare_at_budget.csv written by MA-eval's scripts/compare_methods_at_budget.py
(one row per benchmark x method x budget, scored against held-out ground truth) and keeps
the columns the page draws. That script writes one file per model, so pass one --csv per
model as MODEL=PATH; the page offers a switch between them.

    python site/tools/build_budget_results.py \
        --csv gpt-5.4-nano=out_nano/compare_at_budget.csv \
        --csv gpt-5-mini=out_mini/compare_at_budget.csv

Without a results file, --placeholder writes made-up curves with the same shape so the
page can be laid out; the page labels them as placeholders.

    python site/tools/build_budget_results.py --placeholder
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "budget-results.js"

BENCHMARKS = {"dyval": "DyVal", "steer_me": "STEER-ME", "grid": "Grid Reasoning"}
# The QuickScope variants are drawn as one band; the rest are baselines drawn as lines.
QUICKSCOPE = ("coup_nocert", "coup_cert", "coup_adaptive")
BASELINES = ("uniform", "successive_halving", "hyperband", "apt", "smac_ac")
METRICS = ("high_explored", "topk_true", "topk_gt_mean", "explored", "certified", "cert_true")


def number(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def from_csv(model: str, path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["benchmark"] not in BENCHMARKS or row["method"] not in (*QUICKSCOPE, *BASELINES):
                continue
            rows.append({
                "model": model,
                "benchmark": row["benchmark"],
                "method": row["method"],
                "budget": int(float(row["budget"])),
                "tau": number(row["tau"]),
                **{m: number(row.get(m)) for m in METRICS},
            })
    return rows


def placeholder() -> list[dict]:
    """Made-up curves shaped like the paper's: the QuickScope variants climb faster."""
    rng = random.Random(4)
    models = {"gpt-5.4-nano": 1.0, "gpt-5-mini": 0.7}
    budgets = [250, 500, 1000, 2000, 5000, 10000, 20000]
    speed = {"coup_nocert": 1.0, "coup_cert": 1.08, "coup_adaptive": 1.12, "uniform": 0.25,
             "successive_halving": 0.45, "hyperband": 0.42, "apt": 0.6, "smac_ac": 0.5}
    hard_configs = {"dyval": 180, "steer_me": 40, "grid": 260}
    taus = {"dyval": 0.9, "steer_me": 0.7, "grid": 0.9}
    rows = []
    for model, hardness in models.items():
        for bench in BENCHMARKS:
            for method, s in speed.items():
                for b in budgets:
                    progress = 1 - math.exp(-s * b / 6000)
                    noise = 1 + rng.uniform(-0.06, 0.06)
                    found = round(hard_configs[bench] * hardness * progress * noise)
                    top = min(10, round(10 * min(1, 1.6 * progress) * noise))
                    rows.append({
                        "model": model, "benchmark": bench, "method": method, "budget": b, "tau": taus[bench],
                        "high_explored": found, "topk_true": top,
                        "topk_gt_mean": round(min(0.99, 0.45 + 0.5 * min(1, 1.5 * progress) * noise), 3),
                        "explored": None, "certified": None, "cert_true": None,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", action="append", metavar="MODEL=PATH",
                        help="a model's compare_at_budget.csv from compare_methods_at_budget.py; repeat per model")
    source.add_argument("--placeholder", action="store_true", help="write made-up curves for layout")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.placeholder:
        rows = placeholder()
    else:
        rows = []
        for spec in args.csv:
            model, sep, path = spec.partition("=")
            if not sep:
                parser.error(f"--csv takes MODEL=PATH, got {spec!r}")
            rows += from_csv(model, Path(path))
    data = {
        "placeholder": bool(args.placeholder),
        "models": list(dict.fromkeys(r["model"] for r in rows)),
        "benchmarks": {k: v for k, v in BENCHMARKS.items() if any(r["benchmark"] == k for r in rows)},
        "quickscope": list(QUICKSCOPE),
        "baselines": [m for m in BASELINES if any(r["method"] == m for r in rows)],
        "rows": rows,
    }
    args.out.write_text("window.QUICKSCOPE_BUDGET = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
