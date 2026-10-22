#!/usr/bin/env python3
"""Build site/assets/budget-results.js for the results section of the project page.

Reads the project-page bundle on the site-data branch of MA-eval (see its README.md):

    git -C MA-eval fetch origin site-data
    git -C MA-eval archive origin/site-data | tar -x -C site-data
    python site/tools/build_budget_results.py --site-data site-data \
        --gt gpt-oss-120b/grid=pooled --skip gpt-oss-120b/dyval

- comparisons/<model>/<benchmark>_<gt>.csv: one row per method x budget. Each model and
  benchmark is scored against held-out ground truth unless --gt picks pooled.
- found/<model>_<benchmark>.json: the configurations a QuickScope run ranks hardest (by KL
  lower bound), with certification by the betting test, for the "What a run hands you" list.

--skip leaves a model and benchmark off the page; --max-budget cuts one off at a budget.

Without the bundle, --placeholder writes made-up curves with the same shape so the
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
# naive_true is the paper's "hard configurations found": sampled at least --naive-min-n times
# (20 in the paper) with empirical and held-out error rates both at least tau.
METRICS = ("naive_true", "high_explored", "topk_true", "topk_gt_mean", "explored", "certified", "cert_true")
TAUS = {"dyval": 0.9, "steer_me": 0.7, "grid": 0.9}
# How many of a run's top configurations the page lists.
FOUND_SHOWN = 6


def param_value(value: str) -> int | float | str | None:
    """The bundle writes parameters as strings; "None" and "nan" mark ones a task does not use."""
    if value in ("None", "nan", ""):
        return None
    try:
        x = float(value)
    except ValueError:
        return value
    return int(x) if x.is_integer() else x


def found_configs(path: Path) -> dict:
    """{method, tau, certified, configs} from a found/<model>_<benchmark>.json."""
    run = json.loads(path.read_text())
    configs = [{
        "params": {k: v for k, v in ((k, param_value(v)) for k, v in c["params"].items()) if v is not None},
        "mean": c["error_rate"],
        "lcb": c["lower_bound"],
        "n": c["samples"],
        "certified": bool(c["betting_certified"]),
    } for c in run["top"][:FOUND_SHOWN]]
    return {"method": run["method"], "tau": run["tau"], "certified": run["betting_certified"], "configs": configs}


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
                "gt": row.get("gt"),
                **{m: number(row.get(m)) for m in METRICS},
            })
    return rows


def placeholder() -> list[dict]:
    """Made-up curves shaped like the paper's: the QuickScope variants climb faster."""
    rng = random.Random(4)
    # Open-weight models get every baseline; API models only uniform sampling, as in the real campaigns.
    models = {"gemma-3-27b-instruct": (0.85, None), "gpt-5.4-nano": (1.0, {"uniform"}), "gpt-5-mini": (0.7, {"uniform"})}
    budgets = [250, 500, 1000, 2000, 5000, 10000, 20000]
    speed = {"coup_nocert": 1.0, "coup_cert": 1.08, "coup_adaptive": 1.12, "uniform": 0.25,
             "successive_halving": 0.45, "hyperband": 0.42, "apt": 0.6, "smac_ac": 0.5}
    hard_configs = {"dyval": 180, "steer_me": 40, "grid": 260}
    taus = {"dyval": 0.9, "steer_me": 0.7, "grid": 0.9}
    rows = []
    for model, (hardness, baselines) in models.items():
        for bench in BENCHMARKS:
            for method, s in speed.items():
                if baselines is not None and method in BASELINES and method not in baselines:
                    continue
                for b in budgets:
                    progress = 1 - math.exp(-s * b / 6000)
                    noise = 1 + rng.uniform(-0.06, 0.06)
                    found = round(hard_configs[bench] * hardness * progress * noise)
                    top = min(10, round(10 * min(1, 1.6 * progress) * noise))
                    rows.append({
                        "model": model, "benchmark": bench, "method": method, "budget": b, "tau": taus[bench],
                        "naive_true": round(found * 0.6), "high_explored": found, "topk_true": top,
                        "topk_gt_mean": round(min(0.99, 0.45 + 0.5 * min(1, 1.5 * progress) * noise), 3),
                        "explored": None, "certified": None, "cert_true": None,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--site-data", type=Path, help="a checkout of MA-eval's site-data branch")
    source.add_argument("--placeholder", action="store_true", help="write made-up curves for layout")
    parser.add_argument("--gt", action="append", default=[], metavar="MODEL/BENCHMARK=pooled",
                        help="score this model and benchmark against pooled ground truth (default held-out)")
    parser.add_argument("--skip", action="append", default=[], metavar="MODEL/BENCHMARK",
                        help="leave this model and benchmark off the page")
    parser.add_argument("--max-budget", action="append", default=[], metavar="MODEL/BENCHMARK=CALLS",
                        help="drop budgets above CALLS for this model and benchmark")
    parser.add_argument("--reeval", type=Path,
                        help="re-tested top-K curves: {model: {benchmark: {method: {label, curve}}}}, "
                             "computed as in MA-eval's plot_reeval_comparison.py")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    gt = {}
    for spec in args.gt:
        key, sep, value = spec.partition("=")
        if not sep or value not in ("heldout", "pooled"):
            parser.error(f"--gt takes MODEL/BENCHMARK=heldout|pooled, got {spec!r}")
        gt[key] = value
    skip = set(args.skip)
    max_budget = {}
    for spec in args.max_budget:
        key, sep, value = spec.partition("=")
        if not sep or not value.isdigit():
            parser.error(f"--max-budget takes MODEL/BENCHMARK=CALLS, got {spec!r}")
        max_budget[key] = int(value)

    rows, found = [], {}
    if args.placeholder:
        rows = placeholder()
    else:
        for model_dir in sorted((args.site_data / "comparisons").iterdir()):
            model = model_dir.name
            for bench in BENCHMARKS:
                key = f"{model}/{bench}"
                path = model_dir / f"{bench}_{gt.get(key, 'heldout')}.csv"
                if key in skip or not path.exists():
                    continue
                cap = max_budget.get(key)
                rows += [r for r in from_csv(model, path) if cap is None or r["budget"] <= cap]
                found_path = args.site_data / "found" / f"{model}_{bench}.json"
                if found_path.exists():
                    found.setdefault(model, {})[bench] = found_configs(found_path)
    reeval = json.loads(args.reeval.read_text()) if args.reeval else {}
    reeval_benches = {bench for benches in reeval.values() for bench in benches}
    data = {
        "placeholder": bool(args.placeholder),
        "found": found,
        "reeval": reeval,
        "models": list(dict.fromkeys([*(r["model"] for r in rows), *found, *reeval])),
        "benchmarks": {k: v for k, v in BENCHMARKS.items()
                       if any(r["benchmark"] == k for r in rows) or k in reeval_benches},
        "quickscope": list(QUICKSCOPE),
        "baselines": [m for m in BASELINES if any(r["method"] == m for r in rows)],
        "rows": rows,
    }
    args.out.write_text("window.QUICKSCOPE_BUDGET = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
