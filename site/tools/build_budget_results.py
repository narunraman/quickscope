#!/usr/bin/env python3
"""Build site/assets/budget-results.js for the results section of the project page.

Reads the compare_at_budget.csv written by MA-eval's scripts/compare_methods_at_budget.py
(one row per benchmark x method x budget, scored against held-out ground truth) and keeps
the columns the page draws. That script writes one file per model, so pass one --csv per
model as MODEL=PATH; the page offers a switch between them.

    python site/tools/build_budget_results.py \
        --csv gpt-5.4-nano=out_nano/compare_at_budget.csv \
        --csv gpt-5-mini=out_mini/compare_at_budget.csv

With --runs MODEL=RESULTS_ROOT (the <root>/<benchmark>/<method>/seed0/search_history.jsonl
tree a campaign writes), it also lists the configurations each model's QuickScope run ranks
hardest, for the "What a run hands you" list. Ranking and certification use COUP's KL lower
bound (natural log, delta 0.01, pool = distinct configurations the run sampled).

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
# naive_true is the paper's "hard configurations found": sampled at least --naive-min-n times
# (20 in the paper) with empirical and held-out error rates both at least tau.
METRICS = ("naive_true", "high_explored", "topk_true", "topk_gt_mean", "explored", "certified", "cert_true")
# Configuration parameters per benchmark, as compare_methods_at_budget.py reads them.
PARAMS = {
    "dyval": ["dataset_type_id", "depth", "num_children_per_node", "extra_links_per_node", "add_rand_desc", "order_id"],
    "steer_me": ["element", "type", "domain"],
    "grid": ["grid_reasoning_task", "rows", "cols", "p_blocked", "num_islands", "island_size"],
}
# Which run to list configurations from: a certifying QuickScope run if there is one.
FOUND_FROM = ("coup_cert", "coup_adaptive", "coup_cert_betting", "coup_nocert")
TAUS = {"dyval": 0.9, "steer_me": 0.7, "grid": 0.9}


def kl(p: float, q: float) -> float:
    if p == 0:
        return math.log(1 / (1 - q))
    if p == 1:
        return math.log(1 / q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_lcb(mean: float, n: int, n_pool: int, delta: float = 0.01) -> float:
    """COUP's lower bound (ma_eval/optimization/bounds.py: coup_bounds, natural-log KL)."""
    if mean <= 1e-8:
        return 0.0
    a = math.log(26.71 * n_pool**2 * n**2 / delta) / n
    eps, steps = 1e-8, 1000
    for i in range(steps):
        q = eps + (mean - eps) * i / (steps - 1)
        if kl(mean, q) <= a:
            return q
    return mean


def found_configs(root: Path, k: int = 6) -> dict:
    """{benchmark: {method, tau, configs}} for the QuickScope run in each benchmark folder."""
    out = {}
    for bench, params in PARAMS.items():
        run = next((root / bench / m / "seed0" / "search_history.jsonl" for m in FOUND_FROM
                    if (root / bench / m / "seed0" / "search_history.jsonl").exists()), None)
        if run is None:
            continue
        stats: dict[tuple, list[float]] = {}
        with run.open() as f:
            for line in f:
                row = json.loads(line)
                if not row.get("observed") or row.get("utility") is None:
                    continue
                key = tuple(row.get(p) for p in params)
                s = stats.setdefault(key, [0.0, 0])
                s[0] += float(row["utility"])
                s[1] += 1
        n_pool = max(len(stats), 1)
        tau = TAUS[bench]
        # LCB never exceeds the mean, so scan by mean and stop once no mean can beat the k-th LCB.
        best: list[dict] = []
        for key, (total, n) in sorted(stats.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
            mean = total / n
            if len(best) >= k and mean < best[-1]["lcb"]:
                break
            lcb = kl_lcb(mean, n, n_pool)
            best.append({"params": dict(zip(params, key)), "mean": round(mean, 4), "lcb": round(lcb, 4),
                         "n": n, "certified": lcb >= tau})
            best.sort(key=lambda c: -c["lcb"])
            best = best[:k]
        out[bench] = {"method": run.parent.parent.name, "tau": tau, "configs": best}
    return out


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
    source.add_argument("--csv", action="append", metavar="MODEL=PATH",
                        help="a model's compare_at_budget.csv from compare_methods_at_budget.py; repeat per model")
    source.add_argument("--placeholder", action="store_true", help="write made-up curves for layout")
    parser.add_argument("--runs", action="append", default=[], metavar="MODEL=RESULTS_ROOT",
                        help="a model's campaign results folder, for the list of configurations it found")
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
    found = {}
    for spec in args.runs:
        model, sep, root = spec.partition("=")
        if not sep:
            parser.error(f"--runs takes MODEL=RESULTS_ROOT, got {spec!r}")
        found[model] = found_configs(Path(root))
        # Keep only the benchmarks this model is charted on, when it has chart rows at all.
        charted = {r["benchmark"] for r in rows if r["model"] == model}
        if charted:
            found[model] = {bench: v for bench, v in found[model].items() if bench in charted}
    data = {
        "placeholder": bool(args.placeholder),
        "found": found,
        "models": list(dict.fromkeys([*(r["model"] for r in rows), *found])),
        "benchmarks": {k: v for k, v in BENCHMARKS.items() if any(r["benchmark"] == k for r in rows)},
        "quickscope": list(QUICKSCOPE),
        "baselines": [m for m in BASELINES if any(r["method"] == m for r in rows)],
        "rows": rows,
    }
    args.out.write_text("window.QUICKSCOPE_BUDGET = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
