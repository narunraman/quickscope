"""Public plotting helpers for Quickscope results_*.json artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from quickscope.results import ResultRun


PlotSort = Literal["auto", "lcb", "utility"]


@dataclass
class PlotPoint:
    """One ranked configuration in a plot."""

    rank: int
    mean: float
    lcb: float | None
    ucb: float | None
    n_evals: int | None


@dataclass
class PlotSeries:
    """One run's plotted configuration series."""

    run: str
    scenario: str
    optimizer: str
    utility: str
    sort_by: str
    configs: int
    points: list[PlotPoint]


def plot_intervals(
    runs: Iterable[ResultRun],
    output_path: str | Path,
    *,
    top_k: int = 100,
    sort_by: PlotSort = "auto",
    log_x: bool = False,
) -> list[PlotSeries]:
    """Plot LCB-UCB intervals and mean-utility dots for top configs."""
    series = build_plot_series(
        runs,
        top_k=top_k,
        sort_by=sort_by,
        require_bounds=True,
    )
    if not series:
        return []

    plt = _load_pyplot()
    axes = _make_axes(plt, len(series), height=4.8)
    for ax, item in zip(axes, series):
        xs = np.arange(1, len(item.points) + 1)
        means = np.array([point.mean for point in item.points], dtype=float)
        lcbs = np.array([point.lcb for point in item.points], dtype=float)
        ucbs = np.array([point.ucb for point in item.points], dtype=float)

        ax.fill_between(xs, lcbs, ucbs, alpha=0.16)
        ax.plot(xs, lcbs, alpha=0.45, linewidth=0.8, label="LCB")
        ax.plot(xs, ucbs, alpha=0.45, linewidth=0.8, label="UCB")
        ax.scatter(
            xs,
            means,
            s=18,
            color="black",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label="mean",
        )
        _style_axis(
            ax,
            _series_title(item),
            f"Config rank by {item.sort_by}",
            "Utility",
            log_x=log_x,
        )

    _finish_figure(plt, output_path, "Config Intervals")
    return series


def plot_cumulative_average(
    runs: Iterable[ResultRun],
    output_path: str | Path,
    *,
    top_k: int = 100,
    sort_by: PlotSort = "auto",
    log_x: bool = False,
) -> list[PlotSeries]:
    """Plot cumulative top-K mean utility with a lower running envelope."""
    series = build_plot_series(
        runs,
        top_k=top_k,
        sort_by=sort_by,
        require_bounds=False,
    )
    if not series:
        return []

    plt = _load_pyplot()
    axes = _make_axes(plt, len(series), height=4.8)
    for ax, item in zip(axes, series):
        means = np.array([point.mean for point in item.points])
        xs = np.arange(1, len(means) + 1)
        cumavg = np.cumsum(means) / xs
        running_min = np.minimum.accumulate(means)

        line = ax.plot(
            xs,
            cumavg,
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            label="cumulative mean",
        )
        color = line[0].get_color()
        ax.fill_between(xs, running_min, cumavg, color=color, alpha=0.14)
        ax.plot(
            xs,
            running_min,
            color=color,
            alpha=0.4,
            linewidth=0.7,
            label="running minimum",
        )
        _style_axis(
            ax,
            _series_title(item),
            "Cutoff K",
            "Utility",
            log_x=log_x,
        )

    _finish_figure(plt, output_path, "Cumulative Top-K Utility")
    return series


def plot_utility_rankings(
    runs: Iterable[ResultRun],
    output_path: str | Path,
    *,
    top_k: int = 25,
    log_x: bool = False,
) -> list[PlotSeries]:
    """Plot top configurations by observed mean utility."""
    series = build_plot_series(
        runs,
        top_k=top_k,
        sort_by="utility",
        require_bounds=False,
    )
    if not series:
        return []

    plt = _load_pyplot()
    axes = _make_axes(plt, len(series), height=4.8)
    for ax, item in zip(axes, series):
        xs = np.arange(1, len(item.points) + 1)
        means = np.array([point.mean for point in item.points], dtype=float)
        colors = ["#2f6f73" if index % 2 else "#d65f3f" for index in range(len(xs))]

        ax.bar(xs, means, color=colors, alpha=0.88, width=0.72)
        _style_axis(
            ax,
            _series_title(item),
            "Config rank by mean utility",
            "Mean utility",
            log_x=log_x,
            show_legend=False,
        )
        ax.set_ylim(bottom=0, top=max(1.0, float(means.max()) * 1.08))

    _finish_figure(plt, output_path, "Top Configurations by Utility")
    return series


def build_plot_series(
    runs: Iterable[ResultRun],
    *,
    top_k: int,
    sort_by: PlotSort,
    require_bounds: bool,
) -> list[PlotSeries]:
    """Build ranked config series for result plots."""
    output: list[PlotSeries] = []
    for run in runs:
        rows = []
        for config in _config_rows(run):
            mean = _numeric(config.get("mean_utility"))
            if mean is None:
                continue
            lcb = _numeric(config.get("lcb"))
            ucb = _numeric(config.get("ucb"))
            if require_bounds and (lcb is None or ucb is None):
                continue
            rows.append(
                {
                    "mean": mean,
                    "lcb": lcb,
                    "ucb": ucb,
                    "n_evals": _int_or_none(config.get("n_evals")),
                }
            )

        chosen_sort = _resolve_sort(rows, sort_by)
        rows.sort(key=lambda row: _sort_value(row.get(chosen_sort)), reverse=True)
        points = [
            PlotPoint(
                rank=index,
                mean=row["mean"],
                lcb=row["lcb"],
                ucb=row["ucb"],
                n_evals=row["n_evals"],
            )
            for index, row in enumerate(rows[:top_k], start=1)
        ]
        if not points:
            continue
        output.append(
            PlotSeries(
                run=run.label,
                scenario=run.scenario,
                optimizer=run.optimizer,
                utility=run.utility_name,
                sort_by=chosen_sort,
                configs=len(_config_rows(run)),
                points=points,
            )
        )
    return output


def render_terminal_intervals(series: Iterable[PlotSeries], *, max_rows: int = 25) -> str:
    """Render a compact ASCII interval plot for terminal output."""
    chunks = []
    for item in series:
        points = item.points[:max_rows]
        values = [
            value
            for point in points
            for value in (point.lcb, point.ucb, point.mean)
            if value is not None
        ]
        if not values:
            continue
        lo, hi = _axis_limits(values)
        width = 36
        chunks.append(f"{_series_title(item)} | bounds and mean")
        chunks.append(f"scale {lo:.3f} .. {hi:.3f}")
        for point in points:
            line = [" "] * width
            left = _to_column(point.lcb, lo, hi, width)
            right = _to_column(point.ucb, lo, hi, width)
            dot = _to_column(point.mean, lo, hi, width)
            if left is not None and right is not None:
                for index in range(min(left, right), max(left, right) + 1):
                    line[index] = "-"
                line[left] = "["
                line[right] = "]"
            if dot is not None:
                line[dot] = "*"
            chunks.append(
                f"{point.rank:>3} |{''.join(line)}| "
                f"mean={point.mean:.3f} ci={_fmt(point.lcb)}-{_fmt(point.ucb)}"
            )
        chunks.append("")
    return "\n".join(chunks).rstrip()


def render_terminal_utility(series: Iterable[PlotSeries], *, max_rows: int = 25) -> str:
    """Render a compact ASCII ranking by mean utility."""
    chunks = []
    width = 32
    for item in series:
        points = item.points[:max_rows]
        if not points:
            continue
        hi = max(point.mean for point in points) or 1.0
        chunks.append(f"{_series_title(item)} | top mean utility")
        for point in points:
            filled = int(round((point.mean / hi) * width))
            bar = "#" * filled + " " * (width - filled)
            evals = "" if point.n_evals is None else f" evals={point.n_evals}"
            chunks.append(f"{point.rank:>3} |{bar}| {point.mean:.3f}{evals}")
        chunks.append("")
    return "\n".join(chunks).rstrip()


def render_terminal_cumulative(series: Iterable[PlotSeries], *, height: int = 12) -> str:
    """Render a compact ASCII cumulative-average plot for terminal output."""
    chunks = []
    for item in series:
        if not item.points:
            continue
        means = np.array([point.mean for point in item.points])
        xs = np.arange(1, len(means) + 1)
        cumavg = np.cumsum(means) / xs
        running_min = np.minimum.accumulate(means)
        values = cumavg.tolist() + running_min.tolist()
        lo, hi = _axis_limits(values)
        width = min(72, max(24, len(means)))
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        _draw_curve(canvas, cumavg, lo, hi, char="*")
        _draw_curve(canvas, running_min, lo, hi, char=".")
        chunks.append(f"{_series_title(item)} | cumulative mean")
        chunks.append(f"scale {lo:.3f} .. {hi:.3f} | * average, . running min")
        for row_index, row in enumerate(canvas):
            y_value = hi - (hi - lo) * row_index / max(1, height - 1)
            chunks.append(f"{y_value:>6.3f} |{''.join(row)}|")
        chunks.append(f"       +{'-' * width}+")
        chunks.append(f"        1{' ' * max(1, width - 8)}K={len(means)}")
        chunks.append("")
    return "\n".join(chunks).rstrip()


def default_plot_path(results_path: str | Path, filename: str) -> Path:
    """Return the default plot path for a result file or directory."""
    root = Path(results_path).expanduser()
    base_dir = root.parent if root.is_file() else root
    return base_dir / "plots" / filename


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    return plt


def _make_axes(plt, n_axes: int, *, height: float):
    fig_width = min(18.0, max(6.0, 5.5 * n_axes))
    _, axes = plt.subplots(1, n_axes, figsize=(fig_width, height), squeeze=False)
    return axes.ravel().tolist()


def _finish_figure(plt, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _style_axis(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    log_x: bool,
    show_legend: bool = True,
) -> None:
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if show_legend:
        ax.legend(fontsize=7, frameon=False)


def _series_title(series: PlotSeries) -> str:
    utility = f" / {series.utility}" if series.utility else ""
    return f"{series.scenario} / {series.run}{utility}"


def _config_rows(run: ResultRun) -> list[dict[str, Any]]:
    return [row for row in run.configs if isinstance(row, dict)]


def _resolve_sort(rows: list[dict[str, Any]], sort_by: PlotSort) -> str:
    if sort_by == "auto":
        if any(row.get("lcb") is not None for row in rows):
            return "lcb"
        return "mean"
    if sort_by == "utility":
        return "mean"
    return "lcb"


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _int_or_none(value: Any) -> int | None:
    numeric = _numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def _sort_value(value: Any) -> float:
    numeric = _numeric(value)
    if numeric is None:
        return float("-inf")
    return numeric


def _axis_limits(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = 0.05 if lo == 0 else abs(lo) * 0.05
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _to_column(value: float | None, lo: float, hi: float, width: int) -> int | None:
    if value is None or hi <= lo:
        return None
    scaled = (value - lo) / (hi - lo)
    return max(0, min(width - 1, int(round(scaled * (width - 1)))))


def _draw_curve(
    canvas: list[list[str]],
    values: np.ndarray,
    lo: float,
    hi: float,
    *,
    char: str,
) -> None:
    height = len(canvas)
    width = len(canvas[0]) if canvas else 0
    if width == 0 or hi <= lo:
        return
    for index, value in enumerate(values):
        x = int(round(index * (width - 1) / max(1, len(values) - 1)))
        y_scaled = (float(value) - lo) / (hi - lo)
        y = height - 1 - int(round(y_scaled * (height - 1)))
        y = max(0, min(height - 1, y))
        canvas[y][x] = char


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"
