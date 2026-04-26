#!/usr/bin/env python3
"""
Plot summaries from `accuracy_speed_report` (summary.tsv).
Requires: matplotlib

  python3 scripts/plot_accuracy_speed.py --tsv target/accuracy_sweep/summary.tsv --out target/accuracy_sweep/figs
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in r]


def group_by_scenario(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    d: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        d[row["scenario"]].append(row)
    return dict(d)


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_mae_rmse(
    by_scen: dict[str, list[dict[str, str]]], out: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted(by_scen.keys())
    paths = ["homo", "hetero"]
    x0 = list(range(len(scenarios)))
    w = 0.35
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    for i, p in enumerate(paths):
        off = (i - 0.5) * w
        mae = [_f(next(r for r in by_scen[s] if r["path"] == p), "mae_prevalence") for s in scenarios]
        rmse = [
            _f(next(r for r in by_scen[s] if r["path"] == p), "rmse_prevalence")
            for s in scenarios
        ]
        xp = [xi + off for xi in x0]
        ax0.bar(xp, mae, w, label=p)
        ax1.bar(xp, rmse, w, label=p)
    for ax, ylab in ((ax0, "MAE (prevalence)"), (ax1, "RMSE (prevalence)")):
        ax.set_xticks(x0, scenarios, rotation=15, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
    ax0.legend()
    ax1.legend()
    fig.suptitle(title)
    fig.tight_layout()
    dest = out / "accuracy_mae_rmse.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Wrote {dest}")


def plot_timings(
    by_scen: dict[str, list[dict[str, str]]], out: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted(by_scen.keys())
    paths = ["homo", "hetero"]
    # One grouped cluster per (scenario, path) with stacked build / pred parts
    labels: list[str] = []
    fit_ms: list[float] = []
    build_ms: list[float] = []
    ptest_ms: list[float] = []
    pgrid_ms: list[float] = []
    for s in scenarios:
        for p in paths:
            row = next(r for r in by_scen[s] if r["path"] == p)
            labels.append(f"{s}\n{p}")
            fit_ms.append(_f(row, "fit_variogram_ms"))
            build_ms.append(_f(row, "build_model_ms"))
            ptest_ms.append(_f(row, "pred_test_ms"))
            pgrid_ms.append(_f(row, "pred_grid_ms"))

    x = list(range(len(labels)))
    w = 0.55
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 4.5))
    ax.bar(x, fit_ms, w, label="fit variogram")
    ax.bar(x, build_ms, w, bottom=fit_ms, label="build model")
    ax.bar(
        x,
        ptest_ms,
        w,
        bottom=[a + b for a, b in zip(fit_ms, build_ms)],
        label="pred test set",
    )
    bot = [a + b + c for a, b, c in zip(fit_ms, build_ms, ptest_ms)]
    ax.bar(x, pgrid_ms, w, bottom=bot, label="pred grid")
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    dest = out / "accuracy_timings_stacked.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Wrote {dest}")


def plot_scatter_accuracy_vs_speed(
    rows: list[dict[str, str]], out: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for row in rows:
        t = _f(row, "build_model_ms") + _f(row, "pred_grid_ms")
        mae = _f(row, "mae_prevalence")
        ax.scatter(t, mae, s=60, label=f"{row['scenario']}/{row['path']}", alpha=0.85)
    ax.set_xlabel("build + pred_grid (ms)")
    ax.set_ylabel("MAE (prevalence)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    dest = out / "accuracy_vs_speed_mae.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True, help="summary.tsv from accuracy_speed_report")
    ap.add_argument(
        "--out", type=Path, required=True, help="output directory for PNGs"
    )
    ap.add_argument(
        "--title", default="Binomial kriging (homo vs hetero)", help="plot title"
    )
    args = ap.parse_args()
    rows = load_rows(args.tsv)
    if not rows:
        raise SystemExit("empty TSV")
    args.out.mkdir(parents=True, exist_ok=True)
    by = group_by_scenario(rows)
    plot_mae_rmse(by, args.out, args.title)
    plot_timings(by, args.out, args.title)
    plot_scatter_accuracy_vs_speed(rows, args.out, args.title)


if __name__ == "__main__":
    main()
