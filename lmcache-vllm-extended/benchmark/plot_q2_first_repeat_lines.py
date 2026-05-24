#!/usr/bin/env python3
"""Q2 plots: cache sweep (avg first/repeat) or per-request first vs repeat lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BENCH_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCH_DIR / "results"
_DEFAULT_CACHE_GB = (0.05, 0.1, 0.2, 0.4)


def _metric_value(row: dict, metric: str = "response_time") -> float:
    if metric == "response_time":
        val = row.get("response_time_sec")
        if val is None:
            raise KeyError("response_time_sec")
        return float(val)
    return float(row.get("ttft_sec") or row.get("latency_sec") or 0.0)


def _summary_avgs(summary: dict, metric: str) -> tuple[float, float]:
    if metric == "response_time":
        first = summary.get("avg_response_time_first_sec")
        repeat = summary.get("avg_response_time_repeat_sec")
    else:
        first = summary.get("avg_ttft_first_sec", summary.get("avg_latency_first_sec"))
        repeat = summary.get("avg_ttft_repeat_sec", summary.get("avg_latency_repeat_sec"))
    if first is None or repeat is None:
        raise ValueError(f"summary missing avg for metric={metric}")
    return float(first), float(repeat)


def _load_cache_sweep(
    stem: str,
    cache_sizes: tuple[float, ...],
    metric: str,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for gb in cache_sizes:
        path = _RESULTS_DIR / f"{stem}_cache{gb:g}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        first, repeat = _summary_avgs(payload["summary"], metric)
        points.append((gb, first, repeat))
    return points


def _plot_cache_sweep(
    points: list[tuple[float, float, float]],
    *,
    title: str,
    out_png: Path,
    metric: str = "response_time",
) -> None:
    xs = [p[0] for p in points]
    first_y = [p[1] for p in points]
    repeat_y = [p[2] for p in points]
    ylab = (
        "Avg full response time (s)"
        if metric == "response_time"
        else "Avg TTFT (s)"
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, first_y, "o-", label="first (avg)", color="tab:blue", linewidth=2, markersize=7)
    ax.plot(xs, repeat_y, "s-", label="repeat (avg)", color="tab:orange", linewidth=2, markersize=7)
    ax.set_xlabel("Local cache size (GB)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def plot_cache_sweeps(
    cache_sizes: tuple[float, ...] = _DEFAULT_CACHE_GB,
    metric: str = "response_time",
) -> None:
    try:
        q2 = _load_cache_sweep("q2", cache_sizes, metric)
        gap = _load_cache_sweep("repeat_gap", cache_sizes, metric)
    except ValueError as exc:
        print(f"  [plot] FAILED: {exc}")
        print("  Re-run Q2 / repeat_gap experiments to record response_time_sec.")
        return

    _plot_cache_sweep(
        q2,
        title="Q2 back-to-back: avg full response time vs cache",
        out_png=_RESULTS_DIR / "q2_avg_response_back_to_back.png",
        metric=metric,
    )
    _plot_cache_sweep(
        gap,
        title="Q2 two-phase: avg full response time vs cache",
        out_png=_RESULTS_DIR / "q2_avg_response_two_phase.png",
        metric=metric,
    )


def _extract_pairs(payload: dict) -> list[tuple[dict, dict]]:
    rows = [r for r in payload["results"] if r["success"]]
    first = {r["request_id"]: r for r in rows if r["visit_type"] == "first"}
    pairs: list[tuple[dict, dict]] = []
    for r in rows:
        if r["visit_type"] == "exact_repeat" and r.get("source_request_id"):
            src = first.get(r["source_request_id"])
            if src:
                pairs.append((src, r))
    pairs.sort(key=lambda p: p[0]["sequence_index"])
    return pairs


def _one_pair_per_context(pairs: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    seen: set[str] = set()
    out: list[tuple[dict, dict]] = []
    for src, rep in pairs:
        if src["context_id"] in seen:
            continue
        seen.add(src["context_id"])
        out.append((src, rep))
    return out


def _plot_per_request(
    pairs: list[tuple[dict, dict]],
    *,
    title: str,
    subtitle: str,
    out_png: Path,
    metric: str = "response_time",
) -> None:
    x = list(range(1, len(pairs) + 1))
    ylab = (
        "Full response time (s)"
        if metric == "response_time"
        else "TTFT (s)"
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, [_metric_value(p[0], metric) for p in pairs], "o-", label="first", color="tab:blue")
    ax.plot(x, [_metric_value(p[1], metric) for p in pairs], "s-", label="repeat", color="tab:orange")
    ax.set_xlabel("Request pair index (execution order)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def plot_per_request(
    cache_gb: float, *, one_question: bool = False, metric: str = "response_time"
) -> None:
    q2_path = _RESULTS_DIR / f"q2_cache{cache_gb:g}.json"
    gap_path = _RESULTS_DIR / f"repeat_gap_cache{cache_gb:g}.json"
    q2_pairs = _extract_pairs(json.loads(q2_path.read_text(encoding="utf-8")))
    gap_pairs = _extract_pairs(json.loads(gap_path.read_text(encoding="utf-8")))
    if one_question:
        q2_pairs = _one_pair_per_context(q2_pairs)
    label = f"cache={cache_gb:g} GB"
    _plot_per_request(
        q2_pairs,
        title="Q2: back-to-back repeat",
        subtitle=f"{len(q2_pairs)} pairs — {label}",
        out_png=_RESULTS_DIR / f"q2_lines_back_to_back_cache{cache_gb:g}.png",
        metric=metric,
    )
    _plot_per_request(
        gap_pairs,
        title="Q2: two-phase repeat",
        subtitle=f"{len(gap_pairs)} pairs — {label}",
        out_png=_RESULTS_DIR / f"q2_lines_two_phase_cache{cache_gb:g}.png",
        metric=metric,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["cache-sweep", "per-request"],
        default="cache-sweep",
        help="cache-sweep: x=cache, y=avg first/repeat (default); per-request: x=pair index",
    )
    p.add_argument("--cache-gb", type=float, default=0.2, help="Only for --mode per-request")
    p.add_argument("--one-question", action="store_true")
    p.add_argument("--metric", choices=["ttft", "response_time"], default="response_time")
    p.add_argument(
        "--cache-sizes",
        type=float,
        nargs="+",
        default=list(_DEFAULT_CACHE_GB),
        help="GB values for cache-sweep (default: 0.05 0.1 0.2 0.4)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cache-sweep":
        plot_cache_sweeps(tuple(args.cache_sizes), metric=args.metric)
    else:
        plot_per_request(args.cache_gb, one_question=args.one_question, metric=args.metric)


if __name__ == "__main__":
    main()
