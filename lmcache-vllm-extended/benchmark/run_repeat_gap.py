#!/usr/bin/env python3
"""
Q2 variant: 14 papers, 1 question each — run all first pass, then all exact repeat (28 total).

  python run_repeat_gap.py --cache-gb 0.2
  python run_repeat_gap.py --cache-gb 0.2 --plot-only

Optional interleaved mode (84 reqs):  --mode gap --repeat-gap 4

Output:
  results/repeat_gap_cache0.2.json
  results/repeat_gap_cache0.2.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BENCHMARK_DIR.parent
_RESULTS_DIR = _BENCHMARK_DIR / "results"
_RUN_NAME = "repeat_gap"

if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from request_generator import (  # noqa: E402
    annotate_prompt_tokens,
    build_repeat_gap_schedule,
    build_repeat_two_phase_schedule,
    load_contexts,
)

import run_task1 as bench  # noqa: E402

TITLE = "Two-phase repeat: first round then repeat round (full response time)"
PLOT_METRIC = "response_time"


def _build_schedule(args: argparse.Namespace):
    contexts = load_contexts(
        args.data_dir, exclude=args.exclude, max_contexts=args.num_contexts
    )
    if args.mode == "gap":
        sched = build_repeat_gap_schedule(
            contexts,
            requests_per_context=args.questions_per_paper,
            repeat_gap=args.repeat_gap,
            seed=args.seed,
            data_dir=str(args.data_dir),
        )
    else:
        sched = build_repeat_two_phase_schedule(
            contexts,
            requests_per_context=args.questions_per_paper,
            seed=args.seed,
            data_dir=str(args.data_dir),
        )
    if not args.no_token_count:
        sched = annotate_prompt_tokens(sched)
    return sched


def _avg_visit(rows: list[dict], visit_type: str) -> float | None:
    xs = [r for r in rows if r.get("visit_type") == visit_type]
    if not xs:
        return None
    try:
        return sum(bench._metric_value(r, PLOT_METRIC) for r in xs) / len(xs)
    except KeyError:
        return None


def _paths(cache_gb: float) -> tuple[Path, Path]:
    stem = f"{_RUN_NAME}_cache{cache_gb:g}"
    return _RESULTS_DIR / f"{stem}.json", _RESULTS_DIR / f"{stem}.png"


def _plot(payload: dict, out_png: Path) -> None:
    rows = [r for r in payload["results"] if r["success"]]
    if not bench._has_response_time(rows):
        print(
            "  [plot] FAILED: JSON has no response_time_sec (old run used TTFT only). "
            f"Re-run: python run_repeat_gap.py --cache-gb {payload['summary'].get('local_cache_gb', 0.2)}"
        )
        return

    first = {r["request_id"]: r for r in rows if r["visit_type"] == "first"}
    pairs: list[tuple[dict, dict]] = []
    for r in rows:
        if r["visit_type"] == "exact_repeat" and r.get("source_request_id"):
            src = first.get(r["source_request_id"])
            if src:
                pairs.append((src, r))
    if not pairs:
        print("  [plot] FAILED: no first/exact_repeat pairs in JSON")
        return

    try:
        show = pairs[:12]
        first_vals = [bench._metric_value(p[0], PLOT_METRIC) for p in show]
        repeat_vals = [bench._metric_value(p[1], PLOT_METRIC) for p in show]
    except KeyError:
        print("  [plot] FAILED: missing response_time_sec in some rows")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [p[0]["context_id"][:10] for p in show]
    x = range(len(show))
    w = 0.35
    axes[0].bar(
        [i - w / 2 for i in x],
        first_vals,
        w,
        label="first",
        color="tab:blue",
    )
    axes[0].bar(
        [i + w / 2 for i in x],
        repeat_vals,
        w,
        label="repeat",
        color="tab:orange",
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    axes[0].set_ylabel(bench._ylabel_for_metric(PLOT_METRIC))
    axes[0].set_title("Per paper: first vs repeat")
    axes[0].legend()

    s = payload["summary"]
    avg_first, avg_rep = bench._summary_avg_first_repeat(s, PLOT_METRIC)
    if avg_first is None or avg_rep is None:
        avg_first = _avg_visit(rows, "first")
        avg_rep = _avg_visit(rows, "exact_repeat")
    if avg_first is None or avg_rep is None:
        print("  [plot] FAILED: missing full-response averages")
        plt.close(fig)
        return
    axes[1].bar(
        ["first pass", "exact repeat"],
        [avg_first, avg_rep],
        color=["tab:blue", "tab:orange"],
    )
    axes[1].set_ylabel(bench._ylabel_for_metric(PLOT_METRIC, average=True))
    axes[1].set_title(f"Average (n={len(pairs)} pairs, cache={s.get('local_cache_gb')} GB)")

    fig.suptitle(TITLE, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def _plot_cache_sweep() -> None:
    files = sorted(_RESULTS_DIR.glob(f"{_RUN_NAME}_cache*.json"))
    points: list[tuple[float, dict]] = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        gb = d["summary"].get("local_cache_gb")
        if gb is not None:
            points.append((float(gb), d["summary"]))
    if len(points) < 2:
        return
    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    out = _RESULTS_DIR / f"{_RUN_NAME}_cache_sweep.png"

    first_y, rep_y = [], []
    for _, s in points:
        af, ar = bench._summary_avg_first_repeat(s, PLOT_METRIC)
        first_y.append(af)
        rep_y.append(ar)
    if not all(x is not None for x in first_y + rep_y):
        print("  [plot] cache sweep skipped: missing response_time averages (re-run)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, [p[1]["throughput_req_per_sec"] for p in points], "o-", color="steelblue")
    axes[0].set(xlabel="max_local_cache_size (GB)", ylabel="Throughput (req/s)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, first_y, "o-", label="first", color="tab:blue")
    axes[1].plot(xs, rep_y, "s--", label="repeat", color="tab:orange")
    axes[1].set(
        xlabel="max_local_cache_size (GB)",
        ylabel=bench._ylabel_for_metric(PLOT_METRIC, average=True),
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{_RUN_NAME}: local cache size")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out} ({len(points)} settings)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-gb", type=float, default=0.2)
    p.add_argument(
        "--mode",
        choices=["two_phase", "gap"],
        default="two_phase",
        help="two_phase: 14 first + 14 repeat = 28 (default); gap: interleaved with fillers",
    )
    p.add_argument("--repeat-gap", type=int, default=4, help="Only for --mode gap")
    p.add_argument("--questions-per-paper", type=int, default=1)
    p.add_argument("--data-dir", type=Path, default=_PROJECT_DIR / "frontend" / "data")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v2")
    p.add_argument("--num-contexts", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plot", action="store_true", help="Save JSON only (batch sweeps)")
    p.add_argument("--no-token-count", action="store_true")
    p.add_argument("--exclude", nargs="*", default=["sample"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path, png_path = _paths(args.cache_gb)

    if args.plot_only:
        if not json_path.is_file():
            raise SystemExit(f"Missing {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        _plot(payload, png_path)
        _plot_cache_sweep()
        return

    schedule = _build_schedule(args)
    n_first = sum(1 for r in schedule.requests if r.visit_type == "first")
    print(f"repeat_gap ({args.mode}): {len(schedule.requests)} requests, {n_first} papers")
    print(f"  {schedule.notes}")

    if args.dry_run:
        return

    payload = bench.run_schedule(
        schedule,
        api_base=args.api_base,
        max_tokens=64,
        temperature=0.0,
        warmup=0,
        question=_RUN_NAME,
        cache_gb=args.cache_gb,
        plot_metric=PLOT_METRIC,
    )
    payload["summary"]["mode"] = args.mode
    payload["summary"]["experiment"] = _RUN_NAME

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")
    if not args.no_plot:
        try:
            _plot(payload, png_path)
            _plot_cache_sweep()
        except Exception as exc:  # noqa: BLE001
            print(f"  [plot] ERROR: {exc}")
            print(f"  Re-run: python .../run_repeat_gap.py --cache-gb {args.cache_gb} --plot-only")
            raise

    s = payload["summary"]
    rf, rr = bench._summary_avg_first_repeat(s, PLOT_METRIC)
    print(f"\n--- summary ---")
    print(f"  throughput : {s['throughput_req_per_sec']:.3f} req/s")
    print(f"  RT first   : {rf:.3f} s" if rf is not None else "  RT first   : n/a")
    print(f"  RT repeat  : {rr:.3f} s" if rr is not None else "  RT repeat  : n/a")
    print(f"  TTFT (ref) : {s.get('avg_ttft_sec', 0):.3f} s")


if __name__ == "__main__":
    main()
