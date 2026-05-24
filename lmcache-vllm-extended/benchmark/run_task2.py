#!/usr/bin/env python3
"""
IK2221 Task 2 — batch requests + context-grouped scheduler on /v2.

  python run_task2.py --scheduler both --diversity medium --batch-size 28 --cache-gb 0.2
  python run_task2.py --suite main
  python run_task2.py --plot-only --stem task2_main_cache0.2

Requires vLLM with extended /v2 API (restart after updating custom_api.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BENCHMARK_DIR.parent
_RESULTS_DIR = _BENCHMARK_DIR / "results"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from request_generator import (  # noqa: E402
    BatchSchedule,
    DiversityLevel,
    InferenceRequest,
    build_task2_schedule,
    load_contexts,
)

SchedulerMode = Literal["baseline", "context_grouped"]


def _stem(
    *,
    suite: str | None,
    diversity: str,
    batch_size: int,
    context_set: str,
    cache_gb: float,
) -> str:
    tag = suite or f"div-{diversity}_bs{batch_size}_{context_set}"
    return f"task2_{tag}_cache{cache_gb:g}"


# [Task2] Build HTTP payload for /v2/batch/chat/completions
def _batch_payload(
    batch: list[InferenceRequest],
    scheduler: SchedulerMode,
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "scheduler": scheduler,
        "requests": [
            {
                "request_id": r.request_id,
                "context_id": r.context_id,
                "messages": r.to_messages(),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            for r in batch
        ],
    }


def _summarize_batch_response(data: dict[str, Any]) -> dict[str, Any]:
    all_rows = data.get("results", [])
    results = [r for r in all_rows if r.get("success")]
    failed = [r for r in all_rows if not r.get("success")]
    n = len(results)
    first_error = failed[0].get("error") if failed else None
    base = {
        "num_success": n,
        "num_failed": len(failed),
        "throughput_req_per_sec": float(data.get("throughput_req_per_sec", 0.0)),
        "total_wall_time_sec": float(data.get("total_wall_time_sec", 0.0)),
        "adjacent_same_context_pairs": data.get("adjacent_same_context_pairs"),
        "execution_order": data.get("execution_order"),
        "first_error": first_error,
    }
    if not n:
        base.update({"avg_response_time_sec": 0.0, "avg_ttft_sec": 0.0})
        return base
    base.update(
        {
            "avg_response_time_sec": sum(r["response_time_sec"] for r in results) / n,
            "avg_ttft_sec": sum(r["ttft_sec"] for r in results) / n,
        }
    )
    return base


def _probe_vllm_up(api_base: str) -> None:
    """Raise SystemExit if vLLM is not accepting connections."""
    url = f"{api_base.rstrip('/')}/models"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"vLLM is not running at {api_base} ({exc}).\n"
            "Restart LMCache server + vLLM before continuing."
        ) from exc
    if resp.status_code != 200:
        raise SystemExit(f"GET {url} -> HTTP {resp.status_code}")


# [Task2] POST one batch to /v2/batch/chat/completions with chosen scheduler mode
def run_batch(
    api_base: str,
    scheduler: SchedulerMode,
    batch: list[InferenceRequest],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/batch/chat/completions"
    payload = _batch_payload(batch, scheduler, max_tokens=max_tokens, temperature=temperature)
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Lost connection to vLLM during batch ({exc}).\n"
            "The previous scheduler run likely crashed the server (OOM / engine exit).\n"
            "Restart vLLM, then re-run with a smaller --batch-size or --context-set short,\n"
            "or run one scheduler at a time: --scheduler baseline"
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"POST {url} -> HTTP {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


# [Task2] Run all batches for each scheduler mode, aggregate results
def run_experiment(
    api_base: str,
    batch_sched: BatchSchedule,
    schedulers: list[SchedulerMode],
    *,
    max_tokens: int,
    temperature: float,
    cache_gb: float,
    timeout: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "summary": {
            "experiment": "task2",
            "local_cache_gb": cache_gb,
            "diversity_level": batch_sched.diversity_level,
            "context_set": batch_sched.context_set,
            "batch_size": batch_sched.batch_size,
            "num_batches": len(batch_sched.batches),
            "num_requests": batch_sched.num_requests,
            "notes": batch_sched.notes,
        },
        "runs": {},
    }
    for mode in schedulers:
        _probe_vllm_up(api_base)
        print(f"\n=== scheduler: {mode} ===")
        batch_runs: list[dict[str, Any]] = []
        for bi, batch in enumerate(batch_sched.batches):
            print(f"  batch {bi + 1}/{len(batch_sched.batches)} ({len(batch)} requests)")
            data = run_batch(
                api_base,
                mode,
                batch,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            summary = _summarize_batch_response(data)
            print(
                f"    OK {summary['num_success']}/{len(batch)}  "
                f"{summary['throughput_req_per_sec']:.2f} req/s  "
                f"avg RT {summary['avg_response_time_sec']:.3f}s  "
                f"adjacent same-ctx {summary.get('adjacent_same_context_pairs')}"
            )
            if summary["num_success"] == 0 and summary.get("first_error"):
                print(f"    ERROR (first): {summary['first_error'][:300]}")
            if summary["num_success"] == 0:
                print(
                    "    WARN: all requests failed. If vLLM logs show shutdown/OOM, "
                    "restart and retry with --batch-size 4 --context-set short"
                )
            batch_runs.append(
                {
                    "batch_index": bi,
                    "scheduler_response": data,
                    "summary": summary,
                }
            )
        # Aggregate across batches for this scheduler mode.
        all_results = [
            r
            for br in batch_runs
            for r in br["scheduler_response"].get("results", [])
            if r.get("success")
        ]
        n = len(all_results)
        agg = {
            "num_success": n,
            "num_failed": sum(
                br["summary"]["num_failed"] for br in batch_runs
            ),
            "throughput_req_per_sec": (
                n
                / sum(br["summary"].get("total_wall_time_sec", 0.0) for br in batch_runs)
                if batch_runs
                and sum(br["summary"].get("total_wall_time_sec", 0.0) for br in batch_runs) > 0
                else 0.0
            ),
            "avg_response_time_sec": (
                sum(r["response_time_sec"] for r in all_results) / n if n else 0.0
            ),
            "avg_ttft_sec": (
                sum(r["ttft_sec"] for r in all_results) / n if n else 0.0
            ),
        }
        out["runs"][mode] = {
            "aggregate": agg,
            "batches": batch_runs,
        }
    return out


def _print_table(payload: dict[str, Any]) -> None:
    print("\n--- Task 2 summary ---")
    print(f"  {'scheduler':<18} {'n':>4} {'req/s':>8} {'RT(s)':>8} {'TTFT(s)':>8}")
    for mode, run in payload.get("runs", {}).items():
        s = run["aggregate"]
        print(
            f"  {mode:<18} {s['num_success']:>4} "
            f"{s['throughput_req_per_sec']:>8.3f} "
            f"{s['avg_response_time_sec']:>8.3f} "
            f"{s['avg_ttft_sec']:>8.3f}"
        )


# [Task2] Plot baseline vs context_grouped: throughput + avg latency
def plot_compare(payload: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = payload.get("runs", {})
    modes = [m for m in ("baseline", "context_grouped") if m in runs]
    if len(modes) < 2:
        print("  [plot] need baseline and context_grouped in JSON")
        return

    labels = modes
    through = [runs[m]["aggregate"]["throughput_req_per_sec"] for m in modes]
    resp = [runs[m]["aggregate"]["avg_response_time_sec"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, through, color=["steelblue", "seagreen"])
    axes[0].set_ylabel("Throughput (req/s)")
    axes[0].set_title("Batch throughput")
    for i, v in enumerate(through):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, resp, color=["coral", "darkorange"])
    axes[1].set_ylabel("Avg full response time (s)")
    axes[1].set_title("Avg latency")
    for i, v in enumerate(resp):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    s = payload.get("summary", {})
    fig.suptitle(
        f"Task 2: {s.get('diversity_level')} diversity, "
        f"batch={s.get('batch_size')}, cache={s.get('local_cache_gb')} GB",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def _grouped_adjacent_pairs_total(payload: dict[str, Any]) -> int:
    """Sum adjacent_same_context_pairs across HTTP batches (context_grouped only)."""
    run = payload.get("runs", {}).get("context_grouped", {})
    total = 0
    for br in run.get("batches", []):
        resp = br.get("scheduler_response") or {}
        total += int(resp.get("adjacent_same_context_pairs") or 0)
    return total


def _load_suite_json(stem: str) -> dict[str, Any]:
    path = _RESULTS_DIR / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


# [Task2] Batch-size sweep: throughput + latency + speedup% across batch sizes
def plot_suite_batch_size(
    *,
    cache_gb: float,
    batch_sizes: list[int] | None = None,
    diversity: str = "medium",
    context_set: str = "all",
) -> Path:
    """Overlay batch-size sweep: throughput/latency vs bs + scheduler speedup."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch_sizes = batch_sizes or list(SUITE_BATCH["batch_sizes"])
    base_through: list[float] = []
    grp_through: list[float] = []
    base_rt: list[float] = []
    grp_rt: list[float] = []
    adj_pairs: list[int] = []
    speedup_pct: list[float] = []

    for bs in batch_sizes:
        stem = f"task2_bs{bs}_cache{cache_gb:g}"
        payload = _load_suite_json(stem)
        b = payload["runs"]["baseline"]["aggregate"]
        g = payload["runs"]["context_grouped"]["aggregate"]
        bt = float(b["throughput_req_per_sec"])
        gt = float(g["throughput_req_per_sec"])
        base_through.append(bt)
        grp_through.append(gt)
        base_rt.append(float(b["avg_response_time_sec"]))
        grp_rt.append(float(g["avg_response_time_sec"]))
        adj_pairs.append(_grouped_adjacent_pairs_total(payload))
        speedup_pct.append(100.0 * (gt - bt) / bt if bt > 0 else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = batch_sizes

    axes[0].plot(x, base_through, "o-", label="baseline", color="steelblue", linewidth=2)
    axes[0].plot(x, grp_through, "s-", label="context_grouped", color="seagreen", linewidth=2)
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Throughput (req/s)")
    axes[0].set_title("Throughput vs batch size")
    axes[0].set_xticks(x)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, base_rt, "o-", label="baseline", color="coral", linewidth=2)
    axes[1].plot(x, grp_rt, "s-", label="context_grouped", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Avg full response time (s)")
    axes[1].set_title("Avg latency vs batch size")
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    colors = ["#888888" if p <= 0 else "seagreen" for p in speedup_pct]
    axes[2].bar([str(b) for b in x], speedup_pct, color=colors)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Batch size")
    axes[2].set_ylabel("Throughput gain (%)")
    axes[2].set_title("Grouped vs baseline (throughput)")
    for i, (p, a) in enumerate(zip(speedup_pct, adj_pairs)):
        axes[2].text(i, p, f"{p:+.1f}%\nadj={a}", ha="center", va="bottom" if p >= 0 else "top", fontsize=8)

    fig.suptitle(
        f"Task 2 batch sweep: {diversity} diversity, {context_set} contexts, cache={cache_gb:g} GB",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = _RESULTS_DIR / f"task2_suite_batch_cache{cache_gb:g}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")
    return out_png


# [Task2] Experiment suite presets
SUITE_MAIN = {  # Single run: medium diversity, all 28 requests in 1 batch
    "diversity": "medium",
    "batch_size": 28,
    "context_set": "all",
}

SUITE_CACHE = {
    "diversity": "medium",
    "batch_size": 28,
    "context_set": "all",
    "cache_sizes": [0.05, 0.1, 0.2, 0.4],
}

SUITE_BATCH = {  # Batch-size sweep: how batch size affects grouped scheduler benefit
    "diversity": "medium",
    "context_set": "all",
    "batch_sizes": [4, 7, 14, 28],
}

SUITE_DIVERSITY = {  # Diversity sweep: grouped scheduler effect under low/medium/high
    "batch_size": 28,
    "context_set": "all",
    "levels": ["low", "medium", "high"],
}

SUITE_CONTEXT = {  # Context-set sweep: short / all / long documents
    "diversity": "medium",
    "batch_size": 28,
    "context_sets": ["short", "all", "long"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scheduler",
        choices=["baseline", "context_grouped", "both"],
        default="both",
    )
    p.add_argument("--suite", choices=["main", "cache", "batch", "diversity", "context"], default=None)
    p.add_argument("--diversity", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--batch-size", type=int, default=28)
    p.add_argument("--context-set", choices=["all", "short", "long"], default="all")
    p.add_argument("--cache-gb", type=float, default=0.2, help="Label; must match configuration.yaml")
    p.add_argument("--data-dir", type=Path, default=_PROJECT_DIR / "frontend" / "data")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v2")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument(
        "--plot-suite",
        choices=["batch"],
        default=None,
        help="Plot combined figure from suite JSONs (no new experiments)",
    )
    p.add_argument("--stem", default=None, help="Results stem for --plot-only")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--exclude", nargs="*", default=["sample"])
    return p.parse_args()


def _probe_batch_api(api_base: str) -> None:
    """Fail fast if /v2/batch/chat/completions is missing (restart vLLM after code update)."""
    url = f"{api_base.rstrip('/')}/batch/chat/completions"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"scheduler": "baseline", "requests": []})
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Cannot connect to {api_base}. Is vLLM running? ({exc})"
        ) from exc
    if resp.status_code == 404:
        raise SystemExit(
            f"Missing {url} (HTTP 404). Sync custom_api.py to the server and restart vLLM."
        )
    # Empty/minimal body: 422 (Pydantic) or 400 — both mean the route exists.
    if resp.status_code in (400, 422):
        return
    if resp.status_code != 200:
        raise SystemExit(f"Unexpected {url} -> HTTP {resp.status_code}: {resp.text[:300]}")


def _run_single_config(args: argparse.Namespace, *, suite_tag: str | None = None) -> Path:
    _probe_batch_api(args.api_base)
    contexts = load_contexts(args.data_dir, exclude=args.exclude)
    batch_sched = build_task2_schedule(
        contexts,
        diversity_level=args.diversity,
        batch_size=args.batch_size,
        context_set=args.context_set,
        seed=args.seed,
        data_dir=str(args.data_dir),
        count_tokens=True,
    )
    stem = _stem(
        suite=suite_tag,
        diversity=args.diversity,
        batch_size=args.batch_size,
        context_set=args.context_set,
        cache_gb=args.cache_gb,
    )
    json_path = _RESULTS_DIR / f"{stem}.json"
    png_path = _RESULTS_DIR / f"{stem}.png"

    print(batch_sched.notes)
    if args.dry_run:
        batch_sched.save(_RESULTS_DIR / f"{stem}_schedule.json")
        print(f"Dry-run schedule -> {_RESULTS_DIR / f'{stem}_schedule.json'}")
        return json_path

    schedulers: list[SchedulerMode] = (
        ["baseline", "context_grouped"] if args.scheduler == "both" else [args.scheduler]  # type: ignore[list-item]
    )
    payload = run_experiment(
        args.api_base,
        batch_sched,
        schedulers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        cache_gb=args.cache_gb,
        timeout=args.timeout,
    )
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")
    _print_table(payload)
    if not args.no_plot:
        plot_compare(payload, png_path)
    return json_path


def main() -> None:
    args = parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_suite == "batch":
        plot_suite_batch_size(
            cache_gb=args.cache_gb,
            diversity=args.diversity,
            context_set=args.context_set,
        )
        return

    if args.plot_only:
        stem = args.stem or _stem(
            suite=None,
            diversity=args.diversity,
            batch_size=args.batch_size,
            context_set=args.context_set,
            cache_gb=args.cache_gb,
        )
        json_path = _RESULTS_DIR / f"{stem}.json"
        if not json_path.is_file():
            raise SystemExit(f"Missing {json_path}")
        plot_compare(json.loads(json_path.read_text(encoding="utf-8")), json_path.with_suffix(".png"))
        return

    if args.suite == "main":
        args.diversity = SUITE_MAIN["diversity"]  # type: ignore[assignment]
        args.batch_size = SUITE_MAIN["batch_size"]
        args.context_set = SUITE_MAIN["context_set"]  # type: ignore[assignment]
        _run_single_config(args, suite_tag="main")
        return

    if args.suite == "cache":
        for gb in SUITE_CACHE["cache_sizes"]:
            print(f"\n######## cache {gb} GB (set yaml + restart vLLM, then Enter) ########")
            input()
            args.cache_gb = gb
            args.diversity = SUITE_CACHE["diversity"]  # type: ignore[assignment]
            args.batch_size = SUITE_CACHE["batch_size"]
            args.context_set = SUITE_CACHE["context_set"]  # type: ignore[assignment]
            _run_single_config(args, suite_tag=f"cache{gb:g}")
        return

    if args.suite == "batch":
        for bs in SUITE_BATCH["batch_sizes"]:
            args.batch_size = bs
            args.diversity = SUITE_BATCH["diversity"]  # type: ignore[assignment]
            args.context_set = SUITE_BATCH["context_set"]  # type: ignore[assignment]
            _run_single_config(args, suite_tag=f"bs{bs}")
        plot_suite_batch_size(
            cache_gb=args.cache_gb,
            diversity=SUITE_BATCH["diversity"],
            context_set=SUITE_BATCH["context_set"],
        )
        return

    if args.suite == "diversity":
        for level in SUITE_DIVERSITY["levels"]:
            args.diversity = level  # type: ignore[assignment]
            args.batch_size = SUITE_DIVERSITY["batch_size"]
            args.context_set = SUITE_DIVERSITY["context_set"]  # type: ignore[assignment]
            _run_single_config(args, suite_tag=f"div-{level}")
        return

    if args.suite == "context":
        for cs in SUITE_CONTEXT["context_sets"]:
            args.context_set = cs  # type: ignore[assignment]
            args.diversity = SUITE_CONTEXT["diversity"]  # type: ignore[assignment]
            args.batch_size = SUITE_CONTEXT["batch_size"]
            _run_single_config(args, suite_tag=f"ctx-{cs}")
        return

    _run_single_config(args)


if __name__ == "__main__":
    main()
