#!/usr/bin/env python3
"""
IK2221 Task 1 — one command per assignment question, plot when done.

  python run_task1.py q1   # Q1: prompt length vs response time  → results/q1_cache0.2.png
  python run_task1.py q2   # Q2: first → immediate repeat (full response time)
  python run_task1.py q3   # Q3: diversity low/medium/high → q3_cache0.2.png

Gap repeat (separate script):  python benchmark/run_repeat_gap.py

Prerequisites (project root, venv active):
  1. LMCache server on :65432
  2. vLLM on :8000 with LMCACHE_CONFIG_FILE=.../configuration.yaml

Q2 cache sweep: change max_local_cache_size, restart vLLM, re-run q2 with another --cache-gb.
  When >=2 q2 result files exist, also writes results/q2_cache_sweep.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BENCHMARK_DIR.parent
_RESULTS_DIR = _BENCHMARK_DIR / "results"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from request_generator import (  # noqa: E402
    InferenceRequest,
    RequestSchedule,
    _adjacent_same_context_count,
    build_schedule,
)

# ---------------------------------------------------------------------------
# [Task1] Experiment presets — each question maps to one subcommand
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "q1": {  # [Task1 Q1] length experiment: 1 request per paper, sorted by prompt length
        "kind": "length",
        "requests_per_context": 1,
        "warmup": 1,
        "max_tokens": 64,
        "temperature": 0.0,
        "title": "Q1: Full response time vs prompt length",
        "plot_metric": "response_time",
    },
    "q2": {  # [Task1 Q2] repeat experiment: first + immediate exact_repeat per pair
        "kind": "repeat",
        "requests_per_context": 2,
        "warmup": 0,
        "max_tokens": 64,
        "temperature": 0.0,
        "title": "Q2: Back-to-back exact repeat (full response time)",
        "plot_metric": "response_time",
    },
    "q3": {  # [Task1 Q3] diversity experiment: low/medium/high ordering
        "kind": "diversity",
        "requests_per_context": 2,
        "warmup": 0,
        "max_tokens": 64,
        "temperature": 0.0,
        "title": "Q3: Request diversity (28 reqs per level)",
        "levels": ("low", "medium", "high"),
        "plot_metric": "response_time",
    },
}


@dataclass
class RequestResult:
    request_id: str
    context_id: str
    question: str
    sequence_index: int
    experiment: str
    visit_type: str
    prompt_token_count: int | None
    source_request_id: str | None
    ttft_sec: float  # first token (matches frontend TTFT)
    response_time_sec: float  # full stream end-to-end (assignment "request latency")
    prompt_tokens_api: int | None
    completion_tokens: int | None
    success: bool
    error: str | None = None


def _result_paths(question: str, cache_gb: float) -> tuple[Path, Path]:
    stem = f"{question}_cache{cache_gb:g}"
    return _RESULTS_DIR / f"{stem}.json", _RESULTS_DIR / f"{stem}.png"


def _prompt_tokens(r: dict) -> int | None:
    return r.get("prompt_tokens_api") or r.get("prompt_token_count")


def _ttft_sec(r: dict) -> float:
    return float(r.get("ttft_sec", r.get("latency_sec", 0)))


def _response_time_sec(r: dict) -> float:
    """Full stream end-to-end time (prefill + decode). Do not fall back to TTFT."""
    val = r.get("response_time_sec")
    if val is not None:
        return float(val)
    if r.get("latency_sec") is not None:
        return float(r["latency_sec"])
    raise KeyError("response_time_sec")


def _metric_value(r: dict, metric: str) -> float:
    if metric == "ttft":
        return _ttft_sec(r)
    return _response_time_sec(r)


def _ylabel_for_metric(metric: str, *, average: bool = False) -> str:
    if metric == "response_time":
        return "Avg full response time (s)" if average else "Full response time (s)"
    return "Avg TTFT (s)" if average else "TTFT (s)"


def _summary_avg_first_repeat(
    summary: dict, metric: str
) -> tuple[float | None, float | None]:
    if metric == "response_time":
        return (
            summary.get("avg_response_time_first_sec"),
            summary.get("avg_response_time_repeat_sec"),
        )
    return (
        summary.get("avg_ttft_first_sec", summary.get("avg_latency_first_sec")),
        summary.get("avg_ttft_repeat_sec", summary.get("avg_latency_repeat_sec")),
    )


def _has_response_time(rows: list[dict]) -> bool:
    return any(r.get("response_time_sec") is not None for r in rows)

# [Task1] Core: send one request via streaming, record TTFT + full response time
def run_single_request(
    client: OpenAI,
    model: str,
    req: InferenceRequest,
    *,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    """Stream request; record TTFT and full response time."""
    t0 = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=req.to_messages(),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stop=["\n"],
        )
        ttft: float | None = None
        n_completion = 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0#time to first token
                n_completion += len(delta)

        response_time = time.perf_counter() - t0 #full response time
        if ttft is None:
            ttft = response_time

        return RequestResult(
            request_id=req.request_id,
            context_id=req.context_id,
            question=req.question,
            sequence_index=req.sequence_index,
            experiment=req.experiment,
            visit_type=req.visit_type,
            prompt_token_count=req.prompt_token_count,
            source_request_id=req.source_request_id,
            ttft_sec=ttft,
            response_time_sec=response_time,
            prompt_tokens_api=None,
            completion_tokens=n_completion or None,
            success=True,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return RequestResult(
            request_id=req.request_id,
            context_id=req.context_id,
            question=req.question,
            sequence_index=req.sequence_index,
            experiment=req.experiment,
            visit_type=req.visit_type,
            prompt_token_count=req.prompt_token_count,
            source_request_id=req.source_request_id,
            ttft_sec=elapsed,
            response_time_sec=elapsed,
            prompt_tokens_api=None,
            completion_tokens=None,
            success=False,
            error=str(exc),
        )


def _summarize(results: list[RequestResult], wall: float) -> dict:
    ok = [r for r in results if r.success]
    first = [r for r in ok if r.visit_type == "first"]
    exact = [r for r in ok if r.visit_type == "exact_repeat"]

    def avg_ttft(xs: list[RequestResult]) -> float | None:
        return sum(r.ttft_sec for r in xs) / len(xs) if xs else None

    def avg_resp(xs: list[RequestResult]) -> float | None:
        return sum(r.response_time_sec for r in xs) / len(xs) if xs else None

    return {
        "num_success": len(ok),
        "num_failed": len(results) - len(ok),
        "total_wall_time_sec": wall,
        "throughput_req_per_sec": len(ok) / wall if wall > 0 else 0.0,
        "avg_ttft_sec": avg_ttft(ok),
        "avg_ttft_first_sec": avg_ttft(first),
        "avg_ttft_repeat_sec": avg_ttft(exact),
        "avg_response_time_sec": avg_resp(ok),
        "avg_response_time_first_sec": avg_resp(first),
        "avg_response_time_repeat_sec": avg_resp(exact),
    }


# [Task1] Run all requests in a schedule sequentially, skip warmup, collect results
def run_schedule(
    schedule: RequestSchedule,
    *,
    api_base: str,
    max_tokens: int,
    temperature: float,
    warmup: int,  # skip first N requests (cold start)
    question: str,
    cache_gb: float,
    diversity_level: str | None = None,
    plot_metric: str | None = None,
) -> dict:
    client = OpenAI(api_key="EMPTY", base_url=api_base)
    model = client.models.list().data[0].id
    results: list[RequestResult] = []
    t0 = time.perf_counter()

    for i, req in enumerate(schedule.requests):
        print(
            f"  [{i + 1}/{len(schedule.requests)}] {req.context_id} "
            f"({req.visit_type})"
        )
        res = run_single_request(
            client, model, req, max_tokens=max_tokens, temperature=temperature
        )
        if i >= warmup:
            results.append(res)
        if res.success:
            tok = res.prompt_token_count or res.prompt_tokens_api or "?"
            print(
                f"       TTFT {res.ttft_sec:.3f}s  "
                f"response {res.response_time_sec:.3f}s  tokens={tok}"
            )
        else:
            print(f"       FAIL: {res.error}")

    wall = time.perf_counter() - t0
    summary = _summarize(results, wall)
    summary.update(
        {
            "question": question,
            "plot_metric": plot_metric,
            "model": model,
            "api_base": api_base,
            "local_cache_gb": cache_gb,
            "experiment_kind": schedule.experiment,
            "diversity_level": diversity_level or schedule.diversity_level,
            "num_requests": len(results),
            "warmup_skipped": warmup,
        }
    )
    return {"summary": summary, "results": [asdict(r) for r in results]}


# ---------------------------------------------------------------------------
# Plots (called immediately after each experiment)
# ---------------------------------------------------------------------------


def plot_q1(payload: dict, out_png: Path) -> None:
    """Q1: prompt length vs full response time (stream end), not TTFT."""
    metric = PRESETS["q1"].get("plot_metric", "response_time")
    rows = [r for r in payload["results"] if r["success"]]
    if metric == "response_time" and not any(r.get("response_time_sec") is not None for r in rows):
        print(
            "  [plot] FAILED: JSON has no response_time_sec (old Q1 run used TTFT only). "
            "Re-run: python run_task1.py q1 --cache-gb <GB>"
        )
        return

    xs, ys, labels = [], [], []
    for r in rows:
        t = _prompt_tokens(r)
        if t is None:
            continue
        try:
            ys.append(_metric_value(r, metric))
        except KeyError:
            print(f"  [plot] skip {r.get('context_id')}: missing response_time_sec")
            continue
        xs.append(t)
        labels.append(r["context_id"])

    if not xs:
        print("  [plot] skipped: no token counts")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys, s=80, alpha=0.8, c="tab:blue", edgecolors="k", linewidths=0.3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")

    if len(xs) >= 2:
        import numpy as np

        coef = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, np.polyval(coef, xline), "r--", alpha=0.6, label="linear fit")

    ylabel = "Full response time (s)" if metric == "response_time" else "TTFT (s)"
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(PRESETS["q1"]["title"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def plot_q2(payload: dict, out_png: Path) -> None:
    preset = PRESETS["q2"]
    metric = preset.get("plot_metric", "response_time")
    rows = [r for r in payload["results"] if r["success"]]
    if metric == "response_time" and not _has_response_time(rows):
        print(
            "  [plot] FAILED: JSON has no response_time_sec (old Q2 run used TTFT only). "
            "Re-run: python run_task1.py q2 --cache-gb <GB>"
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
        print("  [plot] skipped: no repeat pairs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: paired bars (first 12 pairs)
    show = pairs[:12]
    labels = [p[0]["context_id"][:10] for p in show]
    x = range(len(show))
    w = 0.35
    try:
        first_vals = [_metric_value(p[0], metric) for p in show]
        repeat_vals = [_metric_value(p[1], metric) for p in show]
    except KeyError:
        print("  [plot] FAILED: missing response_time_sec in some rows")
        return
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
    axes[0].set_ylabel(_ylabel_for_metric(metric))
    axes[0].set_title("Per-request: first vs immediate repeat")
    axes[0].legend()

      # Right: aggregate
    s = payload["summary"]
    names = ["first pass", "exact repeat"]
    avg_first, avg_repeat = _summary_avg_first_repeat(s, metric)
    if avg_first is None or avg_repeat is None:
        print("  [plot] FAILED: missing average first/repeat in summary")
        return
    axes[1].bar(names, [avg_first, avg_repeat], color=["tab:blue", "tab:orange"])
    axes[1].set_ylabel(_ylabel_for_metric(metric, average=True))
    axes[1].set_title(
        f"Average (n={len(pairs)} pairs, cache={s.get('local_cache_gb')} GB)"
    )

    fig.suptitle(preset["title"], fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def plot_q3(combined: dict, out_png: Path) -> None:
    runs = combined["runs"]
    labels = []
    through, latency = [], []
    for level in PRESETS["q3"]["levels"]:
        if level not in runs:
            continue
        labels.append(level)
        s = runs[level]["summary"]
        through.append(s["throughput_req_per_sec"])
        metric = PRESETS["q3"].get("plot_metric", "response_time")
        if metric == "ttft":
            latency.append(s.get("avg_ttft_sec", s.get("avg_latency_sec", 0)))
        else:
            latency.append(s.get("avg_response_time_sec", 0))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, through, color="steelblue")
    axes[0].set_ylabel("Throughput (req/s)")
    ylab = "Avg TTFT (s)" if PRESETS["q3"].get("plot_metric") == "ttft" else "Avg response time (s)"
    axes[1].bar(labels, latency, color="coral")
    axes[1].set_ylabel(ylab)
    for ax in axes:
        ax.set_xlabel("Diversity")
    fig.suptitle(PRESETS["q3"]["title"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def _print_diversity_schedule_info(schedule: RequestSchedule) -> None:
    adj = _adjacent_same_context_count(schedule.requests)
    n = len(schedule.requests)
    print(f"  {n} requests | adjacent same-context: {adj}/{max(n - 1, 0)}")
    print(f"  {schedule.notes}")


PLOT_FN = {"q1": plot_q1, "q2": plot_q2, "q3": plot_q3}
# ---------------------------------------------------------------------------
# Run one question
# ---------------------------------------------------------------------------

# [Task1 Q1] Build length schedule, sort by token count, run sequentially
def run_q1(args: argparse.Namespace) -> None:
    preset = PRESETS["q1"]
    # every paper has 1 request
    schedule = build_schedule(
        preset["kind"],
        args.data_dir,
        requests_per_context=preset["requests_per_context"],
        num_contexts=args.num_contexts,
        seed=args.seed,
        exclude=args.exclude,
        count_tokens=not args.no_token_count,
    )
    json_path, png_path = _result_paths("q1", args.cache_gb)
    # Sorted by length 
    schedule.requests.sort(
        key=lambda r: r.prompt_token_count or 0
    )
    for i, r in enumerate(schedule.requests):
        schedule.requests[i] = InferenceRequest(
            request_id=r.request_id,
            context_id=r.context_id,
            question=r.question,
            context_text=r.context_text,
            sequence_index=i,
            experiment=r.experiment,
            visit_type=r.visit_type,
            prompt_token_count=r.prompt_token_count,
            source_request_id=r.source_request_id,
        )

    print(
        f"Q1: {len(schedule.requests)} requests (1 per paper), "
        f"sorted by prompt length + {preset['warmup']} warmup"
    )
    # send requests to the LLM.
    payload = run_schedule(
        schedule,
        api_base=args.api_base,
        max_tokens=preset["max_tokens"],
        temperature=preset["temperature"],
        warmup=preset["warmup"],
        question="q1",
        cache_gb=args.cache_gb,
        plot_metric=preset.get("plot_metric", "response_time"),
    )
    _save_and_plot(payload, json_path, png_path, "q1", plot=not args.no_plot)


# [Task1 Q2] Build repeat schedule (first + immediate repeat), run, then sweep plot
def run_q2(args: argparse.Namespace) -> None:
    preset = PRESETS["q2"]
    schedule = build_schedule(
        preset["kind"],
        args.data_dir,
        requests_per_context=preset["requests_per_context"],
        num_contexts=args.num_contexts,
        seed=args.seed,
        exclude=args.exclude,
        count_tokens=not args.no_token_count,
    )
    json_path, png_path = _result_paths("q2", args.cache_gb)
    n_pairs = sum(1 for r in schedule.requests if r.visit_type == "first")
    print(f"Q2: {len(schedule.requests)} requests ({n_pairs} pairs, first → immediate repeat)")
    print(f"  {schedule.notes}")
    payload = run_schedule(
        schedule,
        api_base=args.api_base,
        max_tokens=preset["max_tokens"],
        temperature=preset["temperature"],
        warmup=preset["warmup"],
        question="q2",
        cache_gb=args.cache_gb,
        plot_metric=preset.get("plot_metric", "response_time"),
    )
    _save_and_plot(payload, json_path, png_path, "q2", plot=not args.no_plot)
    if not args.no_plot:
        plot_q2_cache_sweep(_RESULTS_DIR)  # auto-generates sweep plot if >=2 cache sizes exist


# [Task1 Q2] Sweep plot: throughput + first/repeat latency across different cache sizes
def plot_q2_cache_sweep(results_dir: Path) -> None:
    metric = PRESETS["q2"].get("plot_metric", "response_time")
    files = sorted(results_dir.glob("q2_cache*.json"))
    files = [f for f in files if "sweep" not in f.name]
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
    out = results_dir / "q2_cache_sweep.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, [p[1]["throughput_req_per_sec"] for p in points], "o-", color="steelblue")
    axes[0].set(xlabel="max_local_cache_size (GB)", ylabel="Throughput (req/s)", title="Throughput")
    axes[0].grid(True, alpha=0.3)

    first_y, rep_y = [], []
    for _, s in points:
        af, ar = _summary_avg_first_repeat(s, metric)
        first_y.append(af)
        rep_y.append(ar)
    if not all(x is not None for x in first_y + rep_y):
        print("  [plot] cache sweep skipped: missing response_time averages (re-run Q2)")
        plt.close(fig)
        return
    axes[1].plot(xs, first_y, "o-", label="first", color="tab:blue")
    axes[1].plot(xs, rep_y, "s--", label="repeat", color="tab:orange")
    axes[1].set(
        xlabel="max_local_cache_size (GB)",
        ylabel=_ylabel_for_metric(metric, average=True),
        title="First vs repeat",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Q2: Effect of local cache size")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out} ({len(points)} cache settings)")


# [Task1 Q3] Run diversity experiment: low/medium/high, same 28 requests, different ordering
def run_q3(args: argparse.Namespace) -> None:
    preset = PRESETS["q3"]
    json_path, png_path = _result_paths("q3", args.cache_gb)
    combined: dict = {
        "summary": {
            "question": "q3",
            "local_cache_gb": args.cache_gb,
            "levels": list(preset["levels"]),
        },
        "runs": {},
    }
    print(f"Q3: levels {preset['levels']} (same request count per level)")
    for level in preset["levels"]:
        print(f"\n--- diversity: {level} ---")
        schedule = build_schedule(
            preset["kind"],
            args.data_dir,
            diversity_level=level,
            requests_per_context=preset["requests_per_context"],
            num_contexts=args.num_contexts,
            seed=args.seed,
            exclude=args.exclude,
            count_tokens=False,
        )
        _print_diversity_schedule_info(schedule)
        payload = run_schedule(
            schedule,
            api_base=args.api_base,
            max_tokens=preset["max_tokens"],
            temperature=preset["temperature"],
            warmup=preset["warmup"],
            question="q3",
            cache_gb=args.cache_gb,
            diversity_level=level,
        )
        run = payload
        run["schedule_notes"] = schedule.notes
        combined["runs"][level] = run

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")
    plot_q3(combined, png_path)
    _print_q3_table(combined)


def _save_and_plot(
    payload: dict,
    json_path: Path,
    png_path: Path,
    question: str,
    *,
    plot: bool = True,
) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")
    if plot:
        PLOT_FN[question](payload, png_path)
    _print_summary(payload["summary"], question)


def _print_summary(s: dict, question: str) -> None:
    print(f"\n--- {question.upper()} summary ---")
    print(f"  requests OK : {s['num_success']}")
    print(f"  throughput  : {s['throughput_req_per_sec']:.3f} req/s")
    primary = s.get("plot_metric") or PRESETS.get(question, {}).get("plot_metric")
    if question in ("q1", "q2") or primary == "response_time":
        print(f"  avg full response: {s.get('avg_response_time_sec', 0):.3f} s")
        print(f"  avg TTFT (ref)    : {s.get('avg_ttft_sec', 0):.3f} s")
        rf, rr = _summary_avg_first_repeat(s, "response_time")
        if rf is not None and rr is not None:
            print(f"  RT first/repeat   : {rf:.3f}s / {rr:.3f}s")
            if rr > 0:
                print(f"  RT speedup        : {rf / rr:.2f}x (repeat faster if >1)")
    else:
        print(f"  avg TTFT         : {s.get('avg_ttft_sec', 0):.3f} s")
        print(f"  avg response time: {s.get('avg_response_time_sec', 0):.3f} s")
        rep = s.get("avg_ttft_repeat_sec", s.get("avg_latency_repeat_sec"))
        first = s.get("avg_ttft_first_sec", s.get("avg_latency_first_sec"))
        if rep is not None and first is not None:
            print(f"  TTFT first/repeat : {first:.3f}s / {rep:.3f}s")


def _print_q3_table(combined: dict) -> None:
    print("\n--- Q3 summary (same workload per level) ---")
    print(f"  {'level':<8} {'n':>4} {'req/s':>8} {'resp(s)':>10} {'TTFT(s)':>10}")
    for level, run in combined["runs"].items():
        s = run["summary"]
        n = s.get("num_success", 0)
        resp = s.get("avg_response_time_sec", 0)
        ttft = s.get("avg_ttft_sec", 0)
        print(
            f"  {level:<8} {n:>4} {s['throughput_req_per_sec']:>8.3f} "
            f"{resp:>10.3f} {ttft:>10.3f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IK2221 Task 1: q1 | q2 | q3 (run + plot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("question", choices=["q1", "q2", "q3"], help="Assignment question")
    p.add_argument("--cache-gb", type=float, default=0.2, help="Label for max_local_cache_size in yaml")
    p.add_argument("--data-dir", type=Path, default=_PROJECT_DIR / "frontend" / "data")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v2")
    p.add_argument("--num-contexts", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Only build schedule, no API calls")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-draw PNG from existing results JSON (no API calls)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Save JSON only (faster when batching many cache sizes)",
    )
    p.add_argument("--no-token-count", action="store_true")
    p.add_argument("--exclude", nargs="*", default=["sample"])
    return p.parse_args()


def _plot_only(args: argparse.Namespace) -> None:
    json_path, png_path = _result_paths(args.question, args.cache_gb)
    if not json_path.is_file():
        raise SystemExit(f"Missing {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if args.question == "q3":
        plot_q3(payload, png_path)
        _print_q3_table(payload)
    else:
        PLOT_FN[args.question](payload, png_path)
        if "summary" in payload:
            _print_summary(payload["summary"], args.question)
    if args.question == "q2":
        plot_q2_cache_sweep(_RESULTS_DIR)


def main() -> None:
    args = parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        _plot_only(args)
        return

    if args.no_plot and args.question != "q2":
        print("Note: --no-plot is mainly used with q2 cache sweeps.")

    if args.dry_run:
        preset = PRESETS[args.question]
        if args.question == "q3":
            for level in preset["levels"]:
                s = build_schedule(
                    preset["kind"],
                    args.data_dir,
                    diversity_level=level,
                    requests_per_context=preset["requests_per_context"],
                    seed=args.seed,
                    exclude=args.exclude,
                    count_tokens=False,
                )
                print(f"  {level}: {len(s.requests)} requests — {s.notes}")
        else:
            s = build_schedule(
                preset["kind"],
                args.data_dir,
                requests_per_context=preset["requests_per_context"],
                seed=args.seed,
                exclude=args.exclude,
                count_tokens=False,
            )
            print(f"  {len(s.requests)} requests")
        return

    print(f"Cache label: {args.cache_gb} GB (must match configuration.yaml)\n")
    runners = {"q1": run_q1, "q2": run_q2, "q3": run_q3}
    runners[args.question](args)
    print(f"\nDone. Results in {_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
