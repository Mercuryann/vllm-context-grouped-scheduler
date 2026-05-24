#!/usr/bin/env python3
"""
IK2221 Task 3: RAG retrieval + scheduler-aware inference benchmark.

Pipeline:
  1. Build a TF-IDF embedding index from frontend/data/*.txt.
  2. Generate evaluation questions with a known expected context.
  3. Retrieve the most similar context for each question.
  4. Send the classified requests to the Task 2 batch endpoint.
  5. Report retrieval accuracy plus latency/throughput for baseline vs grouped scheduling.

Examples:
  python benchmark/run_task3.py --retrieve-only
  python benchmark/run_task3.py --scheduler both --batch-size 28 --cache-gb 0.2
  python benchmark/run_task3.py --plot-only --stem task3_rag_cache0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BENCHMARK_DIR.parent
_RESULTS_DIR = _BENCHMARK_DIR / "results"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from rag_retriever import TfidfRagIndex, load_text_contexts  # noqa: E402
from request_generator import InferenceRequest  # noqa: E402


# [Task3] Question templates — {keywords} is replaced with top TF-IDF terms from each paper
QUERY_TEMPLATES = [
    "What is the main contribution of the work about {keywords}?",
    "What problem is addressed by the paper discussing {keywords}?",
]

# [Task3] Stores one RAG evaluation result: expected vs retrieved context + score
@dataclass(frozen=True)
class RagQuery:
    request_id: str
    expected_context_id: str
    question: str
    retrieved_context_id: str
    retrieval_score: float
    retrieval_rank: int
    top_contexts: list[dict[str, float | str | int]]

    @property
    def correct(self) -> bool:
        return self.expected_context_id == self.retrieved_context_id


# [Task3] Generate evaluation queries: extract top keywords per paper, fill templates, then retrieve
def _make_eval_queries(
    contexts: dict[str, str],
    index: TfidfRagIndex,
    *,
    questions_per_context: int,
    top_k: int,
) -> list[RagQuery]:
    queries: list[RagQuery] = []
    for context_id in sorted(contexts):
        terms = index.top_terms(context_id, n=6)  # [Task3] top-6 TF-IDF keywords for this paper
        if not terms:
            terms = [context_id.replace("-", " ")]
        for i in range(questions_per_context):
            selected = terms[i : i + 3] or terms[:3]
            keywords = " ".join(selected)
            template = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)]
            question = template.format(keywords=keywords)
            hits = index.search(question, top_k=top_k)  # [Task3] retrieve top-k matching papers
            best = hits[0]  # [Task3] top-1 hit = our selected context
            queries.append(
                RagQuery(
                    request_id=f"rag-{len(queries):04d}",
                    expected_context_id=context_id,
                    question=question,
                    retrieved_context_id=best.context_id,
                    retrieval_score=best.score,
                    retrieval_rank=best.rank,
                    top_contexts=[
                        {
                            "context_id": hit.context_id,
                            "score": hit.score,
                            "rank": hit.rank,
                        }
                        for hit in hits
                    ],
                )
            )
    return queries


# [Task3] Convert RAG results → InferenceRequest using the *retrieved* (not expected) context
def _to_inference_requests(
    rag_queries: list[RagQuery],
    contexts: dict[str, str],
) -> list[InferenceRequest]:
    requests: list[InferenceRequest] = []
    for i, item in enumerate(rag_queries):
        context_text = contexts[item.retrieved_context_id]  # [Task3] use retrieved paper, not expected
        requests.append(
            InferenceRequest(
                request_id=item.request_id,
                context_id=item.retrieved_context_id,
                question=item.question,
                context_text=context_text,
                sequence_index=i,
                experiment="rag",
                visit_type="first",
            )
        )
    return requests


def _batch_requests(
    requests: list[InferenceRequest],
    batch_size: int,
) -> list[list[InferenceRequest]]:
    batch_size = max(1, min(batch_size, len(requests)))
    return [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]


def _retrieval_summary(rag_queries: list[RagQuery]) -> dict[str, Any]:
    n = len(rag_queries)
    correct = sum(1 for item in rag_queries if item.correct)
    return {
        "num_queries": n,
        "num_correct_top1": correct,
        "top1_accuracy": correct / n if n else 0.0,
        "avg_top1_score": (
            sum(item.retrieval_score for item in rag_queries) / n if n else 0.0
        ),
    }


def _stem(cache_gb: float) -> str:
    return f"task3_rag_cache{cache_gb:g}"


def _print_retrieval_table(rag_queries: list[RagQuery], *, limit: int = 12) -> None:
    print("\n--- RAG retrieval sample ---")
    print(f"  {'expected':<22} {'retrieved':<22} {'score':>8}  question")
    for item in rag_queries[:limit]:
        mark = "" if item.correct else " *"
        print(
            f"  {item.expected_context_id:<22} "
            f"{item.retrieved_context_id:<22} "
            f"{item.retrieval_score:>8.3f}{mark}  "
            f"{item.question[:70]}"
        )
    if len(rag_queries) > limit:
        print(f"  ... {len(rag_queries) - limit} more")
    print("  * means the retriever selected a different context than expected.")


def _plot(payload: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    retrieval = payload["retrieval_summary"]
    runs = payload.get("generation", {}).get("runs", {})
    modes = [m for m in ("baseline", "context_grouped") if m in runs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(["top-1"], [retrieval["top1_accuracy"]], color="steelblue")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Retrieval accuracy")
    axes[0].set_title("RAG retrieval")
    axes[0].text(
        0,
        retrieval["top1_accuracy"],
        f"{retrieval['top1_accuracy']:.2f}",
        ha="center",
        va="bottom",
    )

    if modes:
        through = [runs[m]["aggregate"]["throughput_req_per_sec"] for m in modes]
        resp = [runs[m]["aggregate"]["avg_response_time_sec"] for m in modes]
        axes[1].bar(modes, through, color=["coral", "seagreen"][: len(modes)])
        axes[1].set_ylabel("Throughput (req/s)")
        axes[1].set_title("Generation throughput")
        axes[2].bar(modes, resp, color=["darkorange", "seagreen"][: len(modes)])
        axes[2].set_ylabel("Avg full response time (s)")
        axes[2].set_title("Generation latency")
    else:
        axes[1].axis("off")
        axes[2].axis("off")

    s = payload["summary"]
    fig.suptitle(
        f"Task 3 RAG: {s['num_contexts']} contexts, "
        f"{s['num_queries']} queries, cache={s['local_cache_gb']} GB",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=_PROJECT_DIR / "frontend" / "data")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v2")
    p.add_argument("--scheduler", choices=["baseline", "context_grouped", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=28)
    p.add_argument("--questions-per-context", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--cache-gb", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep RAG-classified requests in context order before batch scheduling.",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--retrieve-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--stem", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--exclude", nargs="*", default=["sample"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.stem or _stem(args.cache_gb)
    json_path = _RESULTS_DIR / f"{stem}.json"
    png_path = _RESULTS_DIR / f"{stem}.png"

    if args.plot_only:
        if not json_path.is_file():
            raise SystemExit(f"Missing {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        _plot(payload, png_path)
        return

    # [Task3] Step 1: load papers + build TF-IDF index
    contexts = load_text_contexts(args.data_dir, exclude=args.exclude)
    index = TfidfRagIndex(contexts)
    # [Task3] Step 2: generate eval questions and retrieve matching contexts
    rag_queries = _make_eval_queries(
        contexts,
        index,
        questions_per_context=args.questions_per_context,
        top_k=args.top_k,
    )
    # [Task3] Step 3: compute retrieval accuracy
    retrieval = _retrieval_summary(rag_queries)
    _print_retrieval_table(rag_queries)
    print(
        f"\nRetrieval top-1 accuracy: "
        f"{retrieval['num_correct_top1']}/{retrieval['num_queries']} "
        f"({retrieval['top1_accuracy']:.2%})"
    )

    payload: dict[str, Any] = {
        "summary": {
            "experiment": "task3_rag",
            "data_dir": str(args.data_dir),
            "num_contexts": len(contexts),
            "num_queries": len(rag_queries),
            "questions_per_context": args.questions_per_context,
            "top_k": args.top_k,
            "local_cache_gb": args.cache_gb,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "shuffled_before_generation": not args.no_shuffle,
        },
        "retrieval_summary": retrieval,
        "retrieval_results": [asdict(item) | {"correct": item.correct} for item in rag_queries],
        "generation": None,
    }

    if not args.retrieve_only:
        import run_task2 as task2

        task2._probe_batch_api(args.api_base)
#        requests = _to_inference_requests(rag_queries, contexts)
        generation_queries = list(rag_queries)
        if not args.no_shuffle:
            random.Random(args.seed).shuffle(generation_queries)  # [Task3] shuffle before batching
        payload["generation_input_order"] = [
            {
                "request_id": item.request_id,
                "expected_context_id": item.expected_context_id,
                "retrieved_context_id": item.retrieved_context_id,
            }
            for item in generation_queries
        ]
        # [Task3] Step 4: convert RAG results → InferenceRequest, split into batches
        requests = _to_inference_requests(generation_queries, contexts)
        batches = _batch_requests(requests, args.batch_size)
        schedulers = (
            ["baseline", "context_grouped"]
            if args.scheduler == "both"
            else [args.scheduler]
        )
        batch_sched = task2.BatchSchedule(
            batches=batches,
            diversity_level="rag_retrieved",
            context_set="retrieved",
            batch_size=max(1, min(args.batch_size, len(requests))),
            data_dir=str(args.data_dir),
            notes=(
                "Task3 RAG: contexts were selected by TF-IDF retrieval, then "
                "the resulting classified requests were submitted to the Task2 scheduler."
            ),
        )
        # [Task3] Step 5: reuse Task 2's run_experiment() for batch inference
        payload["generation"] = task2.run_experiment(
            args.api_base,
            batch_sched,
            schedulers,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            cache_gb=args.cache_gb,
            timeout=args.timeout,
        )
        task2._print_table(payload["generation"])

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")
    if not args.no_plot:
        _plot(payload, png_path)


if __name__ == "__main__":
    main()