"""
Task 1 / Task 2 request generator (IK2221 / Figure 4).

Each request is tagged with context_id and experiment metadata so analysis can answer:
  Q1  latency vs prompt length (tokens)
  Q2  back-to-back exact repeat vs first; effect of local cache size
  Q3  low / medium / high request diversity
  Task 2  batched requests for baseline vs context_grouped scheduler
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

ExperimentKind = Literal[
    "baseline",       # Figure 4: unordered stream, baseline scheduler
    "repeat",         # exact repeat immediately after first
    "repeat_gap",     # exact repeat after N other requests
    "diversity",      # low | medium | high locality
    "length",         # all papers; token counts for length plot
    "task2",          # batched diversity workload for Task 2
]

DiversityLevel = Literal["low", "medium", "high"]
ContextSet = Literal["all", "short", "long"]
VisitType = Literal["first", "revisit", "exact_repeat", "gap"]

SYSTEM_PROMPT = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document"
)

DEFAULT_QUESTIONS = [
    "What is the main topic of this paper?",
    "What problem does this paper try to solve?",
    "What method or system does the paper propose?",
    "What are the key experimental results?",
    "Write a short summary of this document in 3-5 sentences.",
]

CONTEXT_QUESTIONS: dict[str, list[str]] = {
    "vllm": [
        "What is PagedAttention and why does vLLM use it?",
        "How does vLLM improve throughput compared to prior systems?",
    ],
    "cacheblend": [
        "How does CacheBlend reuse KV caches in RAG?",
        "What speedup does CacheBlend report over full KV recomputation?",
    ],
}

# [Task1] Core data structure: one inference request with context + question + metadata
@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    context_id: str
    question: str # prompt question
    context_text: str # 14 papers
    sequence_index: int 
    experiment: str
    visit_type: VisitType = "first"
    prompt_token_count: int | None = None
    source_request_id: str | None = None  # for exact_repeat: id of first copy

    # [Task1] 3-turn chat format: system_prompt+paper → assistant ack → question
    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{self.context_text}"},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": self.question},
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("context_text")
        return d


@dataclass
class RequestSchedule:
    requests: list[InferenceRequest] = field(default_factory=list)
    experiment: str = "baseline"
    diversity_level: str | None = None
    seed: int | None = None
    data_dir: str = ""
    num_contexts: int = 0
    requests_per_context: int = 0
    notes: str = ""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": self.experiment,
            "diversity_level": self.diversity_level,
            "seed": self.seed,
            "data_dir": self.data_dir,
            "num_contexts": self.num_contexts,
            "requests_per_context": self.requests_per_context,
            "notes": self.notes,
            "num_requests": len(self.requests),
            "requests": [r.to_dict() for r in self.requests],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __iter__(self) -> Iterator[InferenceRequest]:
        return iter(self.requests)


# [Task2] Batched workload: list of batches, each batch is a list of InferenceRequest
@dataclass
class BatchSchedule:
    """Task 2: one or more batches submitted to /v2/batch/chat/completions."""

    batches: list[list[InferenceRequest]]
    experiment: str = "task2"
    diversity_level: str | None = None
    context_set: str = "all"
    batch_size: int = 28
    seed: int | None = None
    data_dir: str = ""
    notes: str = ""

    @property
    def num_requests(self) -> int:
        return sum(len(b) for b in self.batches)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": self.experiment,
            "diversity_level": self.diversity_level,
            "context_set": self.context_set,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "data_dir": self.data_dir,
            "notes": self.notes,
            "num_batches": len(self.batches),
            "num_requests": self.num_requests,
            "batches": [
                [r.to_dict() for r in batch] for batch in self.batches
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_contexts(
    data_dir: str | Path,
    *,
    exclude: Sequence[str] = ("sample",),
    max_contexts: int | None = None,
) -> dict[str, str]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    contexts: dict[str, str] = {}
    for path in sorted(data_dir.glob("*.txt")):
        if path.stem in exclude:
            continue
        contexts[path.stem] = path.read_text(encoding="utf-8")
    if not contexts:
        raise ValueError(f"No .txt contexts in {data_dir}")
    if max_contexts is not None:
        keys = sorted(contexts.keys())[:max_contexts]
        contexts = {k: contexts[k] for k in keys}
    return contexts


def _questions_for_context(context_id: str, n: int) -> list[str]:
    pool: list[str] = []
    pool.extend(CONTEXT_QUESTIONS.get(context_id, []))
    pool.extend(DEFAULT_QUESTIONS)
    seen: set[str] = set()
    unique: list[str] = []
    for q in pool:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    if n <= len(unique):
        return unique[:n]
    return [unique[i % len(unique)] for i in range(n)]


def _make_request(
    idx: int,
    context_id: str,
    text: str,
    question: str,
    experiment: str,
    visit_type: VisitType = "first",
    source_request_id: str | None = None,
) -> InferenceRequest:
    rid = f"req-{idx:04d}"
    return InferenceRequest(
        request_id=rid,
        context_id=context_id,
        question=question,
        context_text=text,
        sequence_index=idx,
        experiment=experiment,
        visit_type=visit_type,
        source_request_id=source_request_id,
    )


def _reindex(requests: list[InferenceRequest]) -> list[InferenceRequest]:
    out: list[InferenceRequest] = []
    for i, r in enumerate(requests):
        out.append(
            InferenceRequest(
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
        )
    return out


def _base_requests(
    contexts: dict[str, str],
    *,
    requests_per_context: int,
    experiment: str,
) -> list[InferenceRequest]:
    reqs: list[InferenceRequest] = []
    idx = 0
    for cid in sorted(contexts.keys()):
        for q in _questions_for_context(cid, requests_per_context):
            reqs.append(
                _make_request(idx, cid, contexts[cid], q, experiment, "first")
            )
            idx += 1
    return reqs


def build_baseline_schedule(
    contexts: dict[str, str],
    *,
    requests_per_context: int = 2,
    seed: int = 42,
    data_dir: str = "",
) -> RequestSchedule:
    """Figure 4: unordered stream over multiple contexts (baseline scheduler)."""
    rng = random.Random(seed)
    reqs = _base_requests(
        contexts, requests_per_context=requests_per_context, experiment="baseline"
    )
    rng.shuffle(reqs)
    reqs = _reindex(reqs)
    return RequestSchedule(
        requests=reqs,
        experiment="baseline",
        seed=seed,
        data_dir=data_dir,
        num_contexts=len(contexts),
        requests_per_context=requests_per_context,
        notes="Unordered requests; maps to Figure 4 request stream.",
    )


def build_repeat_schedule(
    contexts: dict[str, str],
    *,
    requests_per_context: int = 2,
    seed: int = 42,
    data_dir: str = "",
) -> RequestSchedule:
    """
    Q2: shuffle (context, question) pairs, then for each pair send the exact
    same request twice back-to-back (first → immediate exact_repeat).
    """
    rng = random.Random(seed)

    # [Task1 Q2] Shuffle all (context, question) pairs
    base = _base_requests(
        contexts, requests_per_context=requests_per_context, experiment="repeat"
    )
    rng.shuffle(base)

    all_reqs: list[InferenceRequest] = []
    idx = 0

    # [Task1 Q2] Each pair: first (cache miss) → immediate exact_repeat (cache hit)
    for src in base:
        first_id = f"req-{idx:04d}"
        all_reqs.append(
            InferenceRequest(
                request_id=first_id,
                context_id=src.context_id,
                question=src.question,
                context_text=src.context_text,
                sequence_index=idx,
                experiment="repeat",
                visit_type="first",  # [Task1 Q2] cache miss
            )
        )
        all_reqs.append(
            InferenceRequest(
                request_id=f"req-{idx + 1:04d}-dup",
                context_id=src.context_id,
                question=src.question,
                context_text=src.context_text,
                sequence_index=idx + 1,
                experiment="repeat",
                visit_type="exact_repeat",  # [Task1 Q2] should be cache hit
                source_request_id=first_id,
            )
        )
        idx += 2

    all_reqs = _reindex(all_reqs)
    n_pairs = len(base)
    return RequestSchedule(
        requests=all_reqs,
        experiment="repeat",
        seed=seed,
        data_dir=data_dir,
        num_contexts=len(contexts),
        requests_per_context=requests_per_context,
        notes=(
            f"{n_pairs} (context, question) pairs; each asked twice consecutively "
            "(first, then immediate exact_repeat)."
        ),
    )


def build_repeat_two_phase_schedule(
    contexts: dict[str, str],
    *,
    requests_per_context: int = 1,
    seed: int = 42,
    data_dir: str = "",
) -> RequestSchedule:
    """
    14 papers × 1 question: Phase A all first pass (shuffled), then Phase B all exact repeats.
    Total = 2 × num_contexts × requests_per_context (default 28).
    """
    rng = random.Random(seed)
    base = _base_requests(
        contexts,
        requests_per_context=requests_per_context,
        experiment="repeat_gap",
    )
    rng.shuffle(base)

    # [Task1 Q2] Phase A: all first-pass requests (shuffled order)
    phase_a: list[InferenceRequest] = []
    for i, src in enumerate(base):
        phase_a.append(
            _make_request(i, src.context_id, src.context_text, src.question, "repeat_gap", "first")
        )

    # [Task1 Q2] Phase B: exact repeats in same order; gap = 13 other papers in between
    phase_b: list[InferenceRequest] = []
    for i, src in enumerate(phase_a):
        phase_b.append(
            InferenceRequest(
                request_id=f"req-{len(phase_a) + i:04d}-dup",
                context_id=src.context_id,
                question=src.question,
                context_text=src.context_text,
                sequence_index=len(phase_a) + i,
                experiment="repeat_gap",
                visit_type="exact_repeat",  # [Task1 Q2] cache hit depends on cache capacity
                source_request_id=src.request_id,
            )
        )

    all_reqs = _reindex(phase_a + phase_b)
    n = len(phase_a)
    return RequestSchedule(
        requests=all_reqs,
        experiment="repeat_gap",
        seed=seed,
        data_dir=data_dir,
        num_contexts=len(contexts),
        requests_per_context=requests_per_context,
        notes=(
            f"Phase A: {n} first pass (shuffled). Phase B: {n} exact repeat (same order). "
            f"Total {len(all_reqs)} requests."
        ),
    )


def build_repeat_gap_schedule(
    contexts: dict[str, str],
    *,
    requests_per_context: int = 1,
    repeat_gap: int = 4,
    seed: int = 42,
    data_dir: str = "",
) -> RequestSchedule:
    """
    Q2 variant: for each (context, question), send first → `repeat_gap` other
    requests → exact_repeat of the same pair.
    """
    if repeat_gap < 1:
        raise ValueError("repeat_gap must be >= 1")

    rng = random.Random(seed)
    base = _base_requests(
        contexts, requests_per_context=requests_per_context, experiment="repeat_gap"
    )
    rng.shuffle(base)
    n = len(base)

    all_reqs: list[InferenceRequest] = []
    idx = 0
    for i, src in enumerate(base):
        first_id = f"req-{idx:04d}"
        all_reqs.append(
            InferenceRequest(
                request_id=first_id,
                context_id=src.context_id,
                question=src.question,
                context_text=src.context_text,
                sequence_index=idx,
                experiment="repeat_gap",
                visit_type="first",
            )
        )
        idx += 1

        added = 0
        scan = 1
        while added < repeat_gap and scan < n * 2:
            filler = base[(i + scan) % n]
            scan += 1
            if filler.context_id == src.context_id and filler.question == src.question:
                continue
            all_reqs.append(
                InferenceRequest(
                    request_id=f"req-{idx:04d}-gap{added}",
                    context_id=filler.context_id,
                    question=filler.question,
                    context_text=filler.context_text,
                    sequence_index=idx,
                    experiment="repeat_gap",
                    visit_type="gap",
                    source_request_id=first_id,
                )
            )
            idx += 1
            added += 1

        all_reqs.append(
            InferenceRequest(
                request_id=f"req-{idx:04d}-dup",
                context_id=src.context_id,
                question=src.question,
                context_text=src.context_text,
                sequence_index=idx,
                experiment="repeat_gap",
                visit_type="exact_repeat",
                source_request_id=first_id,
            )
        )
        idx += 1

    all_reqs = _reindex(all_reqs)
    n_pairs = len(base)
    total = len(all_reqs)
    return RequestSchedule(
        requests=all_reqs,
        experiment="repeat_gap",
        seed=seed,
        data_dir=data_dir,
        num_contexts=len(contexts),
        requests_per_context=requests_per_context,
        notes=(
            f"{n_pairs} pairs × (1 first + {repeat_gap} gap + 1 repeat) = {total} requests; "
            f"repeat is {repeat_gap} requests after first."
        ),
    )


def _adjacent_same_context_count(reqs: list[InferenceRequest]) -> int:
    """How often the same context appears on consecutive requests (lower = more diverse)."""
    if len(reqs) < 2:
        return 0
    return sum(
        1 for i in range(1, len(reqs)) if reqs[i].context_id == reqs[i - 1].context_id
    )


def build_diversity_schedule(
    contexts: dict[str, str],
    level: DiversityLevel,
    *,
    requests_per_context: int = 2,
    seed: int = 42,
    data_dir: str = "",
) -> RequestSchedule:
    """
    Q3 diversity (fair compare: same #requests for all levels).

      low    — grouped by context_id (q1,q2 for doc A, then doc B, …)
      medium — all requests shuffled (Figure 4)
      high   — round-robin across contexts (doc1-q1, doc2-q1, …, doc1-q2, …)
    """
    rng = random.Random(seed)
    n_ctx = len(contexts)
    n_req = n_ctx * requests_per_context

    if level == "low":
        # [Task1 Q3] Grouped by context (A-q1,A-q2,B-q1,B-q2): max cache reuse
        reqs = _base_requests(
            contexts,
            requests_per_context=requests_per_context,
            experiment="diversity",
        )
        notes = (
            f"Low diversity: {n_req} requests grouped by context "
            f"({requests_per_context} questions per paper)."
        )
    elif level == "medium":
        # [Task1 Q3] Fully shuffled: random cache reuse
        reqs = _base_requests(
            contexts,
            requests_per_context=requests_per_context,
            experiment="diversity",
        )
        rng.shuffle(reqs)
        notes = (
            f"Medium diversity: {n_req} requests fully shuffled (Figure 4 stream)."
        )
    else:  # high — maximum interleaving, same 28 requests as low/medium
        # [Task1 Q3] Round-robin (A-q1,B-q1,C-q1...): every request switches context, min cache hit
        reqs = []
        idx = 0
        cids = sorted(contexts.keys())
        for q_i in range(requests_per_context):
            for cid in cids:
                q = _questions_for_context(cid, requests_per_context)[q_i]
                reqs.append(
                    _make_request(idx, cid, contexts[cid], q, "diversity", "first")
                )
                idx += 1
        notes = (
            f"High diversity: {n_req} requests round-robin across {n_ctx} contexts "
            f"(max spacing between same-document requests)."
        )

    reqs = _reindex(reqs)
    adj = _adjacent_same_context_count(reqs)
    notes += f" Adjacent same-context pairs: {adj}/{n_req - 1}."
    return RequestSchedule(
        requests=reqs,
        experiment="diversity",
        diversity_level=level,
        seed=seed,
        data_dir=data_dir,
        num_contexts=n_ctx,
        requests_per_context=requests_per_context,
        notes=notes,
    )


def _filter_requests_by_context_set(
    requests: list[InferenceRequest],
    contexts: dict[str, str],
    context_set: ContextSet,
    *,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> list[InferenceRequest]:
    if context_set == "all":
        return requests

    annotated = [
        r
        if r.prompt_token_count is not None
        else annotate_prompt_tokens(
            RequestSchedule(requests=[r]), model_name=model_name
        ).requests[0]
        for r in requests
    ]
    by_ctx: dict[str, list[InferenceRequest]] = {}
    for r in annotated:
        by_ctx.setdefault(r.context_id, []).append(r)
    avg_len = {
        cid: sum(x.prompt_token_count or 0 for x in rs) / len(rs)
        for cid, rs in by_ctx.items()
    }
    n_pick = max(1, len(contexts) // 2)
    ranked = sorted(avg_len.keys(), key=lambda c: avg_len[c])
    pick = set(ranked[:n_pick] if context_set == "short" else ranked[-n_pick:])
    filtered = [r for r in requests if r.context_id in pick]
    return _reindex(filtered)


# [Task2] Build batched workload: diversity schedule → filter by context_set → split into batches
def build_task2_schedule(
    contexts: dict[str, str],
    *,
    diversity_level: DiversityLevel = "medium",
    requests_per_context: int = 2,
    batch_size: int = 28,
    context_set: ContextSet = "all",
    seed: int = 42,
    data_dir: str = "",
    count_tokens: bool = True,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> BatchSchedule:
    """
    Task 2 workload: Q3-style diversity on flat requests, split into batches.

    Scheduler (baseline vs context_grouped) is applied server-side per batch.
    """
    # [Task2] Step 1: reuse Q3's diversity schedule as base workload
    sched = build_diversity_schedule(
        contexts,
        diversity_level,
        requests_per_context=requests_per_context,
        seed=seed,
        data_dir=data_dir,
    )
    if count_tokens:
        sched = annotate_prompt_tokens(sched, model_name=model_name)

    # [Task2] Step 2: filter by context length (short=bottom half, long=top half, all=no filter)
    reqs = _filter_requests_by_context_set(
        sched.requests, contexts, context_set, model_name=model_name
    )
    if not reqs:
        raise ValueError("No requests after context_set filter")

    # [Task2] Step 3: split into batches of batch_size
    batch_size = max(1, min(batch_size, len(reqs)))
    batches: list[list[InferenceRequest]] = []
    for i in range(0, len(reqs), batch_size):
        batches.append(_reindex(reqs[i : i + batch_size]))

    notes = (
        f"Task2: {len(reqs)} requests, diversity={diversity_level}, "
        f"context_set={context_set}, batch_size={batch_size}, "
        f"{len(batches)} batch(es). {sched.notes}"
    )
    return BatchSchedule(
        batches=batches,
        experiment="task2",
        diversity_level=diversity_level,
        context_set=context_set,
        batch_size=batch_size,
        seed=seed,
        data_dir=data_dir,
        notes=notes,
    )


def build_length_schedule(
    contexts: dict[str, str],
    *,
    requests_per_context: int = 1,
    seed: int = 42,
    data_dir: str = "",
) -> RequestSchedule:
    """Q1: one question per paper; natural length variation across summaries."""
    rng = random.Random(seed)
    reqs = _base_requests(
        contexts, requests_per_context=requests_per_context, experiment="length"
    )
    rng.shuffle(reqs)
    reqs = _reindex(reqs)
    return RequestSchedule(
        requests=reqs,
        experiment="length",
        seed=seed,
        data_dir=data_dir,
        num_contexts=len(contexts),
        requests_per_context=requests_per_context,
        notes="One question per context; use prompt_token_count for length plot.",
    )


def annotate_prompt_tokens(
    schedule: RequestSchedule,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> RequestSchedule:
    """Pre-compute prompt tokens (system + document + question)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        from transformers.models.auto.tokenization_auto import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    updated: list[InferenceRequest] = []
    for r in schedule.requests:
        text = f"{SYSTEM_PROMPT}\n\n{r.context_text}\n\n{r.question}"
        n = len(tok.encode(text))
        updated.append(
            InferenceRequest(
                request_id=r.request_id,
                context_id=r.context_id,
                question=r.question,
                context_text=r.context_text,
                sequence_index=r.sequence_index,
                experiment=r.experiment,
                visit_type=r.visit_type,
                prompt_token_count=n,
                source_request_id=r.source_request_id,
            )
        )
    return RequestSchedule(
        requests=updated,
        experiment=schedule.experiment,
        diversity_level=schedule.diversity_level,
        seed=schedule.seed,
        data_dir=schedule.data_dir,
        num_contexts=schedule.num_contexts,
        requests_per_context=schedule.requests_per_context,
        notes=schedule.notes,
    )


def build_schedule(
    experiment: str,
    data_dir: str | Path,
    *,
    diversity_level: DiversityLevel = "medium",
    requests_per_context: int = 2,
    repeat_gap: int = 4,
    num_contexts: int | None = None,
    seed: int = 42,
    exclude: Sequence[str] = ("sample",),
    count_tokens: bool = True,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> RequestSchedule:
    contexts = load_contexts(data_dir, exclude=exclude, max_contexts=num_contexts)
    data_dir_s = str(data_dir)

    if experiment == "baseline":
        sched = build_baseline_schedule(
            contexts,
            requests_per_context=requests_per_context,
            seed=seed,
            data_dir=data_dir_s,
        )
    elif experiment == "repeat":
        sched = build_repeat_schedule(
            contexts,
            requests_per_context=requests_per_context,
            seed=seed,
            data_dir=data_dir_s,
        )
    elif experiment == "repeat_two_phase":
        sched = build_repeat_two_phase_schedule(
            contexts,
            requests_per_context=requests_per_context,
            seed=seed,
            data_dir=data_dir_s,
        )
    elif experiment == "repeat_gap":
        sched = build_repeat_gap_schedule(
            contexts,
            requests_per_context=requests_per_context,
            repeat_gap=repeat_gap,
            seed=seed,
            data_dir=data_dir_s,
        )
    elif experiment == "diversity":
        sched = build_diversity_schedule(
            contexts,
            diversity_level,
            requests_per_context=requests_per_context,
            seed=seed,
            data_dir=data_dir_s,
        )
    elif experiment == "length":
        # Q1 always uses one question per paper (ignore global default=2).
        sched = build_length_schedule(
            contexts,
            requests_per_context=1,
            seed=seed,
            data_dir=data_dir_s,
        )
    elif experiment == "task2":
        raise ValueError(
            "Use build_task2_schedule() for Task 2 (returns BatchSchedule with batches)."
        )
    else:
        raise ValueError(
            f"Unknown experiment {experiment!r}; "
            f"use baseline|repeat|repeat_two_phase|repeat_gap|diversity|length|task2"
        )

    if count_tokens:
        sched = annotate_prompt_tokens(sched, model_name=model_name)
    return sched
