"""
Extended vLLM API (/v2): Task 2 batch endpoint with context-grouped scheduler.
"""

from __future__ import annotations

import os
import time
from typing import Literal

import vllm.entrypoints.openai.api_server as base_api
from fastapi import APIRouter, Request
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

from lmcache_vllm.scheduler import SchedulerMode, adjacent_same_context_pairs, schedule_batch

# [Task1/Task2] /v2 router — mounted with prefix="/v2" by vllm_injection.py:529
extended_router = APIRouter()

# Loopback for per-request inference inside a batch (standard OpenAI API).
# Batch entry itself is still POST /v2/batch/chat/completions.
# If /v1/chat/completions fails but /v2/chat/completions works, set:
#   export VLLM_INTERNAL_BASE=http://127.0.0.1:8000/v2
_INTERNAL_BASE = os.environ.get("VLLM_INTERNAL_BASE", "http://127.0.0.1:8000/v1")
# Per-request timeout inside a batch (seconds). No SDK retries — fail fast if vLLM is down.
_BATCH_REQUEST_TIMEOUT = float(os.environ.get("VLLM_BATCH_REQUEST_TIMEOUT", "300"))


# [Task1/Task2] Loopback OpenAI client: batch endpoint calls /v1 internally for each request
def _internal_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="EMPTY",
        base_url=_INTERNAL_BASE,
        max_retries=0,
        timeout=_BATCH_REQUEST_TIMEOUT,
    )


class BatchChatItem(BaseModel):
    request_id: str
    context_id: str
    messages: list[dict[str, str]]
    max_tokens: int = 64
    temperature: float = 0.0


class BatchChatRequest(BaseModel):
    scheduler: SchedulerMode = "context_grouped"
    requests: list[BatchChatItem] = Field(min_length=1)
    stop: list[str] | None = Field(default_factory=lambda: ["\n"])


class BatchChatResult(BaseModel):
    request_id: str
    context_id: str
    sequence_index: int
    execution_index: int
    ttft_sec: float
    response_time_sec: float
    completion_tokens: int
    success: bool
    error: str | None = None
    completion_preview: str | None = None


class BatchChatResponse(BaseModel):
    scheduler: SchedulerMode
    model: str
    execution_order: list[str]
    adjacent_same_context_pairs: int
    total_wall_time_sec: float
    throughput_req_per_sec: float
    results: list[BatchChatResult]


async def _resolve_model(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    return models.data[0].id


# [Task1/Task2] Execute one request inside a batch, stream response, record TTFT + response_time
async def _run_one(
    client: AsyncOpenAI,
    model: str,
    item: BatchChatItem,
    *,
    stop: list[str] | None,
) -> BatchChatResult:
    t0 = time.perf_counter()
    ttft: float | None = None
    n_completion = 0
    chunks: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=item.messages,
            max_tokens=item.max_tokens,
            temperature=item.temperature,
            stream=True,
            stop=stop,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_completion += len(delta)
                chunks.append(delta)
        response_time = time.perf_counter() - t0
        if ttft is None:
            ttft = response_time
        preview = "".join(chunks)[:120] or None
        return BatchChatResult(
            request_id=item.request_id,
            context_id=item.context_id,
            sequence_index=-1,
            execution_index=-1,
            ttft_sec=ttft,
            response_time_sec=response_time,
            completion_tokens=n_completion or 0,
            success=True,
            completion_preview=preview,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return BatchChatResult(
            request_id=item.request_id,
            context_id=item.context_id,
            sequence_index=-1,
            execution_index=-1,
            ttft_sec=elapsed,
            response_time_sec=elapsed,
            completion_tokens=0,
            success=False,
            error=str(exc),
        )


@extended_router.get("/models")
async def show_available_models(request: Request):
    return await base_api.show_available_models(request)

# [Task1] Baseline single-request endpoint: FIFO, no reordering
# This decorator registers at /chat/completions; prefix /v2 is added by vllm_injection.py
@extended_router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
):
    return await base_api.create_chat_completion(request, raw_request)


# [Task2] Batch endpoint: receives N requests, reorders via schedule_batch(), runs sequentially
@extended_router.post("/batch/chat/completions", response_model=BatchChatResponse)
async def create_batch_chat_completion(body: BatchChatRequest) -> BatchChatResponse:
    """Run a batch through baseline or context_grouped scheduler (sequential execution)."""
    client = _internal_client()
    model = await _resolve_model(client)

    indexed = list(enumerate(body.requests))
    items = [item for _, item in indexed]
    ordered = schedule_batch(items, body.scheduler)  # [Task2] baseline=FIFO, context_grouped=reorder
    order_ids = [item.request_id for item in ordered]

    wall_t0 = time.perf_counter()
    results: list[BatchChatResult] = []
    for exec_idx, item in enumerate(ordered):
        res = await _run_one(client, model, item, stop=body.stop)
        res.execution_index = exec_idx
        # Original submission index in the batch payload.
        for sub_idx, orig in indexed:
            if orig.request_id == item.request_id:
                res.sequence_index = sub_idx
                break
        results.append(res)

    wall = time.perf_counter() - wall_t0
    ok = [r for r in results if r.success]
    through = len(ok) / wall if wall > 0 else 0.0
    adj = adjacent_same_context_pairs(ordered)

    return BatchChatResponse(
        scheduler=body.scheduler,
        model=model,
        execution_order=order_ids,
        adjacent_same_context_pairs=adj,
        total_wall_time_sec=wall,
        throughput_req_per_sec=through,
        results=results,
    )
