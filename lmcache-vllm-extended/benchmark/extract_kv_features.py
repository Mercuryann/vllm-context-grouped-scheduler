#!/usr/bin/env python3
"""
Task 4 从 LMCache 中离线提取 KV Cache 特征向量。

离线提取法：
  1. 向 vLLM 发送推理请求，使 KV Cache 写入 LMCache 存储（本地 + 远端服务器）。
  2. 创建独立的 LMCacheEngine 实例，连接同一个远端 Server，通过 retrieve() 拉回 KV 张量。
  3. 对多层、多头的 KV Tensor 做 Mean Pooling，压缩为每个请求一个固定长度的特征向量。
  4. 将特征向量和对应的 context_id 保存到 .npz 文件，供后续 PCA + KMeans 聚类使用。

用法:
  前提：终端1运行 LMCache Server，终端2运行 vLLM（带 LMCACHE_CONFIG_FILE）。

  # 完整流程：先发请求填充缓存，再提取特征
  python extract_kv_features.py --send-requests

  # 仅提取（假设之前的实验已经填充了缓存）
  python extract_kv_features.py

  # 每个 context 发送多个不同问题的请求
  python extract_kv_features.py --send-requests --questions-per-context 3

输出:
  results/kv_features.npz  — 包含 features (N, D) 和 context_ids (N,)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import logging

logging.getLogger("lmcache").setLevel(logging.INFO)

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BENCHMARK_DIR.parent
_RESULTS_DIR = _BENCHMARK_DIR / "results"

# 将 benchmark/ 和 LMCache/ 加入 import 路径
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
_LMCACHE_ROOT = _PROJECT_DIR.parent / "LMCache"
if str(_LMCACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LMCACHE_ROOT))

from request_generator import (
    DEFAULT_QUESTIONS,
    SYSTEM_PROMPT,
    load_contexts,
)

# ───────────────────────────────────────────────────────────────
# Qwen/Qwen2.5-1.5B-Instruct 的模型参数（必须与 vLLM 启动时一致）
# ───────────────────────────────────────────────────────────────
# [Task4] Model params for Qwen2.5-1.5B-Instruct (must match vLLM startup)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_LAYERS = 28        # transformer 层数
NUM_KV_HEADS = 2       # GQA 的 KV Head 数
HEAD_SIZE = 128        # 每个 Head 的维度
CHUNK_SIZE = 256       # chunk_size
KV_DTYPE = torch.float16


# ───────────────────────────────────────────────────────────────
# 阶段1: 向 vLLM 发送请求，填充 KV Cache
# ───────────────────────────────────────────────────────────────

# [Task4] Build 3-turn chat messages identical to request_generator.to_messages()
def build_messages(context_text: str, question: str) -> list[dict[str, str]]:
    """构造与 request_generator 中 to_messages() 完全一致的对话结构。"""
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{context_text}"},
        {"role": "assistant", "content": "Got it!"},
        {"role": "user", "content": question},
    ]


def _post_json(url: str, payload: dict) -> dict:
    """用标准库 urllib 发送 JSON POST 请求，无需额外依赖。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_model_name(api_base: str) -> str:
    """查询 vLLM /v1/models 端点获取模型名称。"""
    # api_base 可能是 http://...:8000/v2，取根路径再拼 /v1/models
    base = api_base.rsplit("/", 1)[0] if "/v" in api_base else api_base
    url = f"{base}/v1/models"
    resp = _post_json.__wrapped__ if hasattr(_post_json, "__wrapped__") else None
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body["data"][0]["id"]


# [Task4] Stage 1: send inference requests to vLLM so KV Cache gets written to LMCache
def send_requests_to_vllm(
    contexts: dict[str, str],
    questions: list[str],
    api_base: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
) -> None:
    """
    用 urllib 向 vLLM 发送推理请求（无需 openai 包）。
    vLLM 处理请求后，LMCache 会自动将产生的 KV Cache 存入本地 + 远端后端。
    """
    model = _get_model_name(api_base)
    # 构造 chat completions 端点 URL
    chat_url = api_base.rstrip("/") + "/chat/completions"
    total = len(contexts) * len(questions)
    idx = 0

    for cid in sorted(contexts.keys()):
        text = contexts[cid]
        for question in questions:
            idx += 1
            messages = build_messages(text, question)
            print(f"  [{idx}/{total}] context={cid}, q={question[:40]}...")
            try:
                # 非流式请求，等 vLLM 完成推理并将 KV 写入缓存
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                result = _post_json(chat_url, payload)
                preview = result["choices"][0]["message"]["content"][:60]
                print(f"       OK: {preview}")
            except Exception as exc:
                print(f"       FAIL: {exc}")
            # 短暂等待，让 LMCache 的异步写入完成
            time.sleep(0.3)

    # 给远端 server 完成异步写入
    time.sleep(2.0)


# ───────────────────────────────────────────────────────────────
# 阶段2: 创建离线 Engine，从 LMCache 提取 KV 特征
# ───────────────────────────────────────────────────────────────

# [Task4] Create offline LMCacheEngine to retrieve KV tensors without running vLLM
def create_offline_engine(config_path: str):
    """
    创建 LMCacheEngine 实例。
    连接 configuration.yaml 中配置的远端 Server，
    并在初始化时自动 prefetch 远端已有的 KV 数据到本进程的本地缓存中。
    """
    from lmcache.cache_engine import LMCacheEngineBuilder
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

    config = LMCacheEngineConfig.from_file(config_path)
    # [Task4] kv_shape must match vllm_adapter.py's init_lmcache_engine
    kv_shape = (NUM_LAYERS, 2, CHUNK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    metadata = LMCacheEngineMetadata(
        model_name=MODEL_NAME,
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=KV_DTYPE,
        kv_shape=kv_shape,
    )
    engine = LMCacheEngineBuilder.get_or_create(
        "offline_kv_extractor", config, metadata
    )
    return engine


# [Task4] Tokenize prompt using chat_template — must match vLLM's tokenization exactly
def tokenize_prompt(tokenizer, context_text: str, question: str) -> torch.Tensor:
    """
    使用 chat template 将对话转换为 token 序列。
    必须与 vLLM 的分词方式完全一致，否则 prefix hash 不匹配会导致缓存 miss。
    """
    messages = build_messages(context_text, question)
    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    token_ids = tokenizer.encode(prompt_str)
    return torch.tensor(token_ids, dtype=torch.long)


def extract_kv_feature(engine, tokens: torch.Tensor) -> Optional[np.ndarray]:
    """
    从 LMCache 中检索给定 token 序列的 KV Cache，并做 Mean Pooling 得到特征向量。

    retrieve() 返回的 KV Tensor 形状 (vllm 格式):
      [num_layers, 2, num_retrieved_tokens, num_kv_heads, head_size]

    聚合策略:
      1. 对 tokens 维度 (dim=2) 求均值 → [num_layers, 2, num_kv_heads, head_size]
      2. 对 layers 维度 (dim=0) 求均值 → [2, num_kv_heads, head_size]
      3. 展平为一维向量 → [2 * num_kv_heads * head_size]

    对于 Qwen2.5-1.5B (2 heads, 128 dim): 最终特征维度 = 2 * 2 * 128 = 512
    """
    # [Task4] retrieve KV Cache for this token sequence from LMCache
    kv_data, ret_mask = engine.retrieve(tokens, return_tuple=False)

    # 缓存未命中：返回空元组
    if isinstance(kv_data, tuple) and len(kv_data) == 0:
        return None

    num_retrieved = ret_mask.sum().item()
    if num_retrieved == 0:
        return None

    # [Task4] Mean Pooling: tokens dim → layers dim → flatten
    # kv_data: [num_layers, 2, num_tokens, num_kv_heads, head_size]
    kv = kv_data.float()
    pooled = kv.mean(dim=2)   # [Task4] avg over tokens → [layers, 2, heads, dim]
    compact = pooled.mean(dim=0)  # [Task4] avg over layers → [2, heads, dim]
    return compact.cpu().numpy().flatten()  # [Task4] flatten → 512D feature vector


# ───────────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 4 Step 1: 从 LMCache 提取 KV Cache 特征向量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--send-requests", action="store_true",
        help="先向 vLLM 发送请求填充缓存，再提取特征",
    )
    p.add_argument(
        "--questions-per-context", type=int, default=1,
        help="每个 context 发送多少个不同问题 (默认 1)",
    )
    p.add_argument(
        "--config", type=str,
        default=str(_PROJECT_DIR / "configuration.yaml"),
        help="LMCache 配置文件路径 (须与 vLLM 启动时使用的一致)",
    )
    p.add_argument(
        "--data-dir", type=Path,
        default=_PROJECT_DIR / "frontend" / "data",
        help="paper summary .txt 文件所在目录",
    )
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v2")
    p.add_argument(
        "--output", type=Path,
        default=_RESULTS_DIR / "kv_features.npz",
        help="特征向量输出文件",
    )
    p.add_argument("--exclude", nargs="*", default=["sample"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 context 文本（14 篇 paper summaries）
    contexts = load_contexts(args.data_dir, exclude=args.exclude)
    print(f"Loaded {len(contexts)} contexts (from {args.data_dir})")

    # 选择要用的问题集
    questions = DEFAULT_QUESTIONS[: args.questions_per_context]
    print(f"Using {len(questions)} questions per context\n")

    # ── 阶段1: 发送推理请求填充缓存 ──
    if args.send_requests:
        print("=" * 60)
        print("Stage 1: Sending requests to vLLM to populate LMCache")
        print("=" * 60)
        send_requests_to_vllm(contexts, questions, args.api_base)
    else:
        print("Skipped Stage 1 (use --send-requests to populate cache first)")

    # ── 阶段2: 从 LMCache 提取 KV 特征 ──
    print("\n" + "=" * 60)
    print("Stage 2: Extracting KV Cache feature vectors from LMCache")
    print("=" * 60)

    # 加载分词器（必须与 vLLM 使用的同一个模型的 tokenizer）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 创建离线 engine，连接远端 LMCache Server
    print(f"Connecting to LMCache (config: {args.config})...")
    engine = create_offline_engine(args.config)

    features: list[np.ndarray] = []
    context_ids: list[str] = []
    question_labels: list[str] = []
    hit_count = 0
    miss_count = 0

    for cid in sorted(contexts.keys()):
        text = contexts[cid]
        for qi, question in enumerate(questions):
            # 用与 vLLM 完全相同的方式编码 prompt
            tokens = tokenize_prompt(tokenizer, text, question)
            tag = f"{cid}/q{qi}"
            print(f"  {tag}: {len(tokens)} tokens", end="")

            feat = extract_kv_feature(engine, tokens)

            if feat is not None:
                features.append(feat)
                context_ids.append(cid)
                question_labels.append(question[:50])
                hit_count += 1
                print(f"  → HIT, feature dim={feat.shape[0]}")
            else:
                miss_count += 1
                print(f"  → MISS (Token sequence not found in cache)")

    # ── 汇总 & 保存 ──
    print(f"\nHits: {hit_count}, Misses: {miss_count}")

    if not features:
        print("\n⚠  No features extracted! Please check:")
        print("   1. Is LMCache Server running? (python -m lmcache_server.server ...)")
        print("   2. Have requests been sent to vLLM? (use --send-requests)")
        print("   3. Does remote_url in configuration.yaml match the Server?")
        engine.close()
        return

    features_array = np.stack(features)
    context_ids_array = np.array(context_ids)
    question_labels_array = np.array(question_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        features=features_array,
        context_ids=context_ids_array,
        question_labels=question_labels_array,
    )
    print(f"\nSaved {len(features)} feature vectors {features_array.shape} to {args.output}")
    print(f"Feature dimension: {features_array.shape[1]}")
    print(f"Unique contexts: {len(set(context_ids))}")

    engine.close()


if __name__ == "__main__":
    main()
