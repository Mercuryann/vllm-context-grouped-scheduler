# IK2221 Project — Task 2 Demo Cheatsheet

> [!NOTE]
> **前提**: Task 2 使用与 Task 1 相同的 3 终端启动流程（见 Task 1 cheatsheet）。
> 实验命令在 **Terminal 4** 中执行。

## 目录
1. [Task 2 核心概念](#1-task-2-核心概念)
2. [核心代码走读](#2-核心代码走读)
3. [实验命令](#3-实验命令)
4. [已有实验结果速查](#4-已有实验结果速查)
5. [预期 Demo/答辩问题 & 回答](#5-预期-demo答辩问题--回答)

---

## 1. Task 2 核心概念

### 题目要求
> *Implement a context-grouped scheduler that reorders batched requests so that requests sharing the same context run consecutively, maximizing KV cache reuse.*

### Task 1 vs Task 2 对比

| | Task 1 | Task 2 |
|:---|:---|:---|
| **端点** | `/v2/chat/completions` | `/v2/batch/chat/completions` |
| **请求方式** | 逐个发送 | 一批 N 个请求一次性提交 |
| **调度策略** | FIFO (无重排序) | `baseline` (FIFO) vs `context_grouped` (重排序) |
| **核心目标** | 观察 cache 本身的性能影响 | 证明重排序能提升 cache 命中率 |

### 工作原理

```
前端提交一批请求 (打乱顺序):
  A-q1, C-q1, B-q1, A-q2, B-q2, C-q2

baseline scheduler (FIFO):
  A-q1 → C-q1 → B-q1 → A-q2 → B-q2 → C-q2
  (每次切换 context → cache miss)

context_grouped scheduler (重排序):
  A-q1 → A-q2 → C-q1 → C-q2 → B-q1 → B-q2
  (同 context 连续 → cache hit)
```

---

## 2. 核心代码走读

### 2.1 Scheduler — `~/shared/ik2221_project2/lmcache-vllm-extended/lmcache_vllm/scheduler.py`

这是 Task 2 最核心的实现文件（只有 ~47 行）：

```python
# [Task2] 核心调度器：对一批请求进行重排序
def schedule_batch(
    items: list[T],
    mode: SchedulerMode,
) -> list[T]:
    if mode == "baseline" or not items:
        return list(items)  # [Task1] FIFO: 不进行重排序

    # [Task2] context_grouped: 按 context_id 分组，并保留首次出现的顺序
    groups: OrderedDict[str, list[T]] = OrderedDict()
    for item in items:
        cid = item.context_id
        groups.setdefault(cid, []).append(item)
    ordered: list[T] = []
    for group in groups.values():
        ordered.extend(group)
    return ordered
```

**核心逻辑解读：**
- 使用 `OrderedDict`，key 是 `context_id`，value 是该 context 下的所有请求
- `setdefault` 保证了第一次出现的 context 排在前面
- 同一个 context 内部的请求保持原始相对顺序
- 最后 `extend` 展平：**同 context 请求被排在一起**

### 2.2 Batch 端点 — `~/shared/ik2221_project2/lmcache-vllm-extended/lmcache_vllm/custom_api.py`

```python
# [Task2] Batch 端点：接收 N 个请求，通过 schedule_batch() 重排序后，逐一按顺序执行
@extended_router.post("/batch/chat/completions", response_model=BatchChatResponse)
async def create_batch_chat_completion(body: BatchChatRequest) -> BatchChatResponse:
    ...
    items = [item for _, item in indexed]
    ordered = schedule_batch(items, body.scheduler)  # [Task2] baseline=FIFO, context_grouped=重排序
    ...
    # 按照 ordered 顺序逐一执行
    for exec_idx, item in enumerate(ordered):
        res = await _run_one(client, model, item, stop=body.stop)
        ...
```

**与 Task 1 的关键区别：**
- 接收 `BatchChatRequest`（包含 `scheduler` 字段 + `requests` 列表）
- 调用 `schedule_batch()` 重排序后再逐一执行
- 返回 `adjacent_same_context_pairs` 指标（衡量连续同 context 对数）

### 2.3 实验脚本 — `~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/run_task2.py`

**核心函数：**

```python
def run_batch(api_base, scheduler, batch, ...) -> dict:
    # 将一批请求通过 HTTP POST 发送到 /v2/batch/chat/completions
    url = f"{api_base.rstrip('/')}/batch/chat/completions"
    payload = _batch_payload(batch, scheduler, ...)
    ...
    resp = client.post(url, json=payload)
    return resp.json()

def run_experiment(api_base, batch_sched, schedulers, ...) -> dict:
    # 对每种 scheduler mode (baseline / context_grouped)，逐 batch 执行
    for mode in schedulers:
        for bi, batch in enumerate(batch_sched.batches):
            data = run_batch(api_base, mode, batch, ...)
            ...
```

### 2.4 Schedule 构建 — `~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/request_generator.py`

```python
def build_task2_schedule(
    contexts: dict[str, str],
    *,
    diversity_level: DiversityLevel = "medium",
    ...
    batch_size: int = 28,
    context_set: ContextSet = "all",
    ...
) -> BatchSchedule:
    # [Task2] 步骤 1: 使用 Q3 的 diversity_schedule 生成打乱后的请求列表
    sched = build_diversity_schedule(contexts, diversity_level, ...)

    # [Task2] 步骤 2: 按 context_set 过滤 (all / short / long)
    reqs = _filter_requests_by_context_set(sched.requests, ...)

    # [Task2] 步骤 3: 按 batch_size 切分成多个 batch
    batches: list[list[InferenceRequest]] = []
    for i in range(0, len(reqs), batch_size):
        batches.append(_reindex(reqs[i : i + batch_size]))
    ...
```

---

## 3. 实验命令

> [!IMPORTANT]
> 所有命令在 Terminal 4（实验终端）中执行，且需要 Terminal 1 (LMCache Server) 和 Terminal 2 (vLLM) 保持运行。

### 3.1 单次实验（baseline vs context_grouped）

```bash
cd ~/shared/ik2221_project2
source ./venv/bin/activate

# 默认: medium diversity, batch_size=28, 同时跑 baseline 和 context_grouped
python lmcache-vllm-extended/benchmark/run_task2.py \
  --scheduler both --diversity medium --batch-size 28 --cache-gb 0.2
```

### 3.2 Suite 实验（预设的多组实验）

```bash
# Diversity sweep: low / medium / high
python lmcache-vllm-extended/benchmark/run_task2.py --suite diversity --cache-gb 0.2

# Batch size sweep: 4 / 7 / 14 / 28
python lmcache-vllm-extended/benchmark/run_task2.py --suite batch --cache-gb 0.2

# Context set sweep: short / all / long
python lmcache-vllm-extended/benchmark/run_task2.py --suite context --cache-gb 0.2
```

### 3.3 仅重新生成图表（不重跑实验）

```bash
# 单个实验的图
python lmcache-vllm-extended/benchmark/run_task2.py \
  --plot-only --stem task2_div-medium_bs28_all_cache0.2

# Batch sweep 合并图
python lmcache-vllm-extended/benchmark/run_task2.py \
  --plot-suite batch --cache-gb 0.2
```

---

## 4. 已有实验结果速查

### 结果文件路径
```
~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/results/
├── task2_div-low_cache0.2.json / .png       ← diversity low
├── task2_div-medium_cache0.2.json / .png    ← diversity medium
├── task2_div-high_cache0.2.json / .png      ← diversity high
├── task2_bs4_cache0.2.json / .png           ← batch_size=4
├── task2_bs7_cache0.2.json / .png           ← batch_size=7
├── task2_bs14_cache0.2.json / .png          ← batch_size=14
├── task2_bs28_cache0.2.json / .png          ← batch_size=28
├── task2_ctx-short_cache0.2.json / .png     ← context_set=short
├── task2_ctx-all_cache0.2.json / .png       ← context_set=all
└── task2_ctx-long_cache0.2.json / .png      ← context_set=long
```

### Diversity sweep (cache=0.2GB, bs=28)
```
Diversity  Scheduler           req/s    RT(s)    TTFT(s)
low        baseline            0.511    1.956    0.124
low        context_grouped     0.503    1.986    0.121
medium     baseline            0.463    2.159    0.267    ← 最能看出差异
medium     context_grouped     0.505    1.981    0.119    ← context_grouped 优势明显
high       baseline            0.505    1.980    0.127
high       context_grouped     0.503    1.989    0.119
```

### Batch size sweep (diversity=medium, cache=0.2GB)
```
bs   baseline req/s   grouped req/s
4    0.514              0.503
7    0.501              0.505
14   0.501              0.504
28   0.499              0.508      ← batch 越大，grouped 优势越明显
```

### 关键观察
- **medium diversity + 大 batch** 是 context_grouped 效果最好的场景
- low diversity 本身已经有较高的 cache 命中率，重排序收益有限
- batch_size 太小（如 4），每个 batch 内 context 种类少，重排序空间有限

---

## 5. 预期 Demo/答辩问题 & 回答

### Q: context_grouped scheduler 做了什么？
> **A:** 它接收一批请求，按 `context_id` 分组（使用 `OrderedDict` 保持首次出现的顺序），然后展平输出。效果是：同一个 context 的请求被排到一起，这样前一个请求计算的 KV cache 可以被后一个直接复用。

### Q: 为什么用 OrderedDict 而不是普通 dict？
> **A:** `OrderedDict` 保证分组的顺序等于各 context 在原始请求流中的**首次出现顺序**。这样即使输入是打乱的，输出仍有确定性。Python 3.7+ 普通 dict 也保序，但 `OrderedDict` 语义上更明确。

### Q: 为什么 medium diversity 下效果最好？
> **A:** medium diversity 是完全打乱的请求流。baseline 按原顺序执行会频繁切换 context（每次都 cache miss），而 context_grouped 重排序后大幅减少切换。low diversity 本身已经分组好了（adjacent pairs 高），重排序改善空间小。high diversity 的 round-robin 模式虽然每次都切换，但 context_grouped 重排序后效果和 medium 类似。

### Q: batch_size 对性能有什么影响？
> **A:** batch 越大，每个 batch 内包含的同一 context 请求越多，context_grouped 的重排空间越大。batch=4 时每个 batch 可能只有 1-2 种 context，分组收益极小。batch=28 时所有 14 篇 paper 的 2 个问题都在一个 batch 里，重排序效果最大化。

### Q: 为什么 TTFT 的差异比 full response time 更明显？
> **A:** TTFT 主要由 prefill 决定，cache hit 可以跳过 prefill。而 full response time 还包含 decode（~64 tokens），decode 时间与 cache 无关。所以 TTFT 是更敏感的指标。在我们的数据中，medium diversity 下 baseline TTFT=0.267s vs grouped TTFT=0.119s，差异 2.2x。

### Q: adjacent_same_context_pairs 是什么意思？
> **A:** 衡量执行序列中有多少对"相邻请求使用了同一个 context"。在 `scheduler.py` 的 `adjacent_same_context_pairs()` 函数中计算。这个值越高 → cache 命中率越高。context_grouped 的目标就是最大化这个值。
