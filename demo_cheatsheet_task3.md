# IK2221 Project — Task 3 Demo Cheatsheet

> [!NOTE]
> **前提**: Task 3 使用与 Task 1/2 相同的 3 终端启动流程（见 Task 1 cheatsheet）。
> 实验命令在 **Terminal 4** 中执行。

## 目录
1. [Task 3 核心概念](#1-task-3-核心概念)
2. [核心代码走读](#2-核心代码走读)
3. [实验命令](#3-实验命令)
4. [已有实验结果速查](#4-已有实验结果速查)
5. [预期 Demo/答辩问题 & 回答](#5-预期-demo答辩问题--回答)

---

## 1. Task 3 核心概念

### 题目要求
> *Build a RAG (Retrieval-Augmented Generation) pipeline: given a question, retrieve the most relevant context document, then send the question + retrieved context to the LLM for inference.*

### Task 3 与 Task 1/2 的关系

| | Task 1 | Task 2 | Task 3 |
|:---|:---|:---|:---|
| **Context 选择** | 手动指定（每个请求绑定固定论文） | 手动指定 | **RAG 自动检索**（根据问题匹配论文） |
| **端点** | `/v2/chat/completions` | `/v2/batch/chat/completions` | `/v2/batch/chat/completions`（复用 Task 2） |
| **调度** | FIFO | baseline vs context_grouped | baseline vs context_grouped（复用 Task 2） |
| **新增部分** | — | Scheduler 实现 | **TF-IDF 检索 + 准确率评估** |

### 完整 Pipeline

```
                        Task 3 Pipeline
┌──────────────────────────────────────────────────────┐
│  1. 加载 14 篇论文 (.txt)                              │
│  2. 构建 TF-IDF 向量索引                               │
│  3. 为每篇论文生成 2 个评估问题 (基于关键词模板)          │
│  4. 对每个问题做 TF-IDF cosine 检索 → 找到最相关的论文    │
│  5. 评估检索准确率 (top-1 accuracy)                     │
│  6. 将 (问题, 检索到的论文) 打包成 InferenceRequest      │
│  7. 打乱 → 切分 batch → 提交到 Task 2 的 batch 端点      │
│  8. 对比 baseline vs context_grouped 的推理性能          │
└──────────────────────────────────────────────────────┘
```

---

## 2. 核心代码走读

### 2.1 TF-IDF 检索引擎 — `~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/rag_retriever.py`

```python
class TfidfRagIndex:
    """稀疏 TF-IDF 文档索引 + 余弦相似度搜索"""

    def __init__(self, contexts: dict[str, str]):
        # [Task3] 对每篇论文做分词 (tokenize)
        self._doc_tokens = {cid: tokenize(text) for cid, text in contexts.items()}
        # [Task3] 计算 IDF: log((N+1)/(df+1)) + 1，衡量每个词的全局稀有度
        self._idf = self._build_idf(self._doc_tokens.values())
        # [Task3] 为每篇文档生成 TF-IDF 向量并 L2 归一化
        self._vectors = {cid: self._vectorize_tokens(tokens) ...}

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        # [Task3] 将 query 也转成 TF-IDF 向量
        q_vec = self.vectorize_query(query)
        # [Task3] 计算 query 与所有文档的余弦相似度，按分数降序排列
        scored = [SearchHit(cid, cosine(q_vec, d_vec), ...) for cid, d_vec in self._vectors.items()]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def top_terms(self, context_id: str, *, n: int = 5) -> list[str]:
        # [Task3] 返回该文档中 TF-IDF 权重最高的 n 个关键词（用于生成评估问题）
        ...
```

### 2.2 评估问题生成 — `~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/run_task3.py`

```python
QUERY_TEMPLATES = [
    "What is the main contribution of the work about {keywords}?",
    "What problem is addressed by the paper discussing {keywords}?",
]

def _make_eval_queries(contexts, index, ...) -> list[RagQuery]:
    for context_id in sorted(contexts):
        # [Task3] 提取该论文的 top-6 关键词
        terms = index.top_terms(context_id, n=6)
        for i in range(questions_per_context):
            # [Task3] 从关键词中选 3 个填入模板，生成评估问题
            keywords = " ".join(terms[i : i + 3])
            question = template.format(keywords=keywords)
            # [Task3] 用该问题去检索 → 得到 top-k 匹配的论文
            hits = index.search(question, top_k=top_k)
            # [Task3] 记录: expected_context_id (真实答案) vs retrieved_context_id (检索结果)
            ...
```

### 2.3 RAG → 推理请求转换 — `run_task3.py`

```python
def _to_inference_requests(rag_queries, contexts) -> list[InferenceRequest]:
    for i, item in enumerate(rag_queries):
        # [Task3] 用检索到的论文（不是原始论文）作为 context
        context_text = contexts[item.retrieved_context_id]
        requests.append(InferenceRequest(
            context_id=item.retrieved_context_id,  # [Task3] 使用检索结果的 ID
            question=item.question,
            context_text=context_text,
            experiment="rag",
            ...
        ))
```

### 2.4 主流程 — `run_task3.py: main()`

```python
def main():
    # [Task3] 步骤 1: 加载论文 + 构建 TF-IDF 索引
    contexts = load_text_contexts(args.data_dir, exclude=args.exclude)
    index = TfidfRagIndex(contexts)

    # [Task3] 步骤 2: 生成评估问题并执行检索
    rag_queries = _make_eval_queries(contexts, index, ...)

    # [Task3] 步骤 3: 统计检索准确率
    retrieval = _retrieval_summary(rag_queries)

    # [Task3] 步骤 4: 打乱后转成 InferenceRequest，复用 Task 2 的 batch 推理
    random.Random(args.seed).shuffle(generation_queries)
    requests = _to_inference_requests(generation_queries, contexts)
    batches = _batch_requests(requests, args.batch_size)

    # [Task3] 步骤 5: 调用 Task 2 的 run_experiment() 执行 baseline vs context_grouped
    batch_sched = task2.BatchSchedule(batches=batches, ...)
    payload["generation"] = task2.run_experiment(args.api_base, batch_sched, schedulers, ...)
```

---

## 3. 实验命令

> [!IMPORTANT]
> 所有命令在 Terminal 4（实验终端）中执行，且需要 Terminal 1 (LMCache Server) 和 Terminal 2 (vLLM) 保持运行。

### 3.1 仅测试检索（不做推理）

```bash
cd ~/shared/ik2221_project2
source ./venv/bin/activate

# 只跑 RAG 检索，查看 top-1 准确率
python lmcache-vllm-extended/benchmark/run_task3.py --retrieve-only
```

### 3.2 完整实验（检索 + 推理）

```bash
# 默认: 28 个请求 (14 papers × 2 questions), batch_size=28, baseline + context_grouped
python lmcache-vllm-extended/benchmark/run_task3.py \
  --scheduler both --batch-size 28 --cache-gb 0.2
```

### 3.3 仅重新绘图

```bash
python lmcache-vllm-extended/benchmark/run_task3.py \
  --plot-only --stem task3_rag_cache0.2
```

---

## 4. 已有实验结果速查

### 结果文件路径
```
~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/results/
└── task3_rag_cache0.2.json / .png
```

### 检索结果
```
Contexts: 14 篇论文
Queries:  28 个评估问题 (14 × 2)
Top-1 Accuracy: 100.00% (28/28)    ← TF-IDF 对这些模板化问题非常有效
Avg Top-1 Score: 0.3043
```

### 推理结果 (baseline vs context_grouped)
```
Scheduler           n    req/s    RT(s)   TTFT(s)
baseline            28   1.401    0.714   0.069
context_grouped     28   1.406    0.711   0.069
```

### 关键观察
- **检索准确率 100%**：因为问题是从各论文的 top 关键词生成的，TF-IDF 能轻松匹配回原文
- **推理性能 baseline ≈ context_grouped**：因为 RAG 检索产生的 context 分布较均匀，且论文长度相对一致，Cache 压力不大，两者差异不明显
- 如果想看到更大差异，可以增加 `--questions-per-context` 或用 `--no-shuffle`

---

## 5. 预期 Demo/答辩问题 & 回答

### Q: Task 3 的 RAG 和传统 RAG 有什么区别？
> **A:** 传统 RAG 通常使用 dense embedding（如 BERT/Sentence-BERT）做检索。我们的 Task 3 使用轻量级的 **TF-IDF + 余弦相似度**做稀疏检索，优势是不需要额外的 GPU 推理来生成 embedding，可以在 CPU 上快速完成。这对于 benchmark 场景已经足够。

### Q: 为什么 top-1 准确率是 100%？
> **A:** 因为评估问题是从每篇论文自身的 **top TF-IDF 关键词**生成的（比如用 "PagedAttention vLLM throughput" 造出 "What is the main contribution of the work about PagedAttention vLLM throughput?"）。这些关键词在原文中出现频率最高且具有区分性，TF-IDF 检索几乎不可能匹配到其他论文。如果用真实用户的自由提问，准确率会下降。

### Q: 为什么 baseline 和 context_grouped 在 Task 3 中性能差不多？
> **A:** 两个原因：(1) RAG 检索准确率是 100%，每个问题只会被分配到一篇论文，所以 28 个请求分布在 14 个 context 上（每个 context 恰好 2 个请求），多样性相对较低；(2) 请求量少（只有 28 个）且论文长度均匀，Cache 不会溢出，所以 baseline 本身表现就不差。在更大规模、更长文档、更不均匀的场景下差异会更明显。

### Q: Task 3 怎么和 Task 2 结合的？
> **A:** Task 3 的 **前半段是 RAG 检索**（生成问题 → TF-IDF 匹配论文 → 评估准确率），**后半段完全复用 Task 2 的 batch 推理**（`run_task2.run_experiment()`）。它把 RAG 检索结果转成 `InferenceRequest` 列表，打乱后打包成 `BatchSchedule`，再交给 `/v2/batch/chat/completions` 端点执行。

### Q: TF-IDF 的 IDF 公式是什么？
> **A:** `IDF(t) = log((N+1)/(df(t)+1)) + 1`，其中 N 是总文档数（14），df(t) 是包含词 t 的文档数。+1 是平滑项，防止除零和零权重。TF 使用的是子线性 TF：`TF(t) = 1 + log(count(t))`。最终向量做 L2 归一化，使得余弦相似度 = 内积。
