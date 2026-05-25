# IK2221 Project — Task 4 Demo Cheatsheet

> [!NOTE]
> **前提**: Task 4 使用与 Task 1/2/3 相同的 3 终端启动流程（见 Task 1 cheatsheet）。
> 实验命令在 **Terminal 4** 中执行。

## 目录
1. [Task 4 核心概念](#1-task-4-核心概念)
2. [核心代码走读](#2-核心代码走读)
3. [实验命令](#3-实验命令)
4. [已有实验结果速查](#4-已有实验结果速查)
5. [预期 Demo/答辩问题 & 回答](#5-预期-demo答辩问题--回答)

---

## 1. Task 4 核心概念

### 题目要求
> *Extract KV cache feature vectors from LMCache, apply PCA + KMeans clustering, and evaluate whether requests sharing the same context produce similar KV representations.*

### 两阶段 Pipeline

```
Stage 1: extract_kv_features.py — 特征提取
┌──────────────────────────────────────────────────────┐
│  1. 向 vLLM 发送 14×5=70 个推理请求（填充 LMCache）     │
│  2. 创建离线 LMCacheEngine，连接远端 Server             │
│  3. 对每个请求: retrieve(tokens) → KV Tensor            │
│     [num_layers, 2, num_tokens, num_kv_heads, head_size]│
│  4. Mean Pooling: tokens→均值, layers→均值, 展平        │
│     → 512 维特征向量 (2 × 2 heads × 128 dim)           │
│  5. 保存 → kv_features.npz (70 × 512)                  │
└──────────────────────────────────────────────────────┘

Stage 2: analyze_kv_clusters.py — 聚类分析
┌──────────────────────────────────────────────────────┐
│  1. 加载 kv_features.npz                               │
│  2. PCA: 512D → 2D (可视化) / 原始 512D (聚类)         │
│  3. KMeans: k=14 (等于 paper 数量)                     │
│  4. 评估: NMI / ARI / Silhouette Score                 │
│  5. 绘图: Ground Truth vs KMeans 散点图                 │
│  6. 报告: kv_cluster_report.txt                        │
└──────────────────────────────────────────────────────┘
```

### 核心假设
> 同一篇论文的不同问题共享同一段长 context（论文全文），它们的 KV Cache 前缀完全相同，只有最后几个 question tokens 不同。因此**同一 context 的 KV 特征向量应该在高维空间中聚在一起**。

---

## 2. 核心代码走读

### 2.1 发送请求填充缓存 — `~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/extract_kv_features.py`

```python
def send_requests_to_vllm(contexts, questions, api_base, ...):
    for cid in sorted(contexts.keys()):
        text = contexts[cid]
        for question in questions:
            # [Task4] 构造与 request_generator 一致的 3-turn 对话
            messages = build_messages(text, question)
            # [Task4] 非流式请求 → vLLM 推理 → KV Cache 自动写入 LMCache
            result = _post_json(chat_url, payload)
            time.sleep(0.3)  # [Task4] 等待 LMCache 异步写入完成
```

### 2.2 创建离线 Engine — `extract_kv_features.py`

```python
def create_offline_engine(config_path):
    # [Task4] 创建独立的 LMCacheEngine，连接同一个远端 Server
    config = LMCacheEngineConfig.from_file(config_path)
    # [Task4] kv_shape 必须与 vLLM 启动时一致 (Qwen2.5-1.5B: 28层, 2KV头, 128维)
    kv_shape = (NUM_LAYERS, 2, CHUNK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    metadata = LMCacheEngineMetadata(
        model_name=MODEL_NAME, world_size=1, worker_id=0,
        fmt="vllm", kv_dtype=KV_DTYPE, kv_shape=kv_shape,
    )
    engine = LMCacheEngineBuilder.get_or_create("offline_kv_extractor", config, metadata)
    return engine
```

### 2.3 KV 特征提取 + Mean Pooling — `extract_kv_features.py`

```python
def extract_kv_feature(engine, tokens) -> Optional[np.ndarray]:
    # [Task4] 从 LMCache 检索该 token 序列的完整 KV Cache
    kv_data, ret_mask = engine.retrieve(tokens, return_tuple=False)
    # kv_data 形状: [num_layers=28, 2(K/V), num_tokens, num_kv_heads=2, head_size=128]

    kv = kv_data.float()
    # [Task4] 对 tokens 维度求均值 → [28, 2, 2, 128]
    pooled = kv.mean(dim=2)
    # [Task4] 对 layers 维度求均值 → [2, 2, 128]
    compact = pooled.mean(dim=0)
    # [Task4] 展平为 1D 特征向量 → [512]  (2 × 2 × 128 = 512)
    return compact.cpu().numpy().flatten()
```

### 2.4 PCA + KMeans — `~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/analyze_kv_clusters.py`

```python
def run_pca(features, n_components):
    # [Task4] PCA 降维: 512D → 2D (用于可视化散点图)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    return reduced, pca

def run_kmeans(features, n_clusters):
    # [Task4] KMeans 聚类: k=14 (论文数), 对比真实 context_id 标签
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    return kmeans.fit_predict(features)

def evaluate(true_labels, pred_labels, features):
    # [Task4] NMI: 归一化互信息 (0~1, 越高越好)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    # [Task4] ARI: 调整兰德指数 (-1~1, 越高越好)
    ari = adjusted_rand_score(true_labels, pred_labels)
    # [Task4] Silhouette: 轮廓系数 (-1~1, 越高越好, 衡量簇内紧密度)
    sil = silhouette_score(features, pred_labels)
    ...
```

---

## 3. 实验命令

> [!IMPORTANT]
> 所有命令在 Terminal 4（实验终端）中执行，且需要 Terminal 1 (LMCache Server) 和 Terminal 2 (vLLM) 保持运行。

### 3.1 Stage 1: 提取 KV 特征

```bash
cd ~/shared/ik2221_project2
source ./venv/bin/activate

# 完整流程: 先发请求填充缓存，再提取特征
python lmcache-vllm-extended/benchmark/extract_kv_features.py --send-requests

# 使用 5 个不同问题 (每个 context 5 个样本 → 70 个特征向量)
python lmcache-vllm-extended/benchmark/extract_kv_features.py \
  --send-requests --questions-per-context 5

# 仅提取 (假设缓存已填充)
python lmcache-vllm-extended/benchmark/extract_kv_features.py
```

### 3.2 Stage 2: 聚类分析

```bash
# 默认: 14 clusters, 原始 512D 特征
python lmcache-vllm-extended/benchmark/analyze_kv_clusters.py

# 自定义聚类数
python lmcache-vllm-extended/benchmark/analyze_kv_clusters.py --n-clusters 7

# 先 PCA 降到 10 维再聚类
python lmcache-vllm-extended/benchmark/analyze_kv_clusters.py --pca-dim 10
```

---

## 4. 已有实验结果速查

### 结果文件路径
```
~/shared/ik2221_project2/lmcache-vllm-extended/benchmark/results/
├── kv_features.npz             ← 70 × 512 特征矩阵
├── kv_cluster_analysis.png     ← Ground Truth vs KMeans 散点图
├── kv_pca_variance.png         ← PCA 方差解释率曲线
└── kv_cluster_report.txt       ← 聚类评估报告
```

### 关键指标
```
Samples:           70 (14 papers × 5 questions)
Feature dim:       512 (2 × 2 heads × 128 head_size)
KMeans clusters:   14

NMI:               1.0000    ← 完美聚类
ARI:               1.0000    ← 完美聚类
Silhouette:        0.9313    ← 簇内极紧密

PCA 2D variance:   54.92%
95% variance:      需要 11 个主成分
```

### 结论
- **NMI = ARI = 1.0** 表明 KMeans 聚类结果与真实 context_id 标签**完全一致**
- 每篇论文的 5 个不同问题被 KMeans 100% 正确地聚到了同一个簇
- 这证明了：**同一 context 的 KV Cache 特征在高维空间中形成了清晰可分的簇，不同问题只改变了极少的末尾 tokens，不影响整体 KV 特征分布**

---

## 5. 预期 Demo/答辩问题 & 回答

### Q: 为什么要从 KV Cache 提取特征做聚类？
> **A:** 这验证了 KV Cache 共享（LMCache）的核心假设：如果同一 context 的请求确实产生了相似的 KV 表示，那么 context-grouped scheduler 的分组策略就有了理论支撑。聚类结果（NMI=1.0）证明了 **context_id 就是 KV 特征空间中的天然簇标签**。

### Q: Mean Pooling 的 512 维特征是怎么算出来的？
> **A:** KV Cache 原始形状是 `[28 layers, 2(K/V), N tokens, 2 heads, 128 dim]`。先对 tokens 维度做 mean（压缩 sequence length），再对 layers 维度做 mean（压缩 transformer 深度），最后 flatten 剩下的 `[2, 2, 128]` = 512 维。这保留了 K/V 的统计特征，丢弃了位置信息。

### Q: 为什么 NMI 和 ARI 都是 1.0？
> **A:** 因为同一篇论文的不同问题共享 **极长的 context prefix**（论文全文通常 > 3000 tokens），而不同的问题只有最后 10-20 个 tokens 不同。Mean Pooling 后，这些末尾差异被稀释到几乎可以忽略，导致同 context 的特征向量高度相似。

### Q: PCA 2D 方差只有 55%，可靠吗？
> **A:** PCA 2D 只用于**可视化**，不用于聚类。聚类使用的是**原始 512 维特征**。55% 的方差解释率在高维数据中是正常的（14 个 context 需要至少 13 维来完全分离）。从 PCA 方差曲线可以看到，11 个主成分就能解释 95% 的方差。

### Q: 这个结果对 Task 2 的调度器设计有什么启示？
> **A:** 由于 KV 特征按 context_id 完美聚类（Silhouette=0.93），这意味着：(1) 同 context 的请求可以高效共享 KV Cache（前缀几乎相同）；(2) context-grouped scheduler 的分组策略（按 context_id 排序）在物理层面是有意义的 —— 它不只是逻辑上的分组，而是确实让相似的 KV 数据被连续访问，最大化了硬件缓存的局部性。
