#!/usr/bin/env python3
"""
Task 4 ：PCA + KMeans + NMI/ARI + Visualization

读取 extract_kv_features.py 生成的 kv_features.npz，完成：
  1. PCA 降维（512D → 2D 用于可视化，可选更高维度用于聚类）
  2. KMeans 聚类
  3. NMI / ARI 指标评估（对照 ground truth context_id）
  4. 生成散点图与评估报告

用法:
  python analyze_kv_clusters.py
  python analyze_kv_clusters.py --input results/kv_features.npz --n-clusters 14
  python analyze_kv_clusters.py --pca-dim 10   # 聚类前先降到 10 维再跑 KMeans
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境下也能生成图片
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

_BENCHMARK_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARK_DIR / "results"


# ───────────────────────────────────────────────────────────────
# 数据加载与预处理
# ───────────────────────────────────────────────────────────────

# [Task4] Load feature vectors from extract_kv_features.py output
def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 extract_kv_features.py 输出的 .npz 文件。"""
    data = np.load(path, allow_pickle=True)
    features = data["features"]           # (N, 512)
    context_ids = data["context_ids"]     # (N,) 字符串数组
    question_labels = data.get("question_labels", np.array([]))
    return features, context_ids, question_labels


def shorten_label(name: str) -> str:
    """缩短 paper 名称以便在图上显示。"""
    name = name.replace("_summary", "").replace("_", " ")
    if len(name) > 18:
        return name[:16] + "…"
    return name


# ───────────────────────────────────────────────────────────────
# PCA 降维
# ───────────────────────────────────────────────────────────────

# [Task4] PCA: reduce feature dimensionality (512D → 2D for visualization)
def run_pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    """对特征矩阵做 PCA 降维，返回降维后的数据和 PCA 模型。"""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    return reduced, pca


# ───────────────────────────────────────────────────────────────
# KMeans 聚类
# ───────────────────────────────────────────────────────────────

# [Task4] KMeans clustering: k = number of unique papers
def run_kmeans(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """对特征矩阵执行 KMeans 聚类，返回每个样本的簇标签。"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


# ───────────────────────────────────────────────────────────────
# 评估指标
# ───────────────────────────────────────────────────────────────

# [Task4] Evaluate clustering quality: NMI, ARI, Silhouette
def evaluate(true_labels: np.ndarray, pred_labels: np.ndarray,
             features: np.ndarray) -> dict[str, float]:
    """计算聚类质量的各项指标。"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)  # [Task4] 0~1, higher=better
    ari = adjusted_rand_score(true_labels, pred_labels)  # [Task4] -1~1, higher=better

    metrics = {"NMI": nmi, "ARI": ari}

    # Silhouette Score 需要至少 2 个簇且样本数 > 簇数
    n_unique = len(set(pred_labels))
    if 2 <= n_unique < len(pred_labels):
        sil = silhouette_score(features, pred_labels)
        metrics["Silhouette"] = sil

    return metrics


# ───────────────────────────────────────────────────────────────
# 可视化
# ───────────────────────────────────────────────────────────────

# [Task4] Plot: Ground Truth (left) vs KMeans (right) scatter plots on PCA 2D
def plot_clusters(
    features_2d: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    context_ids: np.ndarray,
    metrics: dict[str, float],
    pca: PCA,
    output_dir: Path,
) -> None:
    """生成两张并排的散点图：左侧按真实标签着色，右侧按聚类结果着色。"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 编码真实标签为整数以便着色
    le = LabelEncoder()
    true_int = le.fit_transform(true_labels)
    unique_contexts = le.classes_

    # 选择配色方案
    n_colors = max(len(unique_contexts), len(set(pred_labels)))
    cmap = plt.cm.get_cmap("tab20", n_colors)

    # ── 左图：按真实 context_id 着色 ──
    ax = axes[0]
    for i, ctx in enumerate(unique_contexts):
        mask = true_int == i
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[cmap(i)], label=shorten_label(ctx),
            s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Ground Truth (by Context ID)")
    ax.legend(fontsize=7, loc="best", ncol=1, framealpha=0.8)

    # ── 右图：按 KMeans 聚类结果着色 ──
    ax = axes[1]
    for cluster_id in sorted(set(pred_labels)):
        mask = pred_labels == cluster_id
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[cmap(cluster_id)], label=f"Cluster {cluster_id}",
            s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("KMeans Clustering Result")
    ax.legend(fontsize=7, loc="best", ncol=1, framealpha=0.8)

    # 在图片底部显示评估指标
    metrics_str = "  |  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    fig.suptitle(
        f"KV Cache Feature Clustering — {metrics_str}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "kv_cluster_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


# [Task4] Plot PCA explained variance curve to determine optimal number of components
def plot_pca_variance(pca_full: PCA, output_dir: Path) -> None:
    """绘制 PCA 方差解释率曲线，帮助判断该保留多少主成分。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n = len(cumvar)
    ax.bar(range(1, n + 1), pca_full.explained_variance_ratio_,
           alpha=0.6, label="Individual")
    ax.plot(range(1, n + 1), cumvar, "ro-", markersize=4, label="Cumulative")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, n + 0.5)
    plt.tight_layout()
    out_path = output_dir / "kv_pca_variance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA variance plot saved to {out_path}")


# ───────────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4: KV Cache Cluster Analysis")
    p.add_argument(
        "--input", type=Path,
        default=_RESULTS_DIR / "kv_features.npz",
    )
    p.add_argument(
        "--n-clusters", type=int, default=None,
        help="Number of KMeans clusters (default: auto = number of unique contexts)",
    )
    p.add_argument(
        "--pca-dim", type=int, default=None,
        help="PCA dims for clustering (default: use original 512D for KMeans)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_RESULTS_DIR,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──
    print("=" * 60)
    print("Loading features")
    print("=" * 60)
    features, context_ids, question_labels = load_features(args.input)
    print(f"  Features shape: {features.shape}")
    print(f"  Unique contexts: {len(set(context_ids))}")

    n_clusters = args.n_clusters or len(set(context_ids))
    print(f"  KMeans clusters: {n_clusters}")

    # ── PCA 全维分析（用于方差解释率曲线） ──
    max_components = min(features.shape[0], features.shape[1])
    pca_full = PCA(n_components=max_components, random_state=42)
    pca_full.fit(features)
    plot_pca_variance(pca_full, args.output_dir)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"  Components for 95% variance: {n95}")

    # ── PCA 降到 2D（用于散点图可视化） ──
    features_2d, pca_2d = run_pca(features, n_components=2)
    print(f"  PCA 2D variance explained: "
          f"{pca_2d.explained_variance_ratio_.sum():.1%}")

    # ── 确定用于聚类的特征 ──
    if args.pca_dim is not None:
        cluster_features, _ = run_pca(features, n_components=args.pca_dim)
        print(f"  Clustering on PCA-{args.pca_dim}D features")
    else:
        cluster_features = features
        print(f"  Clustering on original {features.shape[1]}D features")

    # ── KMeans 聚类 ──
    print("\n" + "=" * 60)
    print("Running KMeans")
    print("=" * 60)
    pred_labels = run_kmeans(cluster_features, n_clusters)
    print(f"  Cluster distribution: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

    # ── 评估 ──
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    metrics = evaluate(context_ids, pred_labels, cluster_features)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # ── 可视化 ──
    print("\n" + "=" * 60)
    print("Generating plots")
    print("=" * 60)
    plot_clusters(
        features_2d, context_ids, pred_labels,
        context_ids, metrics, pca_2d, args.output_dir,
    )

    # ── 保存聚类结果到文本文件 ──
    report_path = args.output_dir / "kv_cluster_report.txt"
    with open(report_path, "w") as f:
        f.write("KV Cache Cluster Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Samples: {len(features)}\n")
        f.write(f"Feature dim: {features.shape[1]}\n")
        f.write(f"Unique contexts: {len(set(context_ids))}\n")
        f.write(f"KMeans clusters: {n_clusters}\n")
        if args.pca_dim:
            f.write(f"PCA dim for clustering: {args.pca_dim}\n")
        f.write(f"PCA 2D variance: {pca_2d.explained_variance_ratio_.sum():.4f}\n")
        f.write(f"Components for 95% var: {n95}\n\n")

        f.write("Metrics\n")
        f.write("-" * 30 + "\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")

        f.write(f"\nPer-sample results\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Context':<30} {'Cluster':>8}\n")
        for cid, cl in zip(context_ids, pred_labels):
            f.write(f"  {cid:<28} {cl:>6}\n")

    print(f"  Report saved to {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
