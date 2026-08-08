"""
Improved JPCSIT figure generator — fixes Figure 3 readability.
Replace make_jpcsit_figures.py with this file.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

BASE = Path("/mnt/c/Users/juman/hard_ml/rag_mammo/new_article")
RESULTS = BASE / "jpcsit_results"
FIGS = BASE / "jpcsit_results" / "figures"
FIGS.mkdir(exist_ok=True, parents=True)


def load_json(path):
    if not path.exists():
        print(f"[warn] missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1 — Embedding model comparison (bar chart)
# ============================================================
def fig1_embeddings():
    data = load_json(RESULTS / "exp1_embeddings.json")
    if not data:
        return
    encoders = list(data.keys())
    metrics = ["bleu4", "rouge1", "rougeL", "bertscore"]
    metric_labels = ["BLEU-4", "ROUGE-1", "ROUGE-L", "BERTScore"]

    x = np.arange(len(encoders))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (m, lbl) in enumerate(zip(metrics, metric_labels)):
        vals = [data[e][m] for e in encoders]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=lbl, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Effect of embedding model on retrieval-augmented generation quality")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", ncol=4, frameon=True)
    plt.tight_layout()
    out = FIGS / "fig1_embeddings.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ============================================================
# Figure 2 — Top-k sweep (line plot)
# ============================================================
def fig2_topk():
    data = load_json(RESULTS / "exp2_topk.json")
    if not data:
        return
    keys = sorted(data.keys(), key=lambda s: int(s.split("=")[1]))
    ks = [int(k.split("=")[1]) for k in keys]
    rouge_l = [data[k]["rougeL"] for k in keys]
    halluc = [data[k]["hallucination_rate"] for k in keys]
    ctx_words = [data[k].get("avg_context_words", 0) for k in keys]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ks, rouge_l, "o-", color="#1f77b4", linewidth=2, markersize=8, label="ROUGE-L")
    ax1.set_xlabel("Number of retrieved fragments (k)")
    ax1.set_ylabel("ROUGE-L", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, halluc, "s--", color="#d62728", linewidth=2, markersize=8, label="Hallucination rate")
    ax2.set_ylabel("Hallucination rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    for k, c in zip(ks, ctx_words):
        ax1.annotate(f"{c:.0f}w", xy=(k, rouge_l[ks.index(k)]),
                     xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=8, color="gray")

    plt.title("Effect of top-k retrieval on report quality and hallucinations\n"
              "(annotated values: average context length in words)")
    fig.tight_layout()
    out = FIGS / "fig2_topk.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ============================================================
# Figure 3 — IMPROVED: latency bars + hallucination + ROUGE-L
# ============================================================
def fig3_faiss():
    data = load_json(RESULTS / "exp3_faiss.json")
    if not data:
        return
    perf = data["index_performance"]
    e2e = data["end_to_end_nlg"]
    names = list(perf.keys())  # FlatIP, IVFFlat, HNSWFlat

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    bar_colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    # ---------- Panel (a): build time + latency (in microseconds) ----------
    ax = axes[0]
    # Display in microseconds (since values are ~2-3 microseconds)
    latency_us = [perf[n]["mean_ms"] * 1000.0 for n in names]
    bars = ax.bar(names, latency_us, color=bar_colors, edgecolor="black", linewidth=0.7)
    for bar, val, name in zip(bars, latency_us, names):
        recall = perf[name]["recall_at_k=2"]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(latency_us) * 0.02,
                f"recall@2={recall:.3f}", ha="center", va="bottom",
                fontsize=9, color="black")
    ax.set_ylabel("Mean retrieval latency (μs)")
    ax.set_title("(a) Index latency (n=39 vectors)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(latency_us) * 1.25)

    # ---------- Panel (b): end-to-end NLG ----------
    ax = axes[1]
    rouge_l = [e2e[n]["rougeL"] for n in names]
    bertsc = [e2e[n]["bertscore"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, rouge_l, w, label="ROUGE-L", color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, bertsc, w, label="BERTScore", color="#ff7f0e",
           edgecolor="black", linewidth=0.5)
    for i, (r, b) in enumerate(zip(rouge_l, bertsc)):
        ax.text(i - w / 2, r + 0.02, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("(b) End-to-end NLG quality")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")

    # ---------- Panel (c): hallucination rate ----------
    ax = axes[2]
    halluc = [e2e[n]["hallucination_rate"] for n in names]
    bars = ax.bar(names, halluc, color=bar_colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, halluc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Hallucination rate")
    ax.set_title("(c) Hallucination rate per index")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = FIGS / "fig3_faiss.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ============================================================
# Figure 4 — Heatmap (encoders × metrics)
# ============================================================
def fig4_heatmap():
    data = load_json(RESULTS / "exp1_embeddings.json")
    if not data:
        return
    encoders = list(data.keys())
    metrics = ["bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "bertscore"]
    labels = ["BLEU-1", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

    M = np.array([[data[e][m] for m in metrics] for e in encoders])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(encoders)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(encoders)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            color = "white" if M[i, j] > 0.6 else "black"
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Metric value")
    ax.set_title("NLG metric matrix across embedding models")
    plt.tight_layout()
    out = FIGS / "fig4_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ============================================================
# Figure 5 — Context vs hallucination (Exp 2 secondary)
# ============================================================
def fig5_context_halluc():
    data = load_json(RESULTS / "exp2_topk.json")
    if not data:
        return
    keys = sorted(data.keys(), key=lambda s: int(s.split("=")[1]))
    ctx = [data[k].get("avg_context_words", 0) for k in keys]
    halluc = [data[k]["hallucination_rate"] for k in keys]
    rouge = [data[k]["rougeL"] for k in keys]
    ks = [int(k.split("=")[1]) for k in keys]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sc = ax.scatter(ctx, halluc, c=rouge, s=200, cmap="viridis",
                    edgecolors="black", linewidths=1.0, vmin=0.28, vmax=0.32)
    for k, x, y in zip(ks, ctx, halluc):
        ax.annotate(f"k={k}", xy=(x, y), xytext=(10, 8),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("ROUGE-L")
    ax.set_xscale("log")
    ax.set_xlabel("Average retrieved context length (words, log scale)")
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Trade-off between retrieved context length and hallucination rate")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = FIGS / "fig5_context_halluc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def main():
    fig1_embeddings()
    fig2_topk()
    fig3_faiss()
    fig4_heatmap()
    fig5_context_halluc()
    print(f"\nAll figures saved to: {FIGS}")


if __name__ == "__main__":
    main()