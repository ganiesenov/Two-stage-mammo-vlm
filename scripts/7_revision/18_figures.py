"""
Перестроение рисунков рукописи под исправленное сравнение (CPU).

fig_dmid_ablation.png в исходной версии сравнивал DMID-only при r=16
с two-stage при r=64, то есть воспроизводил спутывание схемы обучения
с ёмкостью адаптера — то же, что исправлено в Table 4. Здесь обе схемы
показаны при обоих рангах, парами.

BERTScore берётся из bertscore_test.json (единый протокол, roberta-large),
остальные метрики — из scores_test.json.
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

OUT = f"{ROOT}/manuscript/fig_dmid_ablation.png"
RES = f"{ROOT}/results/revision"

# AMRG приводится по первоисточнику (arXiv:2508.09225); BLEU-4 там не сообщается
AMRG = {"bleu4": None, "rouge1": 0.575, "rougeL": 0.5691, "bertscore": None}

GROUPS = [
    ("MedGemma\n(zero-shot)",        "zero_shot"),
    ("DMID-only\n$r$=16",            "dmid_only_r16"),
    ("Two-stage\n$r$=16",            "two_stage_r16"),
    ("DMID-only\n$r$=64",            "dmid_only_r64"),
    ("Two-stage\n$r$=64",            "two_stage_r64"),
    ("AMRG\n(Sung et al.)",          "__amrg"),
]
METRICS = [("bleu4", "BLEU-4"), ("rouge1", "ROUGE-1"),
           ("rougeL", "ROUGE-L"), ("bertscore", "BERTScore")]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def main():
    scores = json.load(open(f"{RES}/scores_test.json"))
    bert = json.load(open(f"{RES}/bertscore_test.json"))["scores"]

    def val(key, metric):
        if key == "__amrg":
            return AMRG[metric]
        if metric == "bertscore":
            return bert[key]["f1"] if key in bert else None
        return scores[key].get(metric)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    n_m = len(METRICS)
    width = 0.19
    x = np.arange(len(GROUPS))

    for j, (mkey, mlabel) in enumerate(METRICS):
        vals, pos = [], []
        for i, (_, key) in enumerate(GROUPS):
            v = val(key, mkey)
            if v is None:
                continue
            vals.append(v)
            pos.append(x[i] + (j - (n_m - 1) / 2) * width)
        bars = ax.bar(pos, vals, width, label=mlabel, color=COLORS[j],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, rotation=45)

    # скобки над парами при равном ранге
    for lo, hi, label in [(1, 2, "matched $r$=16"), (3, 4, "matched $r$=64")]:
        y = 1.03
        ax.plot([x[lo], x[lo], x[hi], x[hi]], [y, y + 0.02, y + 0.02, y],
                lw=1.1, color="0.35", clip_on=False)
        ax.text((x[lo] + x[hi]) / 2, y + 0.035, label, ha="center",
                va="bottom", fontsize=9.5, style="italic", color="0.25")

    ax.text(x[-1], 0.72, "BLEU-4 and BERTScore\nnot reported by AMRG",
            ha="center", va="bottom", fontsize=8.5, style="italic", color="0.45")

    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in GROUPS], fontsize=11)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_title("DMID test set evaluation ($n$=52, real radiologist reports)",
                 fontsize=14, pad=26)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
