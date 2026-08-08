r"""
Перерисовка Figure 1 — схемы двухэтапной архитектуры (CPU).

Исходная схема (Figma, two_stage_framework_corrected.pdf) содержала четыре
расхождения с текстом и кодом:
  1. в блоке синтеза указан «Gemini 1.5 Flash», тогда как отчёты порождались
     llama3.1:8b через Ollama (scripts/1_generate/generate_ollama.py:11);
  2. «100% QA pass» — на деле прошли проверку 390 из 400;
  3. в конвейере инференса стоял шаг «RAG retrieval», хотя финальная модель
     retrieval не использует (абляция отрицательная);
  4. перенос Stage 1 → Stage 2 изображён единственной стрелкой с подписью
     «pretrained init», хотя механизмов два, и для r=64 работает слияние.

Выход: PDF (вектор, для LaTeX) и PNG (для просмотра).
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

OUT_PDF = f"{ROOT}/manuscript/fig_pipeline_twostage.pdf"
OUT_PNG = f"{ROOT}/manuscript/fig_pipeline_twostage.png"

INK      = "#16202b"
MUTED    = "#5b6b7a"
RULE     = "#c9d4de"
S1       = "#1f5fa8"   # Stage 1
S2       = "#a8331f"   # Stage 2
FROZEN   = "#eef2f6"
TRAIN    = "#fdf3d8"
TRAIN_E  = "#c99a20"
PANEL    = "#fafcfd"


def box(ax, x, y, w, h, fc, ec, lw=1.0, r=1.4, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def txt(ax, x, y, s, size=8.2, color=INK, weight="normal", ha="center", va="center",
        style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style=style, zorder=z, linespacing=1.45)


def arrow(ax, p1, p2, color=INK, lw=1.3, ls="-", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(14.6, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── панели ───────────────────────────────────────────────────────
    box(ax, 1.5, 20, 45, 74, PANEL, RULE, 1.0, z=1)
    box(ax, 53.5, 20, 45, 74, PANEL, RULE, 1.0, z=1)

    box(ax, 1.5, 88.5, 45, 5.5, S1, S1, z=3)
    txt(ax, 24, 91.2, "STAGE 1  ·  SYNTHETIC PRETRAINING", 9.4, "white", "bold")
    box(ax, 53.5, 88.5, 45, 5.5, S2, S2, z=3)
    txt(ax, 76, 91.2, "STAGE 2  ·  REAL-DATA FINE-TUNING", 9.4, "white", "bold")

    # ── Stage 1: источники ───────────────────────────────────────────
    box(ax, 3.5, 78, 20, 8.5, "white", RULE)
    txt(ax, 13.5, 84.6, "VinDr-Mammo", 8.6, INK, "bold")
    txt(ax, 13.5, 81.2, "5,000 exams · 20,000 images\nBI-RADS 0–5 · ACR A–D", 7.3, MUTED)
    box(ax, 24.5, 78, 20, 8.5, "white", RULE)
    txt(ax, 34.5, 84.6, "CBIS-DDSM", 8.6, INK, "bold")
    txt(ax, 34.5, 81.2, "1,566 patients\nmasses · calcifications", 7.3, MUTED)
    txt(ax, 24, 75.6, "structured annotations only — no free-text reports", 7.4, MUTED,
        style="italic")

    arrow(ax, (24, 75.0), (24, 72.2))

    # ── Stage 1: синтез ──────────────────────────────────────────────
    box(ax, 3.5, 60.5, 41, 11.5, "white", RULE)
    txt(ax, 24, 69.6, "Label-to-report synthesis", 8.8, INK, "bold")
    txt(ax, 24, 66.6, "Llama-3.1-8B  (local inference, Ollama)", 8.0, S1)
    txt(ax, 24, 63.2,
        "400 reports generated  →  390 passed automated validation (97.5%)\n"
        "VinDr-Mammo 196/200  ·  CBIS-DDSM 194/200", 7.4, MUTED)

    txt(ax, 24, 58.4, "390 synthetic image–report pairs", 7.6, MUTED, style="italic")
    arrow(ax, (24, 60.0), (24, 55.0))

    # ── Stage 1: модель ──────────────────────────────────────────────
    box(ax, 3.5, 33, 41, 21, "white", RULE)
    txt(ax, 24, 51.4, "MedGemma-4B", 8.8, INK, "bold")
    for x0, name, sub in [(5.5, "SigLIP", "vision encoder"),
                          (18.0, "Gemma-3", "language model")]:
        box(ax, x0, 39.5, 11.5, 8.5, FROZEN, RULE)
        txt(ax, x0 + 5.75, 45.2, name, 8.0, INK, "bold")
        txt(ax, x0 + 5.75, 42.6, sub + "\nfrozen", 7.0, MUTED)
    box(ax, 30.5, 39.5, 12, 8.5, TRAIN, TRAIN_E)
    txt(ax, 36.5, 45.2, "LoRA", 8.0, INK, "bold")
    txt(ax, 36.5, 42.6, "trainable\n$r$=16, $\\alpha$=32", 7.0, MUTED)
    txt(ax, 24, 36.2, "4-bit NF4 · 448×448 · 3 epochs", 7.2, MUTED)

    box(ax, 8.5, 24.5, 31, 6.5, S1, S1, z=3)
    txt(ax, 24, 27.7, "Stage 1 LoRA weights  $B_1A_1$", 8.4, "white", "bold")
    arrow(ax, (24, 32.8), (24, 31.2))

    # ── Stage 2: DMID ────────────────────────────────────────────────
    box(ax, 55.5, 78, 41, 8.5, "white", RULE)
    txt(ax, 76, 84.6, "DMID — real radiologist reports", 8.8, INK, "bold")
    txt(ax, 76, 81.2,
        "225 cases · 510 images  ·  image-level split\n"
        "407 train  ·  51 validation  ·  52 test", 7.3, MUTED)

    # ── Stage 2: модель ──────────────────────────────────────────────
    box(ax, 55.5, 50, 41, 21, "white", RULE)
    txt(ax, 76, 68.4, "MedGemma-4B", 8.8, INK, "bold")
    for x0, name, sub in [(57.5, "SigLIP", "vision encoder"),
                          (70.0, "Gemma-3", "language model")]:
        box(ax, x0, 56.5, 11.5, 8.5, FROZEN, RULE)
        txt(ax, x0 + 5.75, 62.2, name, 8.0, INK, "bold")
        txt(ax, x0 + 5.75, 59.6, sub + "\nfrozen", 7.0, MUTED)
    box(ax, 82.5, 56.5, 12, 8.5, TRAIN, TRAIN_E)
    txt(ax, 88.5, 62.2, "LoRA", 8.0, INK, "bold")
    txt(ax, 88.5, 59.6, "trainable\n$r$=64, $\\alpha$=64", 7.0, MUTED)
    txt(ax, 76, 53.2, "4-bit NF4 · 448×448 · 5 epochs · best by validation loss", 7.2, MUTED)
    arrow(ax, (76, 77.8), (76, 71.2))

    box(ax, 62, 42, 28, 6, S2, S2, z=3)
    txt(ax, 76, 45.0, "Final two-stage model", 8.6, "white", "bold")
    arrow(ax, (76, 49.8), (76, 48.2))

    # ── инференс без RAG ─────────────────────────────────────────────
    box(ax, 55.5, 22.5, 41, 15, "white", RULE)
    txt(ax, 76, 35.4, "INFERENCE", 7.6, MUTED, "bold")
    steps = ["Mammogram", "SigLIP\nencoding", "Prompt", "Greedy\ndecoding", "BI-RADS\nreport"]
    xs = [60.5, 68.0, 75.5, 83.0, 90.5]
    for i, (x0, s) in enumerate(zip(xs, steps)):
        fc = S2 if i == len(steps) - 1 else "white"
        col = "white" if i == len(steps) - 1 else INK
        box(ax, x0 - 3.4, 26.5, 6.8, 6.2, fc, S2 if i == len(steps) - 1 else RULE)
        txt(ax, x0, 29.6, s, 6.9, col)
        if i:
            arrow(ax, (xs[i - 1] + 3.6, 29.6), (x0 - 3.6, 29.6), MUTED, 1.0)
    txt(ax, 76, 24.3, "no retrieval step: the BI-RADS RAG module was implemented and ablated,\n"
                      "degrades generation, and is not used in the reported models", 7.0, MUTED,
        style="italic")

    # ── перенос весов: L-образный маршрут в промежутке между панелями ──
    ax.plot([39.7, 50.0], [27.7, 27.7], color=S1, lw=1.5, ls="--", zorder=4)
    ax.plot([50.0, 50.0], [27.7, 60.5], color=S1, lw=1.5, ls="--", zorder=4)
    arrow(ax, (50.0, 60.5), (55.3, 60.5), S1, 1.5, "--")
    txt(ax, 48.6, 44, "(a)", 7.6, S1, "bold", ha="right")

    # ── два механизма переноса ───────────────────────────────────────
    box(ax, 52.0, 5.5, 46.5, 11.5, "white", S1)
    txt(ax, 53.5, 14.6, "(a)  merge into base weights  —  used for the reported models",
        7.8, S1, "bold", ha="left")
    txt(ax, 53.5, 10.6,
        "$W_0' = W_0 + (\\alpha_1/r_1)\\,B_1A_1$, then a new adapter $B_2A_2$ of rank $r_2$\n"
        "is trained on top of $W_0'$. Adapter ranks need not match, so this is\n"
        "the only option when $r_2 \\neq r_1$ — including the final $r$=64 model.", 7.2, MUTED,
        ha="left")

    box(ax, 1.5, 5.5, 46.5, 11.5, "white", RULE)
    txt(ax, 3.0, 14.6, "(b)  continue training the same adapter", 7.8, INK, "bold", ha="left")
    txt(ax, 3.0, 10.6,
        "$B_1A_1$ is optimised further on DMID. Possible only when the ranks\n"
        "match ($r_2 = r_1$); used for the $r$=16 configuration. The two schemes\n"
        "give different models and are not interchangeable.", 7.2, MUTED, ha="left")

    # ── заголовок и легенда ──────────────────────────────────────────
    txt(ax, 50, 97.6, "Two-stage synthetic-to-real transfer for mammography report generation",
        11.2, INK, "bold")

    for x0, fc, ec, label in [(30.0, FROZEN, RULE, "frozen module"),
                              (46.0, TRAIN, TRAIN_E, "trainable (LoRA)")]:
        box(ax, x0, 1.2, 2.6, 2.2, fc, ec)
        txt(ax, x0 + 3.4, 2.3, label, 7.2, MUTED, ha="left")
    ax.plot([62.5, 66.5], [2.3, 2.3], color=S1, lw=1.5, ls="--")
    txt(ax, 67.3, 2.3, "weight transfer", 7.2, MUTED, ha="left")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=190, bbox_inches="tight")
    print(f"  → {OUT_PDF}\n  → {OUT_PNG}")


if __name__ == "__main__":
    main()
