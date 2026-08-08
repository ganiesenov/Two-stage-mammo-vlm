"""
Доверительные интервалы и парные тесты значимости (CPU, после 01_infer.py).

Отвечает на:
  R1.9 — «bootstrap на 52 примерах даёт нестабильные p; дайте CI для всех метрик»
  R2.2 — «дайте CI или повторные оценки»
  R3.6 — «заявления про 2× экономию данных слишком сильные»

Что исправлено против scripts/4_evaluate/significance.py:
  1. Старый тест перемешивал две выборки как независимые, хотя это одни и те же
     52 изображения → нужен ПАРНЫЙ перестановочный тест (перестановка знаков разностей).
  2. Не было доверительных интервалов вообще → BCa bootstrap (с коррекцией смещения
     и ускорения), а не голый перцентильный.

Дополнительно считается sensitivity analysis на чистом подмножестве теста
(без 3 случаев с дословными дубликатами в train — см. 00_split_manifest.py).
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR, per_sample_scores

OUT_DIR = f"{ROOT}/results/revision"
METRICS = ["bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "meteor"]
N_BOOT = 10000
RNG = np.random.default_rng(42)


def bca_ci(x, alpha=0.05, n_boot=N_BOOT):
    """BCa-интервал для среднего. Перцентильный bootstrap на n=52 смещён,
    BCa корректирует и смещение, и асимметрию."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    theta = x.mean()
    boot = np.array([x[RNG.integers(0, n, n)].mean() for _ in range(n_boot)])

    # z0 — коррекция смещения
    prop = np.mean(boot < theta)
    prop = min(max(prop, 1 / n_boot), 1 - 1 / n_boot)
    from scipy.stats import norm
    z0 = norm.ppf(prop)

    # a — ускорение через jackknife
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    def adj(q):
        z = norm.ppf(q)
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = np.percentile(boot, 100 * adj(alpha / 2))
    hi = np.percentile(boot, 100 * adj(1 - alpha / 2))
    return float(theta), float(lo), float(hi)


def paired_permutation(a, b, n_perm=N_BOOT):
    """Парный перестановочный тест: случайно меняем знаки разностей.
    Корректен для одних и тех же тестовых изображений."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    obs = d.mean()
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p)


def load_rows(name, split, keep_ids=None):
    path = f"{PRED_DIR}/{name}__{split}.json"
    if not os.path.exists(path):
        return None
    preds = json.load(open(path, encoding="utf-8"))["predictions"]
    if keep_ids is not None:
        preds = [p for p in preds if p["id"] in keep_ids]
    rows = per_sample_scores(preds)
    return {m: [r[m] for r in rows] for m in METRICS}, [r["id"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--compare", nargs="*", default=[],
                    help="пары через двоеточие, напр. two_stage_r64:dmid_only_r64")
    ap.add_argument("--clean-only", action="store_true",
                    help="только чистое подмножество (без дословных дубликатов)")
    args = ap.parse_args()

    keep = None
    tag = ""
    if args.clean_only:
        man = json.load(open(f"{OUT_DIR}/split_manifest.json"))
        keep = set(man["clean_test_ids"])
        tag = "_clean"
        print(f"Sensitivity analysis: только {len(keep)} чистых случаев "
              f"(исключены {man['verbatim_duplicate_leakage']['test_ids']})")

    data, out = {}, {"split": args.split, "clean_only": args.clean_only,
                     "n_bootstrap": N_BOOT, "ci": {}, "comparisons": {}}
    for m in args.models:
        r = load_rows(m, args.split, keep)
        if r is None:
            print(f"  пропуск {m} — нет предсказаний")
            continue
        data[m] = r[0]
        out["ci"][m] = {}
        print(f"\n{'='*74}\n  {m}  (n={len(r[1])})\n{'='*74}")
        print(f"  {'metric':<10s} {'mean':>8s}   {'95% CI (BCa)':>22s}")
        for met in METRICS:
            th, lo, hi = bca_ci(data[m][met])
            out["ci"][m][met] = {"mean": round(th, 4), "ci_low": round(lo, 4),
                                 "ci_high": round(hi, 4)}
            print(f"  {met:<10s} {th:>8.4f}   [{lo:>7.4f}, {hi:>7.4f}]")

    for pair in args.compare:
        a, b = pair.split(":")
        if a not in data or b not in data:
            print(f"  пропуск сравнения {pair}")
            continue
        print(f"\n{'='*74}\n  {a}  vs  {b}\n{'='*74}")
        out["comparisons"][pair] = {}
        print(f"  {'metric':<10s} {'Δ':>9s} {'95% CI of Δ':>22s} {'p (paired perm)':>17s}")
        for met in METRICS:
            diff = np.array(data[a][met]) - np.array(data[b][met])
            _, lo, hi = bca_ci(diff)
            obs, p = paired_permutation(data[a][met], data[b][met])
            out["comparisons"][pair][met] = {
                "delta": round(obs, 4), "ci_low": round(lo, 4),
                "ci_high": round(hi, 4), "p_paired_perm": round(p, 4),
                "significant": bool(p < 0.05)}
            star = " *" if p < 0.05 else ""
            print(f"  {met:<10s} {obs:>+9.4f} [{lo:>+8.4f}, {hi:>+8.4f}] {p:>16.4f}{star}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/bootstrap_ci_{args.split}{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
