"""
TOST-анализ свипа по ёмкости (CPU, после 05_capacity_sweep.py).

Реализует правила решения, зафиксированные ЗАРАНЕЕ в results/revision/PREREGISTRATION.md:
  Δ_eq = 0.02 ROUGE-L, 90% CI, эквивалентность объявляется только если весь интервал
  лежит внутри (−Δ_eq, +Δ_eq).

Важно по существу: p=0.92 сам по себе НЕ доказывает отсутствие эффекта — он лишь означает
неспособность отвергнуть нулевую гипотезу. Утверждать «эффекта нет» позволяет только тест
на эквивалентность с заранее объявленным порогом, что здесь и делается.

Дисперсия оценивается на двух уровнях:
  1) внутри прогона — парная разность по 52 тестовым случаям;
  2) между прогонами — разброс по seed'ам (именно его не было в исходной работе).
"""
import os, sys, json, argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR, per_sample_scores

SWEEP = f"{ROOT}/results/revision/capacity_sweep.json"
OUT = f"{ROOT}/results/revision/equivalence.json"
DELTA_EQ = 0.02          # объявлено заранее, см. PREREGISTRATION.md
METRIC = "rougeL"
CONF = 0.90


def cell_name(arm, r, alpha, seed, tag=""):
    """Имя ячейки строго как в 05_capacity_sweep.py:cell_key.

    Тег корпуса относится ТОЛЬКО к two_stage: у dmid_only нет Stage 1, и его ячейки
    переиспользуются как общий компаратор для всех корпусов и доз.
    """
    return f"{arm}{tag if arm == 'two_stage' else ''}_r{r}_a{alpha}_seed{seed}"


def per_case(arm, r, alpha, seed, tag=""):
    p = f"{PRED_DIR}/sweep_{cell_name(arm, r, alpha, seed, tag)}__test.json"
    if not os.path.exists(p):
        return None
    preds = json.load(open(p, encoding="utf-8"))["predictions"]
    rows = per_sample_scores(preds)
    return {row["id"]: row[METRIC] for row in rows}


def tost(deltas, delta_eq=DELTA_EQ, conf=CONF):
    """Возвращает (mean, lo, hi, вердикт). CI по t-распределению."""
    d = np.asarray(deltas, float)
    n = len(d)
    m = d.mean()
    if n < 2:
        return m, None, None, "недостаточно прогонов"
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    lo, hi = m - t * se, m + t * se
    if lo > -delta_eq and hi < delta_eq:
        verdict = "ЭКВИВАЛЕНТНОСТЬ"
    elif lo > 0 and hi >= delta_eq:
        verdict = "ПРЕВОСХОДСТВО"
    elif hi < 0 and lo <= -delta_eq:
        verdict = "УСТУПАЕТ"
    else:
        verdict = "НЕОПРЕДЕЛЁННОСТЬ"
    return m, lo, hi, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-eq", type=float, default=DELTA_EQ)
    ap.add_argument("--corpus-tag", default="",
                    help="ветка корпуса Stage 1: '' — пререгистрированный v1, "
                         "'_v2d1444' / '_v2d800' / '_v2d369' — дозы корпуса v2 (EXPLORATORY), "
                         "'_clean' — v1 без русскоязычных отчётов")
    args = ap.parse_args()
    TAG = args.corpus_tag

    if not os.path.exists(SWEEP):
        sys.exit(f"Нет {SWEEP} — сначала 05_capacity_sweep.py")
    res = json.load(open(SWEEP))

    configs = sorted({(v["r"], v["alpha"]) for v in res.values()})
    out = {"delta_eq": args.delta_eq, "metric": METRIC, "confidence": CONF,
           "corpus_tag": TAG or "v1",
           "preregistered": "results/revision/PREREGISTRATION.md", "by_config": {}}

    print("=" * 78)
    print(f"  TOST: two-stage vs DMID-only,  метрика {METRIC},  Δ_eq = {args.delta_eq}")
    print(f"  корпус Stage 1: {TAG or 'v1 (пререгистрированный)'}")
    print(f"  (порог и правила объявлены до прогонов — PREREGISTRATION.md)")
    print("=" * 78)

    curve, skipped = [], []
    for r, alpha in configs:
        seeds = sorted({v["seed"] for v in res.values()
                        if v["r"] == r and v["alpha"] == alpha})
        per_seed_delta, paired_all = [], []
        for s in seeds:
            ts = res.get(cell_name("two_stage", r, alpha, s, TAG))
            do = res.get(cell_name("dmid_only", r, alpha, s, TAG))
            if not (ts and do):
                continue
            per_seed_delta.append(ts[METRIC] - do[METRIC])
            a = per_case("two_stage", r, alpha, s, TAG)
            b = per_case("dmid_only", r, alpha, s, TAG)
            if a and b:
                ids = sorted(set(a) & set(b))
                paired_all.append(np.array([a[i] - b[i] for i in ids]))

        # Молча пропущенная конфигурация читалась бы как «её не считали»,
        # хотя на деле это может быть опечатка в теге корпуса.
        if not per_seed_delta:
            skipped.append((r, alpha))
            continue

        m, lo, hi, verdict = tost(per_seed_delta, args.delta_eq)
        ts_vals = [res[cell_name("two_stage", r, alpha, s, TAG)][METRIC] for s in seeds
                   if cell_name("two_stage", r, alpha, s, TAG) in res]
        do_vals = [res[cell_name("dmid_only", r, alpha, s, TAG)][METRIC] for s in seeds
                   if cell_name("dmid_only", r, alpha, s, TAG) in res]

        print(f"\n  r = {r}, α = {alpha}   (прогонов на ветвь: {len(per_seed_delta)})")
        print(f"    two-stage  {np.mean(ts_vals):.4f} ± {np.std(ts_vals, ddof=1) if len(ts_vals)>1 else 0:.4f}"
              f"   значения: {[round(v,4) for v in ts_vals]}")
        print(f"    DMID-only  {np.mean(do_vals):.4f} ± {np.std(do_vals, ddof=1) if len(do_vals)>1 else 0:.4f}"
              f"   значения: {[round(v,4) for v in do_vals]}")
        if lo is not None:
            print(f"    Δ = {m:+.4f}   {int(CONF*100)}% CI [{lo:+.4f}, {hi:+.4f}]   → {verdict}")
        else:
            print(f"    Δ = {m:+.4f}   (CI недоступен)   → {verdict}")

        if paired_all:
            pooled = np.concatenate(paired_all)
            print(f"    внутрипрогонная парная разность по случаям: "
                  f"{pooled.mean():+.4f} (n={len(pooled)})")

        out["by_config"][f"r{r}_a{alpha}"] = {
            "r": r, "alpha": alpha,
            "n_runs": len(per_seed_delta),
            "two_stage_mean": round(float(np.mean(ts_vals)), 4),
            "two_stage_sd": round(float(np.std(ts_vals, ddof=1)), 4) if len(ts_vals) > 1 else None,
            "dmid_only_mean": round(float(np.mean(do_vals)), 4),
            "dmid_only_sd": round(float(np.std(do_vals, ddof=1)), 4) if len(do_vals) > 1 else None,
            "delta_mean": round(float(m), 4),
            "ci_low": round(float(lo), 4) if lo is not None else None,
            "ci_high": round(float(hi), 4) if hi is not None else None,
            "verdict": verdict,
            "per_seed_deltas": [round(float(d), 4) for d in per_seed_delta],
        }
        # В кривую ёмкости входит только ряд постоянного отношения α=2r;
        # опубликованная точка (64, 64) лежит вне ряда и в тренд не включается.
        if alpha == 2 * r:
            curve.append((r, m))

    # тренд: спадает ли Δ с ростом ёмкости
    if len(curve) >= 3:
        curve.sort()
        rs = np.log2([c[0] for c in curve])
        ds = np.array([c[1] for c in curve])
        sl, ic, rv, pv, se = stats.linregress(rs, ds)
        rho, prho = stats.spearmanr([c[0] for c in curve], ds)
        print(f"\n  Тренд Δ по log2(r): наклон {sl:+.4f} (p={pv:.4f}), "
              f"Spearman ρ={rho:+.3f} (p={prho:.4f})")
        out["trend"] = {"slope_per_log2r": round(float(sl), 4), "p_value": round(float(pv), 4),
                        "spearman_rho": round(float(rho), 3), "spearman_p": round(float(prho), 4),
                        "interpretation": ("отрицательный наклон = выигрыш претрейна убывает "
                                           "с ростом ёмкости адаптера")}

    if skipped:
        print(f"\n  ПРОПУЩЕНО (нет пары two-stage/dmid-only при теге "
              f"'{TAG or 'v1'}'): {', '.join(f'r{r} α{a}' for r, a in skipped)}")

    # Результаты по разным корпусам не должны затирать друг друга.
    out_path = OUT if not TAG else OUT.replace(".json", f"{TAG}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
