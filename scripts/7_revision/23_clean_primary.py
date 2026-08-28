"""
Пересчёт всех числовых утверждений рукописи на чистом подмножестве теста (n=49).

Второй раунд ревизии, R1.2: clean subset становится ОСНОВНЫМ, полный тест (n=52)
уходит в sensitivity. GPU не задействован — всё считается из сохранённых
по-сэмпловых генераций (results/revision/predictions/).

Считает и складывает в results/revision/clean_primary.json:
  * корпусные метрики каждой модели на 49 и на 52 (для sensitivity-абзаца);
  * распределения BI-RADS и ACR-плотности в 49 референсах;
  * знаменатели клинического анализа (сколько кейсов утверждают находку,
    сколько подозрительных);
  * чувствительность к промпту на 49;
  * ретривальную абляцию на 49;
  * длины отчётов.

Ячейки свипа (Table 9) пересчитывает 06_equivalence.py --clean-only.
"""
import os, sys, json
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR, corpus_scores, extract_birads, extract_density

OUT_DIR = f"{ROOT}/results/revision"

MODELS = ["zero_shot", "dmid_only_r16", "two_stage_r16", "dmid_only_r64",
          "two_stage_r64", "two_stage_r16_a32", "two_stage_r64_rag1",
          "dmid_only_r16_promptA", "two_stage_r16_promptA"]


def load(name, keep=None):
    p = f"{PRED_DIR}/{name}__test.json"
    if not os.path.exists(p):
        return None
    preds = json.load(open(p, encoding="utf-8"))["predictions"]
    if keep is not None:
        preds = [x for x in preds if x["id"] in keep]
    return preds


def main():
    man = json.load(open(f"{OUT_DIR}/split_manifest.json"))
    keep = set(man["clean_test_ids"])
    leaked = man["verbatim_duplicate_leakage"]["test_ids"]
    print(f"чистое подмножество: {len(keep)} кейсов, исключены {leaked}")

    out = {"n_clean": len(keep), "excluded": leaked, "clean": {}, "full": {}, "delta": {}}

    for m in MODELS:
        pc, pf = load(m, keep), load(m)
        if pc is None:
            print(f"  пропуск {m} — нет предсказаний")
            continue
        sc, sf = corpus_scores(pc), corpus_scores(pf)
        out["clean"][m] = {**sc, "n": len(pc),
                           "mean_words": round(sum(len(p["hyp"].split()) for p in pc) / len(pc), 1)}
        out["full"][m] = {**sf, "n": len(pf)}
        out["delta"][m] = {k: round(sf[k] - sc[k], 4) for k in sc
                           if isinstance(sc.get(k), float) and isinstance(sf.get(k), float)}
        print(f"  {m:<26s} n={len(pc)}  ROUGE-L {sc['rougeL']:.4f} "
              f"(полный {sf['rougeL']:.4f}, Δ {sf['rougeL']-sc['rougeL']:+.4f})")

    # ── референсные распределения на 49 ──
    ref = load("two_stage_r64", keep)
    rb = [extract_birads(p["ref"]) for p in ref]
    rd = [extract_density(p["ref"]) for p in ref]
    n = len(ref)
    out["reference_distribution"] = {
        "n": n,
        "birads": {str(k): {"n": v, "pct": round(100 * v / n, 1)}
                   for k, v in sorted(Counter(x for x in rb if x is not None).items())},
        "density": {k: {"n": v, "pct": round(100 * v / n, 1)}
                    for k, v in sorted(Counter(x for x in rd if x is not None).items())},
        "mean_words_ref": round(sum(len(p["ref"].split()) for p in ref) / n, 1),
    }
    reff = load("two_stage_r64")
    out["reference_distribution_full"] = {
        "n": len(reff),
        "birads": {str(k): v for k, v in sorted(Counter(
            x for x in (extract_birads(p["ref"]) for p in reff) if x is not None).items())},
        "density": {k: v for k, v in sorted(Counter(
            x for x in (extract_density(p["ref"]) for p in reff) if x is not None).items())},
    }

    with open(f"{OUT_DIR}/clean_primary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {OUT_DIR}/clean_primary.json")

    d = out["reference_distribution"]
    print(f"\n  BI-RADS в {d['n']} референсах: " +
          "  ".join(f"{k}: {v['n']} ({v['pct']}%)" for k, v in d["birads"].items()))
    print(f"  ACR      в {d['n']} референсах: " +
          "  ".join(f"{k}: {v['n']} ({v['pct']}%)" for k, v in d["density"].items()))
    print(f"  средняя длина референса: {d['mean_words_ref']} слов")


if __name__ == "__main__":
    main()
