"""
Устойчивость negation-правила к ширине окна (R1.1, CPU).

Окно отрицания в Eq. 8 — 45 символов слева от термина. Пересчитываем чувствительность,
пропуски и галлюцинации моделей Table 6 при окнах 30/45/60/90 на тех же генерациях.
Вывод Table 6 («модели не выдумывают находки, они их опускают») устойчив, если при
любом окне галлюцинации остаются нулевыми, а порядок моделей по пропускам не меняется.
"""
import os, sys, json, importlib.util

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, load_preds

spec = importlib.util.spec_from_file_location(
    "clin", os.path.join(os.path.dirname(__file__), "04_clinical_metrics.py"))
clin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clin)

MODELS = ["zero_shot", "dmid_only_r16", "two_stage_r16", "dmid_only_r64", "two_stage_r64"]
WINDOWS = [30, 45, 60, 90]
OUT = f"{ROOT}/results/revision/negation_window_sweep.json"


def main():
    man = json.load(open(f"{ROOT}/results/revision/split_manifest.json"))
    leaked = set(man["verbatim_duplicate_leakage"]["test_ids"])
    out = {}
    for split, drop in (("clean49", leaked), ("full52", set())):
        out[split] = {}
        for m in MODELS:
            preds = [p for p in load_preds(m, "test") if p["id"] not in drop]
            out[split][m] = {}
            for w in WINDOWS:
                clin.NEG_WINDOW = w
                f = clin.analyse(preds)["findings"]
                out[split][m][w] = {k: f[k] for k in
                                    ("n_ref_positive", "n_hyp_positive", "sensitivity",
                                     "omission_rate", "hallucination_rate", "confusion")}
    clin.NEG_WINDOW = 45
    json.dump(out, open(OUT, "w"), indent=2)

    for split in out:
        print(f"\n== {split} ==   ref+ / sens / omis / halluc по окнам {WINDOWS}")
        for m in MODELS:
            cells = "  ".join(f"{r['n_ref_positive']:2d}/{r['sensitivity']:.3f}/{r['omission_rate']:.3f}/{r['hallucination_rate']:.3f}"
                              for r in out[split][m].values())
            print(f"  {m:15s} {cells}")


if __name__ == "__main__":
    main()
