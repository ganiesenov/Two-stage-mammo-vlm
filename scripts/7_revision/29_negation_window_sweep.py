"""
Устойчивость negation-правила к ширине окна (R1.1, CPU).

Окно отрицания в Eq. 8 — 45 символов слева от термина. Пересчитываем чувствительность,
пропуски и галлюцинации моделей Table 6 при окнах 30/45/60/90 на тех же генерациях.
Дополнительно — уровень категорий находок (масса, кальцинат, асимметрия, дисторсия),
который Eq. 8 не измеряет: называет ли модель не тот тип находки.
Вывод Table 6 («модели не выдумывают находки, они их опускают») устойчив, если при
любом окне галлюцинации остаются нулевыми, а порядок моделей по пропускам не меняется.
"""
import os, sys, re, json, importlib.util

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, load_preds

spec = importlib.util.spec_from_file_location(
    "clin", os.path.join(os.path.dirname(__file__), "04_clinical_metrics.py"))
clin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clin)

# Категории находок для проверки на уровне типа (не входит в Eq. 8): Eq. 8 считает по
# кейсам — утверждает ли текст хоть одну находку; здесь — совпадает ли ТИП находки.
CATEGORY = {"mass": "mass", "masses": "mass", "opacity": "mass", "opacities": "mass",
            "nodule": "mass", "lesion": "mass", "lesions": "mass",
            "calcification": "calc", "calcifications": "calc",
            "microcalcification": "calc", "microcalcifications": "calc",
            "asymmetry": "asymmetry", "distortion": "distortion"}

MODELS = ["zero_shot", "dmid_only_r16", "two_stage_r16", "dmid_only_r64", "two_stage_r64"]
WINDOWS = [30, 45, 60, 90]
OUT = f"{ROOT}/results/revision/negation_window_sweep.json"


def asserted_categories(text):
    t = text.lower()
    out = set()
    for term in clin.FINDING_TERMS:
        for m in re.finditer(rf'\b{re.escape(term)}\b', t):
            if not clin.negated(t, m.start()):
                out.add(CATEGORY[term])
    return out


def category_level(preds):
    """Доли на уровне категорий: сколько утверждённых генерацией категорий нет в
    референсе, и сколько категорий референса генерация не утверждает."""
    extra = n_hyp = missing = n_ref = cases_extra = cases_extra_on_normal = 0
    for p in preds:
        a_r, a_h = asserted_categories(p["ref"]), asserted_categories(p["hyp"])
        extra += len(a_h - a_r); n_hyp += len(a_h)
        missing += len(a_r - a_h); n_ref += len(a_r)
        cases_extra += bool(a_h - a_r)
        cases_extra_on_normal += bool(a_h) and not a_r
    return {"extra_categories": extra, "hyp_categories": n_hyp,
            "extra_rate": round(extra / n_hyp, 4) if n_hyp else None,
            "missing_categories": missing, "ref_categories": n_ref,
            "missing_rate": round(missing / n_ref, 4) if n_ref else None,
            "cases_with_extra_category": cases_extra,
            "cases_with_finding_on_normal_reference": cases_extra_on_normal}


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
    out["category_level"] = {}
    for split, drop in (("clean49", leaked), ("full52", set())):
        out["category_level"][split] = {
            m: category_level([p for p in load_preds(m, "test") if p["id"] not in drop])
            for m in MODELS}
    json.dump(out, open(OUT, "w"), indent=2)

    for split in ("clean49", "full52"):
        print(f"\n== {split} ==   ref+ / sens / omis / halluc по окнам {WINDOWS}")
        for m in MODELS:
            cells = "  ".join(f"{r['n_ref_positive']:2d}/{r['sensitivity']:.3f}/{r['omission_rate']:.3f}/{r['hallucination_rate']:.3f}"
                              for r in out[split][m].values())
            print(f"  {m:15s} {cells}")
    print("\n== категории, clean49 (окно 45) ==")
    for m, r in out["category_level"]["clean49"].items():
        print(f"  {m:15s} лишние {r['extra_categories']}/{r['hyp_categories']}={r['extra_rate']}  "
              f"недостающие {r['missing_categories']}/{r['ref_categories']}={r['missing_rate']}  "
              f"кейсов с лишней {r['cases_with_extra_category']}, из них на норме {r['cases_with_finding_on_normal_reference']}")


if __name__ == "__main__":
    main()
