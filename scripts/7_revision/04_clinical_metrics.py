"""
Клинические метрики на всех 52 случаях (CPU, после 01_infer.py).

Отвечает на:
  R3.8 — density accuracy в Table 5 указана как «4/5» и «1/5», потому что считалась
         по пяти сохранённым качественным примерам, стоя рядом с колонками на n=52.
         Здесь она считается на всех случаях.
  R1.3 — «определение hallucination rate не имеет клинической строгости».
         Даём явные, проверяемые критерии и раскладку по типам, а не одну цифру.
  R1.6, R3.4 — раскладка согласия по BI-RADS: точное совпадение, ±1 категория,
         Cohen's κ, доля занижений и — клинически главное — сколько подозрительных
         случаев (BI-RADS 4-5) модель назвала доброкачественными.

Определение галлюцинации (формализовано для ревизии):
  случай считается галлюцинацией, если референс радиолога явно отрицает находки
  («no abnormal soft opacity», «no microcalcification», ...), а в сгенерированном
  отчёте утверждается наличие находки БЕЗ отрицания рядом. Проверка отрицания
  обязательна: старый код искал слово «mass» без учёта контекста, поэтому
  «no mass» засчитывался бы как галлюцинация.

Это автоматический прокси. Он не заменяет разметку радиологом, требуемую R1.3/R1.10,
а даёт воспроизводимую нижнюю границу и разметочный лист для читателей.
"""
import os, sys, re, json, csv, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR, extract_birads, extract_density

OUT_DIR = f"{ROOT}/results/revision"

FINDING_TERMS = ["mass", "masses", "opacity", "opacities", "calcification",
                 "calcifications", "microcalcification", "microcalcifications",
                 "lesion", "lesions", "asymmetry", "distortion", "nodule"]
NEG_CUES = ["no ", "not ", "without ", "free of ", "absence of ", "negative for "]
NEG_WINDOW = 45          # символов слева, где ищем отрицание
BENIGN = {1, 2}
SUSPICIOUS = {4, 5, 6}


def negated(text, pos):
    return any(c in text[max(0, pos - NEG_WINDOW):pos] for c in NEG_CUES)


def asserts_finding(text):
    """Есть ли в тексте УТВЕРЖДЕНИЕ о находке (без отрицания рядом)."""
    t = text.lower()
    for term in FINDING_TERMS:
        for m in re.finditer(rf'\b{re.escape(term)}\b', t):
            if not negated(t, m.start()):
                return True, term
    return False, None


def cohens_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    labels = sorted({v for p in pairs for v in p})
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (None if pe == 1 else (po - pe) / (1 - pe)), n


def analyse(preds):
    res = {"n": len(preds)}
    rb = [extract_birads(p["ref"]) for p in preds]
    hb = [extract_birads(p["hyp"]) for p in preds]
    rd = [extract_density(p["ref"]) for p in preds]
    hd = [extract_density(p["hyp"]) for p in preds]

    # ── BI-RADS ──
    both = [(r, h) for r, h in zip(rb, hb) if r is not None and h is not None]
    res["birads"] = {
        "n_ref_parsed": sum(v is not None for v in rb),
        "n_hyp_parsed": sum(v is not None for v in hb),
        "n_comparable": len(both),
        "exact": round(sum(r == h for r, h in both) / len(both), 4) if both else None,
        "within_1": round(sum(abs(r - h) <= 1 for r, h in both) / len(both), 4) if both else None,
        "under_call": round(sum(h < r for r, h in both) / len(both), 4) if both else None,
        "over_call": round(sum(h > r for r, h in both) / len(both), 4) if both else None,
    }
    k, n = cohens_kappa(rb, hb)
    res["birads"]["cohens_kappa"] = round(k, 4) if k is not None else None

    conf = Counter((r, h) for r, h in both)
    res["birads"]["confusion"] = {f"{r}->{h}": c for (r, h), c in sorted(conf.items())}

    susp = [(r, h) for r, h in both if r in SUSPICIOUS]
    res["birads"]["suspicious_cases"] = {
        "n": len(susp),
        "downgraded_to_benign": sum(h in BENIGN for _, h in susp),
        "note": "клинически самая опасная ошибка: подозрительное названо доброкачественным",
    }

    # ── ACR density ──
    bothd = [(r, h) for r, h in zip(rd, hd) if r is not None and h is not None]
    kd, _ = cohens_kappa(rd, hd)
    res["density"] = {
        "n_comparable": len(bothd),
        "accuracy": round(sum(r == h for r, h in bothd) / len(bothd), 4) if bothd else None,
        "cohens_kappa": round(kd, 4) if kd is not None else None,
        "pred_distribution": dict(Counter(h for _, h in bothd)),
        "ref_distribution": dict(Counter(r for r, _ in bothd)),
    }

    # ── находки: галлюцинации И пропуски ──
    # Исходная работа мерила только «галлюцинации», причём определением, которое
    # засчитывало любое расхождение словаря (см. legacy ниже). Клинически более
    # опасная ошибка — ПРОПУСК находки — не измерялась вообще.
    tp = fp = fn = tn = 0
    hall_cases, miss_cases = [], []
    for p in preds:
        ref_a, _ = asserts_finding(p["ref"])
        hyp_a, term = asserts_finding(p["hyp"])
        if ref_a and hyp_a:
            tp += 1
        elif not ref_a and hyp_a:
            fp += 1
            hall_cases.append({"id": p["id"], "term": term})
        elif ref_a and not hyp_a:
            fn += 1
            miss_cases.append({"id": p["id"]})
        else:
            tn += 1

    res["findings"] = {
        "definition": ("утверждение о находке = термин находки без отрицания в окне 45 символов; "
                       "галлюцинация = референс не утверждает, генерация утверждает; "
                       "пропуск = референс утверждает, генерация нет"),
        "n_ref_positive": tp + fn,
        "n_hyp_positive": tp + fp,
        "hallucination_rate": round(fp / len(preds), 4),
        "omission_rate": round(fn / (tp + fn), 4) if (tp + fn) else None,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "hallucination_cases": hall_cases,
        "omission_cases": miss_cases,
    }

    # Метрика исходной работы, воспроизведена для сопоставимости: засчитывает
    # находку как выдуманную, если её термин не встречается в референсе ДОСЛОВНО.
    # Это детектор расхождения словаря, а не галлюцинаций: референс «opacity»
    # против генерации «mass» помечается как галлюцинация, хотя обе описывают находку.
    legacy = 0
    for p in preds:
        g, r = p["hyp"].lower(), p["ref"].lower()
        for f in ['mass', 'calcification', 'architectural distortion', 'asymmetry']:
            if f in g and f not in r and f'no {f}' not in r:
                legacy += 1
                break
    res["hallucination_legacy"] = {
        "definition": "как в evaluate_dmid_full.py: термин есть в генерации и отсутствует в референсе дословно",
        "rate": round(legacy / len(preds), 4),
        "n": legacy,
        "caveat": "меряет расхождение лексики, а не наличие выдуманных находок",
    }

    # ── структура и длина ──
    res["structure"] = {
        "has_birads": round(sum(v is not None for v in hb) / len(preds), 4),
        "has_density": round(sum(v is not None for v in hd) / len(preds), 4),
        "mean_words_hyp": round(sum(len(p["hyp"].split()) for p in preds) / len(preds), 1),
        "mean_words_ref": round(sum(len(p["ref"].split()) for p in preds) / len(preds), 1),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    keep, tag = None, ""
    if args.clean_only:
        man = json.load(open(f"{OUT_DIR}/split_manifest.json"))
        keep, tag = set(man["clean_test_ids"]), "_clean"

    out = {}
    for m in args.models:
        path = f"{PRED_DIR}/{m}__{args.split}.json"
        if not os.path.exists(path):
            print(f"  пропуск {m} — нет предсказаний")
            continue
        preds = json.load(open(path, encoding="utf-8"))["predictions"]
        if keep:
            preds = [p for p in preds if p["id"] in keep]
        r = analyse(preds)
        out[m] = r

        b, d, h = r["birads"], r["density"], r["findings"]
        print(f"\n{'='*74}\n  {m}  (n={r['n']})\n{'='*74}")
        print(f"  BI-RADS  exact {b['exact']}  ±1 {b['within_1']}  κ {b['cohens_kappa']}"
              f"  (сравнимо {b['n_comparable']})")
        print(f"           занижений {b['under_call']}  завышений {b['over_call']}")
        s = b["suspicious_cases"]
        print(f"           подозрительных {s['n']}, из них названо доброкачественными: "
              f"{s['downgraded_to_benign']}")
        print(f"  ACR      accuracy {d['accuracy']}  κ {d['cohens_kappa']}  "
              f"(сравнимо {d['n_comparable']})")
        print(f"           предсказано {d['pred_distribution']} / эталон {d['ref_distribution']}")
        print(f"  Находки  чувствительность {h['sensitivity']}  пропусков {h['omission_rate']}"
              f"  ({h['confusion']['fn']}/{h['n_ref_positive']} находок упущено)")
        print(f"           галлюцинаций {h['hallucination_rate']} ({h['confusion']['fp']}/{r['n']})"
              f"   [метрика статьи: {r['hallucination_legacy']['rate']}]")
        print(f"  Длина    {r['structure']['mean_words_hyp']} слов "
              f"(эталон {r['structure']['mean_words_ref']})")

    os.makedirs(OUT_DIR, exist_ok=True)
    p = f"{OUT_DIR}/clinical_metrics_{args.split}{tag}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {p}")

    # разметочный лист для радиологов (R1.3, R1.10)
    if not args.clean_only and args.models:
        m0 = args.models[0]
        src = f"{PRED_DIR}/{m0}__{args.split}.json"
        if os.path.exists(src):
            preds = json.load(open(src, encoding="utf-8"))["predictions"]
            sheet = f"{OUT_DIR}/reader_study_sheet.csv"
            with open(sheet, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "reference_report", "generated_report",
                            "density_correct", "findings_correct", "birads_correct",
                            "recommendation_correct", "hallucination_present",
                            "clinically_acceptable", "reader_comment"])
                for p_ in preds:
                    w.writerow([p_["id"], p_["ref"], p_["hyp"], "", "", "", "", "", "", ""])
            print(f"  → {sheet}  (пустой лист для разметки радиологами)")


if __name__ == "__main__":
    main()
