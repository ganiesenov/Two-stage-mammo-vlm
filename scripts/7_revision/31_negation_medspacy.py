"""
Внешняя валидация negation-правила Eq. 8 через medspaCy ConText (R1.1).

Запускается в отдельном окружении (conda env `medspacy`, python 3.11, medspacy 1.3.1),
чтобы не трогать зависимости rag_mammo:
    /home/gani/miniconda3/envs/medspacy/bin/python scripts/7_revision/31_negation_medspacy.py

Сравнение на двух уровнях, по всем референсам и генерациям моделей Table 6 (n=52; отдельно n=49):
  • упоминание — каждое вхождение термина находки: отрицается ли оно (наше окно 45
    символов против ConText с правилами по умолчанию, скоуп до конца предложения);
  • отчёт — утверждает ли отчёт хотя бы одну находку (то, что реально входит в Eq. 8).
Согласие — доля совпадений и κ Коэна; расхождения выписываются для ручного просмотра.
"""
import os, sys, re, json, importlib.util
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, load_preds

spec = importlib.util.spec_from_file_location(
    "clin", os.path.join(os.path.dirname(__file__), "04_clinical_metrics.py"))
clin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clin)

spec = importlib.util.spec_from_file_location(
    "sweep", os.path.join(os.path.dirname(__file__), "29_negation_window_sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

import medspacy
from medspacy.ner import TargetRule

MODELS = ["zero_shot", "dmid_only_r16", "two_stage_r16", "dmid_only_r64", "two_stage_r64"]
OUT = f"{ROOT}/results/revision/negation_medspacy.json"


def build_nlp():
    nlp = medspacy.load()
    tm = nlp.get_pipe("medspacy_target_matcher")
    tm.add([TargetRule(t, "FINDING", pattern=[{"LOWER": t}]) for t in clin.FINDING_TERMS])
    return nlp


def ours_mentions(text):
    t = text.lower()
    out = {}
    for term in clin.FINDING_TERMS:
        for m in re.finditer(rf'\b{re.escape(term)}\b', t):
            out[m.start()] = (clin.negated(t, m.start()), term)
    return out


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] / n * cb[l] / n for l in set(ca) | set(cb))
    return None if pe == 1 else (po - pe) / (1 - pe)


def main():
    nlp = build_nlp()
    texts = {}
    for m in MODELS:
        for p in load_preds(m, "test"):
            texts[("ref", p["id"])] = p["ref"]
            texts[(m, p["id"])] = p["hyp"]

    man = json.load(open(f"{ROOT}/results/revision/split_manifest.json"))
    leaked = set(man["verbatim_duplicate_leakage"]["test_ids"])
    mention_pairs, report_pairs, disagreements, report_dis = [], [], [], []
    clean_m, clean_r = [], []
    unmatched = 0
    for (src, pid), text in texts.items():
        ours = ours_mentions(text)
        doc = nlp(text)
        theirs = {e.start_char: bool(e._.is_negated) for e in doc.ents if e.label_ == "FINDING"}
        for pos, (neg, term) in ours.items():
            if pos not in theirs:
                unmatched += 1
                continue
            mention_pairs.append((neg, theirs[pos]))
            if pid not in leaked:
                clean_m.append((neg, theirs[pos]))
            if neg != theirs[pos]:
                t = text.replace("\n", " ")
                disagreements.append({"src": src, "id": pid, "leaked": pid in leaked,
                                      "term": term, "category": sweep.CATEGORY[term],
                                      "ours_negated": neg, "context_negated": theirs[pos],
                                      "snippet": t[max(0, pos - 90):pos + 30]})
        our_assert = any(not v for v, _ in ours.values())
        their_assert = any(not v for v in theirs.values())
        report_pairs.append((our_assert, their_assert))
        if our_assert != their_assert:
            # Категории, которые утверждает только одна из сторон: из-за них и разошлись
            ours_c = {sweep.CATEGORY[t] for v, t in ours.values() if not v}
            theirs_c = {sweep.CATEGORY[ours[p][1]] for p, v in theirs.items() if not v and p in ours}
            report_dis.append({"src": src, "id": pid, "leaked": pid in leaked,
                               "ours_asserts": our_assert, "context_asserts": their_assert,
                               "categories": sorted(ours_c ^ theirs_c)})
        if pid not in leaked:
            clean_r.append((our_assert, their_assert))

    def summ(pairs):
        return {"n": len(pairs),
                "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
                "kappa": round(kappa(pairs), 4),
                "table": {f"ours={a}/context={b}": c for (a, b), c in sorted(Counter(pairs).items())}}

    out = {"n_texts": len(texts), "unmatched_mentions": unmatched,
           "mention_level": summ(mention_pairs), "report_level": summ(report_pairs),
           "clean49": {"mention_level": summ(clean_m), "report_level": summ(clean_r)},
           "disagreements": disagreements, "report_disagreements": report_dis}
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"текстов {len(texts)}, упоминаний без пары {unmatched}")
    for lvl in ("mention_level", "report_level"):
        s = out[lvl]
        print(f"{lvl:14s} n={s['n']}  agreement={s['agreement']:.4f}  κ={s['kappa']:.4f}  {s['table']}")
    for lvl in ("mention_level", "report_level"):
        s = out["clean49"][lvl]
        print(f"clean49 {lvl:14s} n={s['n']}  agreement={s['agreement']:.4f}  κ={s['kappa']:.4f}")
    print(f"расхождений: {len(disagreements)}")


if __name__ == "__main__":
    main()
