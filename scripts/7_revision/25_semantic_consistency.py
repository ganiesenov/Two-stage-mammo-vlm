"""
Семантическая согласованность синтетического корпуса Stage 1 (CPU).

Валидатор Eq. (1) чисто лексический: он проверяет наличие категории BI-RADS,
рекомендации и минимальной длины, но не сверяет содержание отчёта с исходной
разметкой. Здесь такая сверка делается отдельной проверкой: совпадает ли
заявленная в отчёте категория BI-RADS с аннотацией и названа ли та сторона,
для которой отчёт сгенерирован.

Проверка post-hoc и на корпус не влияет: отчёты не отбраковывались по её итогу.
Считается по сохранённым корпусам data/synthetic_v1_{vindr,cbis}.jsonl,
только по записям is_valid=True (это и есть 390 отчётов Stage 1).
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

OUT = f"{ROOT}/results/revision/semantic_consistency.json"

# Латиница и кириллица: 21 отчёт VinDr сгенерирован по-русски, исключать их
# из проверки нельзя — они входили в обучающий корпус наравне с остальными.
LEFT = ["left breast", "left mammogram", "the left", "левой", "левая", "слева"]
RIGHT = ["right breast", "right mammogram", "the right", "правой", "правая", "справа"]


def birads_in(text):
    """Первая категория BI-RADS в тексте.

    Окно до 40 символов без цифр: генератор пишет и «BI-RADS 4», и
    «BI-RADS category is 4», и «BI-RADS Category: **4**», и по-русски
    «расценивается как BI-RADS 4».
    """
    m = re.search(r'BI-?RADS[^0-9]{0,40}?([0-6])\b', text, re.I)
    return int(m.group(1)) if m else None


def side_in(text):
    t = text.lower()
    l = any(k in t for k in LEFT)
    r = any(k in t for k in RIGHT)
    if l and not r:
        return "L"
    if r and not l:
        return "R"
    return "both" if (l and r) else None


def check(records, get_bir, get_side, tag):
    n = bir_ok = bir_cmp = side_ok = side_cmp = 0
    both = 0
    mism_b, mism_s = [], []
    for r in records:
        if not r.get("is_valid"):
            continue
        n += 1
        txt = r["synthetic_report"]
        b_ref, b_hyp = get_bir(r), birads_in(txt)
        if b_ref is not None and b_hyp is not None:
            bir_cmp += 1
            if b_ref == b_hyp:
                bir_ok += 1
            else:
                mism_b.append({"ref": b_ref, "hyp": b_hyp})
        s_ref, s_hyp = get_side(r), side_in(txt)
        if s_ref and s_hyp:
            if s_hyp == "both":
                both += 1
            else:
                side_cmp += 1
                if s_ref == s_hyp:
                    side_ok += 1
                else:
                    mism_s.append({"ref": s_ref, "hyp": s_hyp})
    res = {"source": tag, "n_valid": n,
           "birads": {"n_comparable": bir_cmp, "n_agree": bir_ok,
                      "agreement": round(bir_ok / bir_cmp, 4) if bir_cmp else None,
                      "n_mismatch": len(mism_b)},
           "laterality": {"n_comparable": side_cmp, "n_agree": side_ok,
                          "agreement": round(side_ok / side_cmp, 4) if side_cmp else None,
                          "n_mismatch": len(mism_s),
                          "n_both_sides_mentioned": both}}
    print(f"  {tag:<12s} валидных {n:>4d}   BI-RADS {res['birads']['n_agree']}/"
          f"{res['birads']['n_comparable']} ({res['birads']['agreement']})   "
          f"сторона {res['laterality']['n_agree']}/{res['laterality']['n_comparable']} "
          f"({res['laterality']['agreement']}), обе стороны названы в {both}")
    return res


def main():
    vindr = [json.loads(l) for l in open(f"{ROOT}/data/synthetic_v1_vindr.jsonl")]
    cbis = [json.loads(l) for l in open(f"{ROOT}/data/synthetic_v1_cbis.jsonl")]

    rv = check(vindr,
               lambda r: birads_in(str(r.get("breast_birads", ""))),
               lambda r: str(r.get("laterality", ""))[:1].upper() or None,
               "VinDr-Mammo")
    rc = check(cbis,
               lambda r: int(r["assessment"]) if str(r.get("assessment", "")).isdigit() else None,
               lambda r: str(r.get("laterality", ""))[:1].upper() or None,
               "CBIS-DDSM")

    tot_b_ok = rv["birads"]["n_agree"] + rc["birads"]["n_agree"]
    tot_b_n = rv["birads"]["n_comparable"] + rc["birads"]["n_comparable"]
    tot_s_ok = rv["laterality"]["n_agree"] + rc["laterality"]["n_agree"]
    tot_s_n = rv["laterality"]["n_comparable"] + rc["laterality"]["n_comparable"]

    out = {"corpus": "Stage 1 v1 (390 retained reports)",
           "note": ("post-hoc проверка, отчёты по её итогу не отбраковывались; "
                    "лексический валидатор Eq. (1) её не содержит"),
           "by_source": [rv, rc],
           "total": {
               "birads_agreement": round(tot_b_ok / tot_b_n, 4),
               "birads_n": tot_b_n, "birads_agree": tot_b_ok,
               "laterality_agreement": round(tot_s_ok / tot_s_n, 4),
               "laterality_n": tot_s_n, "laterality_agree": tot_s_ok}}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    t = out["total"]
    print(f"\n  ИТОГО  BI-RADS {t['birads_agree']}/{t['birads_n']} = {t['birads_agreement']}"
          f"   сторона {t['laterality_agree']}/{t['laterality_n']} = {t['laterality_agreement']}")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
