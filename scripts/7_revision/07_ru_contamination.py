"""
Языковая контаминация Stage 1 и её влияние на кросс-языковой перенос (CPU).

Проблема. Из 200 синтетических отчётов VinDr 21 сгенерирован Llama-3.1-8B на РУССКОМ
языке при англоязычном промпте; все 21 прошли валидацию и вошли в корпус Stage 1
(21/390 = 5.4%). При этом finetune_dmid_ru.py:20 стартует от medgemma_dmid, то есть
от two-stage модели, чья родословная включает эти русские отчёты.

Следствие. Любое преимущество two-stage на dmid_ru становится неатрибутируемым:
это может быть эффект метода, а может быть просто знакомство с русской
маммографической лексикой. Контрольной ветки (dmid_ru от DMID-only) не существует.

Этот скрипт измеряет РАЗМЕР форы: насколько лексика 21 русского синтетического отчёта
пересекается с эталонными заключениями dmid_ru. Обучающего преимущества это не доказывает,
но задаёт верхнюю границу правдоподобия такого объяснения.

Помечено как EXPLORATORY: в PREREGISTRATION.md этой ветки нет.
"""
import os, sys, json, re
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

VINDR_JSONL = f"{ROOT}/vindr/synthetic_reports.jsonl"
RU_EVAL = f"{ROOT}/results/dmid_ru_eval.json"
OUT = f"{ROOT}/results/revision/ru_contamination.json"

CYR = re.compile(r'[а-яА-ЯёЁ]')
WORD = re.compile(r'[а-яё]+', re.IGNORECASE)


def tokens(t):
    return [w.lower() for w in WORD.findall(t) if len(w) > 3]


def main():
    ru_reports, en_reports = [], []
    for line in open(VINDR_JSONL, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("is_valid"):
            continue
        t = r.get("synthetic_report", "")
        (ru_reports if CYR.search(t) else en_reports).append(t)

    print("=" * 74)
    print("  Языковая контаминация корпуса Stage 1")
    print("=" * 74)
    print(f"  русскоязычных отчётов, прошедших валидацию: {len(ru_reports)}")
    print(f"  англоязычных: {len(en_reports)}")
    print(f"  доля русских в корпусе Stage 1 (390 пар): {len(ru_reports)/390*100:.1f}%")

    ru_vocab = Counter()
    for t in ru_reports:
        ru_vocab.update(set(tokens(t)))
    print(f"  уникальных русских словоформ (>3 букв) в контаминации: {len(ru_vocab)}")
    print(f"  самые частые: {[w for w,_ in ru_vocab.most_common(15)]}")

    if not os.path.exists(RU_EVAL):
        print(f"\n  {RU_EVAL} не найден — пропуск анализа пересечения")
        return

    data = json.load(open(RU_EVAL, encoding="utf-8"))
    samples = data.get("samples", [])
    print(f"\n  эталонных заключений dmid_ru (held-out): {len(samples)}")

    covered, totals, per_study = 0, 0, []
    ref_vocab = Counter()
    for s in samples:
        toks = tokens(s["reference"])
        ref_vocab.update(set(toks))
        if not toks:
            continue
        hit = sum(1 for w in toks if w in ru_vocab)
        covered += hit
        totals += len(toks)
        per_study.append({"id": s.get("patient_id"),
                          "coverage": round(hit / len(toks), 4)})

    type_overlap = set(ref_vocab) & set(ru_vocab)
    print("\n" + "=" * 74)
    print("  Пересечение лексики: контаминация Stage 1 ↔ эталоны dmid_ru")
    print("=" * 74)
    print(f"  покрытие по токенам : {covered}/{totals} = {covered/totals*100:.1f}%")
    print(f"  пересечение словарей: {len(type_overlap)} из {len(ref_vocab)} словоформ эталонов"
          f" ({len(type_overlap)/len(ref_vocab)*100:.1f}%)")
    print(f"  примеры общих терминов: {sorted(type_overlap, key=lambda w: -ref_vocab[w])[:20]}")

    print("\n  Как это читать:")
    print("  Пересечение измеряет ФОРУ, а не эффект обучения. Оно задаёт верхнюю границу")
    print("  правдоподобия объяснения «выигрыш two-stage на dmid_ru — это знакомство")
    print("  с языком, а не метод». Отделить одно от другого может только контрольная")
    print("  ветка: dmid_ru, обученный от DMID-only, которой сейчас не существует.")

    res = {
        "exploratory": True,
        "note": "не входит в PREREGISTRATION.md",
        "stage1_corpus": {
            "n_valid_total": 390,
            "n_russian": len(ru_reports),
            "share_russian": round(len(ru_reports) / 390, 4),
            "n_russian_vocab_types": len(ru_vocab),
            "top_terms": [w for w, _ in ru_vocab.most_common(30)],
        },
        "dmid_ru_overlap": {
            "n_reference_reports": len(samples),
            "token_coverage": round(covered / totals, 4) if totals else None,
            "type_overlap": len(type_overlap),
            "reference_vocab_size": len(ref_vocab),
            "type_overlap_share": round(len(type_overlap) / len(ref_vocab), 4) if ref_vocab else None,
            "shared_terms_sample": sorted(type_overlap, key=lambda w: -ref_vocab[w])[:40],
            "per_study": per_study,
        },
        "implication": ("finetune_dmid_ru.py стартует от medgemma_dmid (two-stage), поэтому "
                        "родословная русской модели включает 21 русский синтетический отчёт. "
                        "Требуется контрольная ветка от DMID-only, иначе кросс-языковой "
                        "перенос как вклад не атрибутируется."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n  → {OUT}")


if __name__ == "__main__":
    main()
