"""
DMID-RU: внешняя валидация (R1.2) и причинный тест языковой контаминации (GPU, короткий).

Закрывает два разных вопроса, которые в предыдущей версии ответа были смешаны:

1. R1.2 — ВНЕШНЯЯ ВАЛИДАЦИЯ. Модель, обученная на DMID, применяется к DMID-RU
   БЕЗ дообучения. finetune_dmid_ru.py дообучает на 172 исследованиях, то есть выполняет
   доменную адаптацию, а не внешнюю валидацию, и на требование рецензента не отвечает.

2. Причинный тест контаминации. Лексическое пересечение (28.3%) НЕ показывает, что модель
   взяла русский регистр из 21 контаминированного отчёта: этот регистр есть в предобучении
   Gemma-3 сам по себе. Дешёвый причинный тест — базовая MedGemma-4B zero-shot на DMID-RU.
   Если она и так выдаёт русские маммографические заключения, вклад контаминации близок
   к нулю и тревога снимается без единого прогона обучения.

Ветви (все — только инференс, дообучения нет):
  base          — MedGemma-4B без адаптеров           → причинный тест
  two_stage     — medgemma_dmid (Stage 1 + Stage 2)   → внешняя валидация R1.2
  dmid_only     — medgemma_dmid_only (только Stage 2) → контроль: та же модель без Stage 1

Сравнение two_stage против dmid_only здесь ЗАКОННО и контроля не требует: обе не видели
ни одного русского примера из DMID-RU, а различаются ровно наличием Stage 1. Именно это
и позволяет отделить эффект метода от эффекта знакомства с языком.

Источник данных. Диск D с полным датасетом недоступен (жёсткий диск не установлен),
но весь held-out целиком есть локально:
  изображения — validation_app/images/P##/P##_{RCC,LCC,RMLO,LMLO}.png
                22 исследования × 4 проекции = 88 файлов, ID совпадают со split_ru.json["test"];
                это прямые копии датасета (1024 px по длинной стороне, grayscale),
                build_validation_app.py их не пережимает;
  эталоны     — results/dmid_ru_eval.json (22 записи patient_id + reference).
Обучающая часть dmid_ru локально недоступна, поэтому здесь возможны только прогоны
БЕЗ дообучения — что для обеих задач этого скрипта как раз и требуется.
"""
import os, sys, json, re, argparse

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, MODEL_ID, _bnb, free, corpus_scores, extract_birads

DATA_DIR = "/mnt/d/dmid_ru"
PAIRS = f"{DATA_DIR}/pairs_study.json"
IMG_LOCAL = f"{ROOT}/validation_app/images"
REF_LOCAL = f"{ROOT}/results/dmid_ru_eval.json"
SPLIT = f"{ROOT}/scripts/3_finetune/split_ru.json"
# Порядок проекций по DATASET_DMID_RU.md; ни одна из трёх ветвей не обучалась
# на русском мульти-вью входе, поэтому порядок влияет лишь на подписи в промпте.
VIEW_ORDER = ["RCC", "LCC", "RMLO", "LMLO"]
OUT_DIR = f"{ROOT}/results/revision"
PRED_DIR = f"{OUT_DIR}/predictions"

VIEW_RU = {"RCC": "Правая КК", "LCC": "Левая КК", "RMLO": "Правая МЛК", "LMLO": "Левая МЛК"}
INSTRUCTION = ("Составь структурированное заключение по маммографии обеих молочных желёз "
               "на русском языке: рентгенологический тип плотности по ACR, описание (протокол), "
               "заключение, категорию BI-RADS и рекомендации.")

ARMS = {
    "base":      {"adapter": None},
    "two_stage": {"adapter": f"{ROOT}/medgemma_dmid"},
    "dmid_only": {"adapter": f"{ROOT}/medgemma_dmid_only"},
}

CYR = re.compile(r'[а-яА-ЯёЁ]')
RU_TERMS = ["плотност", "молочн", "желез", "birads", "bi-rads", "категор",
            "заключен", "рекоменд", "проекц", "образован", "ткан"]


def load_test_pairs():
    """Держит два источника: полный датасет с диска D, если он есть, иначе локальные копии."""
    test_ids = json.load(open(SPLIT, encoding="utf-8"))["test"]

    if os.path.exists(PAIRS):
        pairs = json.load(open(PAIRS, encoding="utf-8"))
        sel = [p for p in pairs if p["patient_id"] in set(test_ids)]
        for p in sel:
            p["_img_paths"] = [os.path.join(DATA_DIR, rel) for rel in p["images"]]
        print(f"Источник: полный датасет {DATA_DIR}")
        return sel

    refs = {s["patient_id"]: s["reference"]
            for s in json.load(open(REF_LOCAL, encoding="utf-8"))["samples"]}
    pairs = []
    missing = []
    for pid in test_ids:
        paths, views = [], []
        for v in VIEW_ORDER:
            p = f"{IMG_LOCAL}/{pid}/{pid}_{v}.png"
            if os.path.exists(p):
                paths.append(p)
                views.append(v)
        if len(paths) != 4 or pid not in refs:
            missing.append(pid)
            continue
        pairs.append({"patient_id": pid, "_img_paths": paths,
                      "views": views, "report": refs[pid]})
    print(f"Источник: локальные копии (validation_app/images + dmid_ru_eval.json)")
    if missing:
        print(f"  ПРОПУЩЕНЫ (неполные данные): {missing}")
    return pairs


def build_messages(pair):
    content = []
    for view in pair["views"]:
        content.append({"type": "text", "text": f"{VIEW_RU.get(view, view)} проекция:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


def russian_profile(texts):
    """Насколько вывод вообще является русским маммографическим заключением."""
    n = len(texts)
    cyr = sum(1 for t in texts if CYR.search(t))
    def frac_cyr(t):
        letters = [c for c in t if c.isalpha()]
        return sum(1 for c in letters if CYR.match(c)) / len(letters) if letters else 0.0
    heavy = sum(1 for t in texts if frac_cyr(t) > 0.5)
    term_hits = [sum(1 for k in RU_TERMS if k in t.lower()) for t in texts]
    return {
        "any_cyrillic": round(cyr / n, 4),
        "majority_cyrillic": round(heavy / n, 4),
        "mean_domain_terms": round(sum(term_hits) / n, 2),
        "has_birads": round(sum(1 for t in texts if extract_birads(t) is not None) / n, 4),
        "mean_words": round(sum(len(t.split()) for t in texts) / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()

    if not os.path.exists(PAIRS) and not os.path.isdir(IMG_LOCAL):
        sys.exit(f"Нет ни {PAIRS}, ни локальных копий в {IMG_LOCAL}")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    pairs = load_test_pairs()
    print(f"DMID-RU held-out: {len(pairs)} исследований, по 4 проекции")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    os.makedirs(PRED_DIR, exist_ok=True)
    summary = {}

    for arm in args.arms:
        spec = ARMS[arm]
        if spec["adapter"] and not os.path.exists(spec["adapter"]):
            print(f"[{arm}] пропуск — нет {spec['adapter']}")
            continue
        print(f"\n{'='*70}\n  {arm} → DMID-RU (БЕЗ дообучения)\n{'='*70}", flush=True)

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=_bnb(), device_map="auto", dtype=torch.bfloat16)
        model.config.use_cache = True
        if spec["adapter"]:
            model = PeftModel.from_pretrained(model, spec["adapter"])
        model.eval()

        preds = []
        for i, p in enumerate(pairs):
            imgs = []
            for path in p["_img_paths"]:
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (896, 896), 128))
            prompt = processor.apply_chat_template(
                build_messages(p), add_generation_prompt=True, tokenize=False)
            enc = processor(text=prompt, images=imgs, return_tensors="pt",
                            truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=300, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            hyp = processor.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            preds.append({"id": p["patient_id"], "ref": p["report"].strip(), "hyp": hyp})
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(pairs)}", flush=True)

        sc = corpus_scores(preds)
        prof = russian_profile([x["hyp"] for x in preds])
        with open(f"{PRED_DIR}/ru_external_{arm}__test.json", "w", encoding="utf-8") as f:
            json.dump({"model": arm, "split": "dmid_ru_test", "finetuned_on_ru": False,
                       "corpus_scores": sc, "russian_profile": prof,
                       "predictions": preds}, f, ensure_ascii=False, indent=2)
        summary[arm] = {"scores": sc, "russian_profile": prof}
        print(f"  метрики: {sc}")
        print(f"  русскоязычность: {prof}")
        free(model)

    with open(f"{OUT_DIR}/dmid_ru_external.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\n  ИТОГ\n{'='*70}")
    print(f"  {'ветвь':<12s} {'R-L':>7s} {'BERTScore-нет':>6s}  {'кириллица':>10s} "
          f"{'термины':>8s} {'BI-RADS':>8s}")
    for k, v in summary.items():
        p, s = v["russian_profile"], v["scores"]
        print(f"  {k:<12s} {s['rougeL']:>7.4f} {'':>6s}  {p['majority_cyrillic']:>10.2f} "
              f"{p['mean_domain_terms']:>8.2f} {p['has_birads']:>8.2f}")

    if "base" in summary:
        p = summary["base"]["russian_profile"]
        print("\n  Причинный тест контаминации:")
        if p["majority_cyrillic"] > 0.5 and p["mean_domain_terms"] >= 3:
            print("  Базовая модель БЕЗ Stage 1 уже выдаёт русский маммографический регистр →")
            print("  вклад 21 контаминированного отчёта в языковую способность близок к нулю,")
            print("  тревога о языковой форе снимается.")
        else:
            print("  Базовая модель русского регистра НЕ выдаёт → фора может быть реальна;")
            print("  её величину измеряет ветка --clean-corpus, а не словарное пересечение.")

    print(f"\n  → {OUT_DIR}/dmid_ru_external.json")


if __name__ == "__main__":
    main()
