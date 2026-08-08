"""
Общий код для прогонов ревизии Scientific Reports.

Ключевое отличие от исходных скриптов: генерации СОХРАНЯЮТСЯ по-сэмплово
(это блокировало доверительные интервалы, пересчёт density/hallucination
и вообще любую перепроверку — см. REVISION_PLAN.md, п. 1.7).

Все прогоны идут в одинаковых условиях (4-bit NF4, 448x448, greedy, 200 новых
токенов), чтобы строки итоговых таблиц были сопоставимы между собой.
"""
import os, json, re, gc

# torch/PIL импортируются лениво: CPU-скрипты (манифест сплита, bootstrap,
# клинические метрики) должны работать и без GPU-окружения.

ROOT       = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article"
MODEL_ID   = "google/medgemma-4b-it"
STAGE1     = f"{ROOT}/medgemma_multimodal"
IMGS_DIR   = f"{ROOT}/dmid/TIFF Images/TIFF Images/"
REPS_DIR   = f"{ROOT}/dmid/Reports/Reports/"
PRED_DIR   = f"{ROOT}/results/revision/predictions"

# В проекте использовались ДВА разных обучающих промпта, и это спутывает опорное
# сравнение (см. REVISION_PLAN.md). Оба зафиксированы здесь дословно:
#   A — finetune_dmid_only.py:52 и finetune_dmid.py:49 (опубликованные модели r=16)
#   B — lora_ablation.py:26 и phase2_full.py:27 (конфигурации Table 8 и строки Table 4)
# Модели r=16 обучались промптом A, но оценивались в Table 4 промптом B.
PROMPTS = {
    "A": ("Generate a structured mammography report with breast composition, findings, "
          "BI-RADS category and recommendation."),
    "B": ("Generate a structured mammography radiology report with breast composition "
          "(ACR density), findings, BI-RADS category, and recommendation."),
}
# По умолчанию B — чтобы воспроизводить опубликованные числа Table 4.
PROMPT_TEXT = PROMPTS["B"]
IMG_SIZE     = 448
MAX_NEW_TOK  = 200


# ─────────────────────────── данные и сплиты ───────────────────────────

def split_files():
    """Сплит AMRG, как в исходных скриптах: по изображениям, не по пациентам.

    Оставлен намеренно ради сопоставимости с AMRG (решение по R3.1);
    утечки документируются отдельно в 00_split_manifest.py.
    """
    f = sorted(os.listdir(REPS_DIR))
    return {"train": f[:407], "val": f[407:458], "test": f[-52:]}


def find_image(img_id):
    num = img_id.replace('Img', '').replace('IMG', '')
    for f in os.listdir(IMGS_DIR):
        if num in f and not f.endswith('.txt'):
            return os.path.join(IMGS_DIR, f)
    return None


def load_pairs(files):
    pairs = []
    for rf in files:
        p = find_image(rf.replace('.txt', ''))
        if not p:
            continue
        with open(os.path.join(REPS_DIR, rf), encoding='utf-8', errors='ignore') as fh:
            report = fh.read().strip()
        if report:
            pairs.append({"id": rf.replace('.txt', ''), "img_path": p, "report": report})
    return pairs


# ─────────────────────────── загрузка моделей ───────────────────────────

def _bnb():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def load_model(spec):
    """spec: dict с ключами
         adapter    — путь к адаптеру Stage 2 (или None для zero-shot)
         stage1     — True, если перед Stage 2 нужно влить веса Stage 1

    Механика переноса Stage 1 → Stage 2 повторяет lora_ablation.py:
    адаптер Stage 1 мержится в базовые веса, поверх ставится адаптер Stage 2.
    Это и есть ответ на R3.3 (почему r=16 и r=64 совместимы).
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=_bnb(), device_map="auto", dtype=torch.bfloat16)
    model.config.use_cache = True

    if spec.get("stage1"):
        model = PeftModel.from_pretrained(model, STAGE1)
        model = model.merge_and_unload()

    if spec.get("adapter"):
        model = PeftModel.from_pretrained(model, spec["adapter"])

    model.eval()
    return model, processor


def free(*objs):
    import torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────── генерация ───────────────────────────

def generate_one(model, processor, img_path, rag_context=None):
    import torch
    from PIL import Image
    try:
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        return ""
    ctx = f"\nClinical guidelines:\n{rag_context}\n" if rag_context else ""
    prompt = (f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}{ctx}"
              f"<end_of_turn>\n<start_of_turn>model\n")
    inp = processor(text=prompt, images=img, return_tensors="pt",
                    truncation=True, max_length=768).to(model.device)
    inp["token_type_ids"] = torch.zeros_like(inp["input_ids"])
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOK, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()


def generate_all(model, processor, pairs, retriever=None, log_every=10):
    preds = []
    for i, p in enumerate(pairs):
        ctx = retriever(p) if retriever else None
        preds.append({"id": p["id"], "ref": p["report"],
                      "hyp": generate_one(model, processor, p["img_path"], ctx)})
        if (i + 1) % log_every == 0:
            print(f"    {i+1}/{len(pairs)}", flush=True)
    return preds


def save_preds(name, split, preds, meta=None):
    os.makedirs(PRED_DIR, exist_ok=True)
    path = f"{PRED_DIR}/{name}__{split}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "split": split, "meta": meta or {},
                   "predictions": preds}, f, ensure_ascii=False, indent=2)
    print(f"  → {path}")
    return path


def load_preds(name, split):
    with open(f"{PRED_DIR}/{name}__{split}.json", encoding="utf-8") as f:
        return json.load(f)["predictions"]


# ─────────────────────────── метрики (по-сэмплово) ───────────────────────────

def per_sample_scores(preds):
    """Возвращает по-сэмпловые метрики — основа для парного bootstrap и CI."""
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    for pkg in ("punkt", "punkt_tab", "wordnet"):
        nltk.download(pkg, quiet=True)

    sf = SmoothingFunction().method1
    rs = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rows = []
    for p in preds:
        ref, hyp = p["ref"], p["hyp"]
        rt, ht = ref.split(), hyp.split()
        s = rs.score(ref, hyp)
        try:
            met = meteor_score([nltk.word_tokenize(ref.lower())], nltk.word_tokenize(hyp.lower()))
        except Exception:
            met = 0.0
        rows.append({
            "id": p["id"],
            "bleu1": sentence_bleu([rt], ht, weights=(1, 0, 0, 0), smoothing_function=sf) if ht else 0.0,
            "bleu4": sentence_bleu([rt], ht, weights=(.25, .25, .25, .25), smoothing_function=sf) if ht else 0.0,
            "rouge1": s['rouge1'].fmeasure,
            "rouge2": s['rouge2'].fmeasure,
            "rougeL": s['rougeL'].fmeasure,
            "meteor": met,
        })
    return rows


def corpus_scores(preds):
    """Корпусные метрики (BLEU и CIDEr корпусные по определению)."""
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [p["ref"] for p in preds]
    hyps = [p["hyp"] for p in preds]
    sf = SmoothingFunction().method1
    out = {
        "bleu1": corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps],
                             weights=(1, 0, 0, 0), smoothing_function=sf),
        "bleu4": corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps],
                             weights=(.25, .25, .25, .25), smoothing_function=sf),
    }
    rows = per_sample_scores(preds)
    for k in ("rouge1", "rouge2", "rougeL", "meteor"):
        out[k] = sum(r[k] for r in rows) / len(rows)
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [h] for i, h in enumerate(hyps)}
        out["cider"], _ = Cider().compute_score(gts, res)
    except Exception as e:
        print(f"  CIDEr недоступен: {e}")
        out["cider"] = None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


# ─────────────────────────── клинические извлечения ───────────────────────────

def extract_birads(text):
    for p in [r'BI-?RADS[\s:]*(\d)', r'BIRADS[\s:]*(\d)', r'CATEGORY\s*(\d)']:
        m = re.search(p, text.upper())
        if m:
            return int(m.group(1))
    return None


def extract_density(text):
    """ACR-плотность: явная метка ACR X либо словесное описание DMID."""
    m = re.search(r'ACR\s*\(?\s*([A-Da-d])\b', text)
    if m:
        return m.group(1).upper()
    t = text.lower()
    if 'fatty' in t and 'glandular' in t:
        return 'B'
    if 'dense' in t and 'glandular' in t:
        return 'D'
    if 'fibro glandular' in t or 'fibroglandular' in t:
        return 'C'
    if 'fatty' in t:
        return 'A'
    return None
