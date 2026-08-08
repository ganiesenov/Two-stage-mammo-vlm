"""
Evaluate the Russian (Stage 3) model on the held-out DMID-RU test split.

Uses the SAME split_ru.json produced by finetune_dmid_ru.py so the test
studies are never seen in training. Per-study 4-view input -> generated report.
Metrics: BLEU-4, ROUGE-L, METEOR, BERTScore (multilingual, ru).
"""
import os, json, torch, argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)

MODEL_ID  = "google/medgemma-4b-it"
DATA_DIR  = "/mnt/d/dmid_ru"
PAIRS     = os.path.join(DATA_DIR, "pairs_study.json")
SPLIT     = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/scripts/3_finetune/split_ru.json"
DEFAULT_LORA = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/medgemma_dmid_ru"

VIEW_RU = {"RCC": "Правая КК", "LCC": "Левая КК", "RMLO": "Правая МЛК", "LMLO": "Левая МЛК"}
INSTRUCTION = ("Составь структурированное заключение по маммографии обеих молочных желёз "
               "на русском языке: рентгенологический тип плотности по ACR, описание (протокол), "
               "заключение, категорию BI-RADS и рекомендации.")


def build_messages(pair):
    content = []
    for view in pair["views"]:
        content.append({"type": "text", "text": f"{VIEW_RU.get(view, view)} проекция:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


def generate(model, processor, pair):
    images = []
    for rel in pair["images"]:
        try:
            images.append(Image.open(os.path.join(DATA_DIR, rel)).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (896, 896), 128))
    prompt = processor.apply_chat_template(build_messages(pair), add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--out",  default="/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/results/dmid_ru_eval.json")
    args = ap.parse_args()

    split = json.load(open(SPLIT, encoding="utf-8"))
    test_ids = set(split["test"])
    pairs = [p for p in json.load(open(PAIRS, encoding="utf-8")) if p["patient_id"] in test_ids]
    print(f"Test studies: {len(pairs)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.lora)
    model.eval()

    refs, hyps, rows = [], [], []
    for i, p in enumerate(pairs):
        gen = generate(model, processor, p)
        refs.append(p["report"].strip()); hyps.append(gen)
        rows.append({"patient_id": p["patient_id"], "reference": p["report"].strip(), "generated": gen})
        print(f"[{i+1}/{len(pairs)}] {p['patient_id']}  ({len(gen)} chars)")

    sf = SmoothingFunction().method1
    bleu = corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps], smoothing_function=sf)
    rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rougeL = sum(rs.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyps)) / len(refs)
    meteor = sum(meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)) / len(refs)
    P, R, F1 = bert_score(hyps, refs, lang="ru", verbose=False)
    bert_f1 = float(F1.mean())

    metrics = {"n": len(refs), "BLEU-4": round(bleu, 4), "ROUGE-L": round(rougeL, 4),
               "METEOR": round(meteor, 4), "BERTScore": round(bert_f1, 4)}
    print("\n=== DMID-RU test metrics ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"metrics": metrics, "samples": rows}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
