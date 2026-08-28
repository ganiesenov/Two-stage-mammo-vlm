import json
import re
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_ID    = "google/medgemma-4b-it"
MM_LORA     = "./medgemma_multimodal"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
VINDR_IMG   = "./vindr"

# BI-RADS → правильные рекомендации
BIRADS_RECS = {
    "1": ["routine", "annual", "screening"],
    "2": ["routine", "annual", "screening"],
    "3": ["follow", "short", "6 month", "probably benign"],
    "4": ["biopsy", "tissue", "sampling", "histol"],
    "5": ["biopsy", "tissue", "sampling", "histol", "surgical"],
    "6": ["treatment", "surgical", "therapy", "excision"],
}

def get_birads_num(s):
    m = re.search(r'(\d)', str(s))
    return m.group(1) if m else None

def check_recommendation(text, birads_num):
    if not birads_num or birads_num not in BIRADS_RECS:
        return None
    text_lower = text.lower()
    keywords = BIRADS_RECS[birads_num]
    return any(k in text_lower for k in keywords)

def load_test(n=30):
    data = []
    with open(VINDR_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("is_valid"): continue
                img_path = f"{VINDR_IMG}/{r['study_id']}/{r['image_id']}.png"
                if Path(img_path).exists():
                    r["image_path"] = img_path
                    data.append(r)
            except: pass
    return data[-n:]

def gen(model, processor, r):
    prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report.\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}<end_of_turn>\n<start_of_turn>model\n"
    image = Image.open(r["image_path"]).convert("RGB").resize((448,448))
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                      truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

def evaluate_clinical(test, hyps, label):
    rec_correct = 0
    rec_total = 0
    halluc_count = 0

    for r, hyp in zip(test, hyps):
        birads = get_birads_num(r["breast_birads"])
        rec = check_recommendation(hyp, birads)
        if rec is not None:
            rec_total += 1
            if rec:
                rec_correct += 1

        # Галлюцинации — упоминание находок которых нет
        findings = str(r["finding_categories"]).lower()
        if "no finding" in findings and any(
            w in hyp.lower() for w in ["mass", "calcification", "lesion", "asymmetry", "distortion"]):
            halluc_count += 1

    rec_score = rec_correct / rec_total if rec_total > 0 else 0
    halluc_rate = halluc_count / len(test)

    print(f"\n{label}")
    print(f"  Recommendation accuracy: {rec_score:.3f} ({rec_correct}/{rec_total})")
    print(f"  Hallucination rate:      {halluc_rate:.3f} ({halluc_count}/{len(test)})")
    return {"recommendation_accuracy": round(rec_score, 4),
            "hallucination_rate": round(halluc_rate, 4)}

def main():
    test = load_test(n=30)
    print(f"Тест: {len(test)} примеров")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)

    print("\n[1/2] Baseline...")
    base.eval()
    hyps_base = [gen(base, processor, r) for r in test]
    m1 = evaluate_clinical(test, hyps_base, "Baseline (zero-shot)")

    print("\n[2/2] Multimodal FT...")
    mm = PeftModel.from_pretrained(base, MM_LORA); mm.eval()
    hyps_mm = [gen(mm, processor, r) for r in test]
    m2 = evaluate_clinical(test, hyps_mm, "Multimodal FT (ours)")

    results = {"baseline": m1, "multimodal_ft": m2}
    out = "./clinical_metrics.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено в {out}")

if __name__ == "__main__":
    main()
