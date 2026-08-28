"""
Evaluate all 3 variants on DMID test set (last 52 images)
1. MedGemma zero-shot
2. DMID-only (fresh LoRA)
3. Synthetic → DMID (two-stage)
"""
import os, torch, json
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID      = "google/medgemma-4b-it"
DMID_ONLY     = "./medgemma_dmid_only"
TWO_STAGE     = "./medgemma_dmid"
DMID_IMGS     = "./dmid/TIFF Images/TIFF Images/"
DMID_REPS     = "./dmid/Reports/Reports/"

def load_test_pairs(n=52):
    pairs = []
    report_files = sorted(os.listdir(DMID_REPS))
    test_files = report_files[-n:]
    for rf in test_files:
        img_id = rf.replace('.txt', '')
        img_num = img_id.replace('Img', '').replace('IMG', '')
        img_path = None
        for f in os.listdir(DMID_IMGS):
            if img_num in f and not f.endswith('.txt'):
                img_path = os.path.join(DMID_IMGS, f)
                break
        with open(os.path.join(DMID_REPS, rf), encoding='utf-8', errors='ignore') as f:
            report = f.read().strip()
        if img_path and report:
            pairs.append({"img_path": img_path, "real_report": report})
    print(f"Test pairs: {len(pairs)}/{n}")
    return pairs

def generate(model, processor, img_path):
    try:
        image = Image.open(img_path).convert("RGB").resize((448, 448))
    except:
        return ""
    prompt = "<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report with breast composition, findings, BI-RADS category and recommendation.<end_of_turn>\n<start_of_turn>model\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()

def compute_metrics(refs, hyps, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    _, _, F = bert_score(hyps, refs, lang="en", verbose=False)
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    for k, v in m.items():
        print(f"  {k:12s}: {v}")
    return m

def run_inference(model, processor, pairs, label):
    print(f"\n--- {label} ---")
    model.eval()
    hyps = []
    for i, p in enumerate(pairs):
        hyps.append(generate(model, processor, p["img_path"]))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    return hyps

def main():
    pairs = load_test_pairs(52)
    refs = [p["real_report"] for p in pairs]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)

    results = {}

    # 1. Zero-shot
    hyps = run_inference(base, processor, pairs, "MedGemma zero-shot")
    results["zero_shot"] = compute_metrics(refs, hyps, "MedGemma zero-shot")

    # 2. DMID-only
    model_dmid = PeftModel.from_pretrained(base, DMID_ONLY)
    hyps = run_inference(model_dmid, processor, pairs, "DMID-only (fresh LoRA)")
    results["dmid_only"] = compute_metrics(refs, hyps, "DMID-only (fresh LoRA)")
    del model_dmid; torch.cuda.empty_cache()

    # 3. Two-stage (synthetic → DMID)
    model_2s = PeftModel.from_pretrained(base, TWO_STAGE)
    hyps = run_inference(model_2s, processor, pairs, "Synthetic → DMID (two-stage)")
    results["two_stage"] = compute_metrics(refs, hyps, "Synthetic → DMID (two-stage)")

    # Save
    results["meta"] = {"dataset": "DMID", "n_test": len(pairs),
                       "note": "Real radiologist reports, AMRG test split"}
    out = "./results/eval_dmid_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nСохранено в {out}")

    # Пример
    print("\n--- Пример (two-stage) ---")
    print(f"REAL:\n{refs[0][:300]}")

if __name__ == "__main__":
    main()
