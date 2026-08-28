"""
Full DMID evaluation: p-values, clinical metrics, qualitative examples
Runs inference on all 3 variants and saves everything
"""
import os, torch, json, re
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID   = "google/medgemma-4b-it"
DMID_ONLY  = "./medgemma_dmid_only"
TWO_STAGE  = "./medgemma_dmid"
DMID_IMGS  = "./dmid/TIFF Images/TIFF Images/"
DMID_REPS  = "./dmid/Reports/Reports/"
OUT_DIR    = "./results"

# ── Data loading ──
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
        with open(os.path.join(DMID_REPS, rf), encoding='utf-8', errors='ignore') as fh:
            report = fh.read().strip()
        if img_path and report:
            pairs.append({"img_id": img_id, "img_path": img_path, "real_report": report})
    print(f"Test pairs: {len(pairs)}/{n}")
    return pairs

# ── Inference ──
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

def run_inference(model, processor, pairs, label):
    print(f"\n--- {label} ---")
    model.eval()
    hyps = []
    for i, p in enumerate(pairs):
        hyps.append(generate(model, processor, p["img_path"]))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()
    return hyps

# ── NLP Metrics (per-sample) ──
def per_sample_rouge(refs, hyps):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    return np.array(r1), np.array(r2), np.array(rl)

def compute_all_metrics(refs, hyps):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    r1, r2, rl = per_sample_rouge(refs, hyps)
    _, _, F = bert_score(hyps, refs, lang="en", verbose=False)
    return {
        "bleu1": round(bleu1, 4), "bleu4": round(bleu4, 4),
        "rouge1": round(r1.mean(), 4), "rouge2": round(r2.mean(), 4),
        "rougeL": round(rl.mean(), 4), "bertscore": round(F.mean().item(), 4),
        "rouge1_std": round(r1.std(), 4), "rougeL_std": round(rl.std(), 4),
    }

# ── Bootstrap p-values ──
def bootstrap_pvalue(scores_a, scores_b, n_bootstrap=10000):
    """Test if scores_b > scores_a (one-sided)"""
    observed_diff = scores_b.mean() - scores_a.mean()
    combined = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)
    count = 0
    for _ in range(n_bootstrap):
        perm = np.random.permutation(combined)
        diff = perm[n_a:].mean() - perm[:n_a].mean()
        if diff >= observed_diff:
            count += 1
    return count / n_bootstrap

# ── Clinical Metrics ──
def extract_birads(text):
    """Extract BI-RADS category from report text"""
    text_upper = text.upper()
    patterns = [
        r'BI-?RADS[\s:]*(\d)',
        r'BIRADS[\s:]*(\d)',
        r'BI-?RADS\s*CATEGORY\s*(\d)',
        r'CATEGORY\s*(\d)',
    ]
    for p in patterns:
        m = re.search(p, text_upper)
        if m:
            return int(m.group(1))
    return None

def has_recommendation(text):
    """Check if report contains management recommendation"""
    keywords = [
        'recommend', 'follow-up', 'follow up', 'biopsy', 'routine',
        'screening', 'additional', 'further', 'suggest', 'advised',
        'annual', 'diagnostic', 'tissue sampling', 'short interval',
        'recall', 'aspiration', 'ultrasound', 'MRI'
    ]
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)

BIRADS_RECS = {
    1: ['routine', 'screening', 'annual', 'negative'],
    2: ['routine', 'screening', 'annual', 'benign'],
    3: ['follow-up', 'follow up', 'short interval', '6 month', 'six month', 'probably benign'],
    4: ['biopsy', 'tissue sampling', 'histopatholog', 'suspicious'],
    5: ['biopsy', 'tissue sampling', 'histopatholog', 'highly suggestive', 'malignant'],
}

def check_recommendation_accuracy(text, birads):
    """Check if recommendation matches BI-RADS category"""
    if birads is None or birads not in BIRADS_RECS:
        return None
    text_lower = text.lower()
    return any(k in text_lower for k in BIRADS_RECS[birads])

VALID_FINDINGS = [
    'mass', 'calcif', 'asymmetr', 'architectural distortion', 'distortion',
    'opacity', 'lesion', 'density', 'nodule', 'fibroadenoma', 'cyst',
    'no abnormal', 'no significant', 'normal', 'benign', 'malignant',
    'microcalcif', 'skin', 'nipple', 'axillary', 'lymph'
]

def check_hallucination(gen_text, ref_text):
    """Simple hallucination check: generated findings not in reference"""
    gen_lower = gen_text.lower()
    ref_lower = ref_text.lower()
    hallucinated = False
    for finding in ['mass', 'calcification', 'architectural distortion', 'asymmetry']:
        if finding in gen_lower and finding not in ref_lower:
            # Check if reference has 'no' + finding
            if f'no {finding}' not in ref_lower:
                hallucinated = True
    return hallucinated

def compute_clinical_metrics(refs, hyps, label):
    n = len(refs)
    birads_correct = 0
    birads_total = 0
    rec_correct = 0
    rec_total = 0
    hallucinations = 0
    has_rec_count = 0

    for ref, hyp in zip(refs, hyps):
        ref_birads = extract_birads(ref)
        gen_birads = extract_birads(hyp)

        # BI-RADS accuracy
        if ref_birads is not None:
            birads_total += 1
            if gen_birads == ref_birads:
                birads_correct += 1

        # Recommendation accuracy
        if ref_birads is not None:
            rec_total += 1
            if check_recommendation_accuracy(hyp, ref_birads):
                rec_correct += 1

        # Has recommendation
        if has_recommendation(hyp):
            has_rec_count += 1

        # Hallucination
        if check_hallucination(hyp, ref):
            hallucinations += 1

    metrics = {
        "birads_accuracy": round(birads_correct / birads_total, 4) if birads_total > 0 else None,
        "birads_total": birads_total,
        "rec_accuracy": round(rec_correct / rec_total, 4) if rec_total > 0 else None,
        "has_recommendation_rate": round(has_rec_count / n, 4),
        "hallucination_rate": round(hallucinations / n, 4),
    }

    print(f"\n{'='*55}")
    print(f"  Clinical Metrics: {label}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:28s}: {v}")
    return metrics

# ── Main ──
def main():
    pairs = load_test_pairs(52)
    refs = [p["real_report"] for p in pairs]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)

    all_results = {}
    all_hyps = {}

    # 1. Zero-shot
    hyps_zs = run_inference(base, processor, pairs, "MedGemma zero-shot")
    all_hyps["zero_shot"] = hyps_zs
    all_results["zero_shot"] = compute_all_metrics(refs, hyps_zs)
    all_results["zero_shot_clinical"] = compute_clinical_metrics(refs, hyps_zs, "Zero-shot")

    # 2. DMID-only
    model_do = PeftModel.from_pretrained(base, DMID_ONLY)
    hyps_do = run_inference(model_do, processor, pairs, "DMID-only")
    all_hyps["dmid_only"] = hyps_do
    all_results["dmid_only"] = compute_all_metrics(refs, hyps_do)
    all_results["dmid_only_clinical"] = compute_clinical_metrics(refs, hyps_do, "DMID-only")
    del model_do; torch.cuda.empty_cache()

    # 3. Two-stage
    model_2s = PeftModel.from_pretrained(base, TWO_STAGE)
    hyps_2s = run_inference(model_2s, processor, pairs, "Synthetic → DMID")
    all_hyps["two_stage"] = hyps_2s
    all_results["two_stage"] = compute_all_metrics(refs, hyps_2s)
    all_results["two_stage_clinical"] = compute_clinical_metrics(refs, hyps_2s, "Two-stage")
    del model_2s; torch.cuda.empty_cache()

    # ── Bootstrap p-values ──
    print("\n" + "="*55)
    print("  Bootstrap P-Values (n=10000)")
    print("="*55)

    _, _, rl_do = per_sample_rouge(refs, hyps_do)
    _, _, rl_2s = per_sample_rouge(refs, hyps_2s)
    _, _, rl_zs = per_sample_rouge(refs, hyps_zs)

    r1_do, _, _ = per_sample_rouge(refs, hyps_do)
    r1_2s, _, _ = per_sample_rouge(refs, hyps_2s)

    p_2s_vs_do_rl = bootstrap_pvalue(rl_do, rl_2s)
    p_2s_vs_do_r1 = bootstrap_pvalue(r1_do, r1_2s)
    p_2s_vs_zs = bootstrap_pvalue(rl_zs, rl_2s)
    p_do_vs_zs = bootstrap_pvalue(rl_zs, rl_do)

    print(f"  Two-stage vs DMID-only (ROUGE-L): p = {p_2s_vs_do_rl:.4f}")
    print(f"  Two-stage vs DMID-only (ROUGE-1): p = {p_2s_vs_do_r1:.4f}")
    print(f"  Two-stage vs Zero-shot (ROUGE-L): p = {p_2s_vs_zs:.4f}")
    print(f"  DMID-only vs Zero-shot (ROUGE-L): p = {p_do_vs_zs:.4f}")

    all_results["p_values"] = {
        "two_stage_vs_dmid_only_rougeL": p_2s_vs_do_rl,
        "two_stage_vs_dmid_only_rouge1": p_2s_vs_do_r1,
        "two_stage_vs_zero_shot_rougeL": p_2s_vs_zs,
        "dmid_only_vs_zero_shot_rougeL": p_do_vs_zs,
    }

    # ── Qualitative examples ──
    examples = []
    for i in range(min(5, len(pairs))):
        examples.append({
            "img_id": pairs[i]["img_id"],
            "real_report": refs[i],
            "zero_shot": hyps_zs[i],
            "dmid_only": hyps_do[i],
            "two_stage": hyps_2s[i],
        })

    all_results["qualitative_examples"] = examples

    print("\n" + "="*55)
    print("  Qualitative Examples")
    print("="*55)
    for i, ex in enumerate(examples[:3]):
        print(f"\n--- Example {i+1}: {ex['img_id']} ---")
        print(f"REAL:\n{ex['real_report'][:250]}")
        print(f"\nTWO-STAGE:\n{ex['two_stage'][:250]}")
        print()

    # ── Save ──
    out_path = os.path.join(OUT_DIR, "eval_dmid_full.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nВсё сохранено в {out_path}")

if __name__ == "__main__":
    main()
