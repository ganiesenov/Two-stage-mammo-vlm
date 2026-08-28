import json
import numpy as np
from scipy import stats

# Загружаем все результаты
with open("./eval_results_multimodal.json") as f:
    mm = json.load(f)
with open("./eval_results_sota.json") as f:
    sota = json.load(f)

# Для bootstrap p-value нужны individual scores
# Пересчитаем ROUGE для каждого примера отдельно
import json
from pathlib import Path
from rouge_score import rouge_scorer
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import torch
import faiss, pickle
from sentence_transformers import SentenceTransformer

VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
VINDR_IMG   = "./vindr"
MODEL_ID    = "google/medgemma-4b-it"
MM_LORA     = "./medgemma_multimodal"

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

def gen_mm(model, processor, r, use_rag=False, retriever=None):
    rag_ctx = ""
    if use_rag and retriever:
        rag_ctx = f"\nClinical guidelines:\n{retriever(r)}\n"
    prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report.{rag_ctx}\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}<end_of_turn>\n<start_of_turn>model\n"
    image = Image.open(r["image_path"]).convert("RGB").resize((448,448))
    inputs = processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def rouge1_scores(refs, hyps):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return [scorer.score(r, h)['rouge1'].fmeasure for r, h in zip(refs, hyps)]

def bootstrap_pvalue(scores_a, scores_b, n_bootstrap=10000):
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    combined = np.concatenate([scores_a, scores_b])
    n = len(scores_a)
    diffs = []
    for _ in range(n_bootstrap):
        perm = np.random.permutation(combined)
        diffs.append(np.mean(perm[:n]) - np.mean(perm[n:]))
    p = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p

def main():
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]

    print("Загружаю Multimodal модель...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    mm_model = PeftModel.from_pretrained(base, MM_LORA); mm_model.eval()

    print("Генерирую отчёты (Multimodal FT)...")
    hyps_mm = [gen_mm(mm_model, processor, r) for r in test]

    print("Генерирую отчёты (Baseline)...")
    del mm_model, base; torch.cuda.empty_cache()
    base2 = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    base2.eval()
    hyps_base = [gen_mm(base2, processor, r) for r in test]

    scores_mm   = rouge1_scores(refs, hyps_mm)
    scores_base = rouge1_scores(refs, hyps_base)

    p = bootstrap_pvalue(np.array(scores_mm), np.array(scores_base))
    t_stat, p_ttest = stats.ttest_rel(scores_mm, scores_base)

    print(f"\n{'='*50}")
    print(f"Statistical Significance (ROUGE-1)")
    print(f"{'='*50}")
    print(f"Multimodal FT mean:  {np.mean(scores_mm):.4f} ± {np.std(scores_mm):.4f}")
    print(f"Baseline mean:       {np.mean(scores_base):.4f} ± {np.std(scores_base):.4f}")
    print(f"Bootstrap p-value:   {p:.4f}")
    print(f"Paired t-test p:     {p_ttest:.4f}")
    print(f"Significant (p<0.05): {'YES' if p < 0.05 else 'NO'}")

    results = {
        "multimodal_ft_mean": float(np.mean(scores_mm)),
        "multimodal_ft_std": float(np.std(scores_mm)),
        "baseline_mean": float(np.mean(scores_base)),
        "baseline_std": float(np.std(scores_base)),
        "bootstrap_pvalue": float(p),
        "ttest_pvalue": float(p_ttest),
        "significant": bool(p < 0.05)
    }
    with open("./significance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Сохранено в significance_results.json")

if __name__ == "__main__":
    main()
