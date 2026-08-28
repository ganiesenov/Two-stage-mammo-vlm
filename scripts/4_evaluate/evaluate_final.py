import json
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID   = "google/medgemma-4b-it"
LORA_DIR   = "./medgemma_finetuned"
INDEX_FILE = "./birads_faiss.index"
META_FILE  = "./birads_meta.pkl"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
CBIS_JSONL  = "./cbis-ddsm/synthetic_reports_cbis.jsonl"

# RAG
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=3):
    emb = encoder.encode([query]).astype("float32")
    faiss.normalize_L2(emb)
    _, ids = index.search(emb, k)
    return " ".join(chunks[i]["text"] for i in ids[0])

def load_test(n=40):
    data = []
    for path in [VINDR_JSONL, CBIS_JSONL]:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("is_valid") and r.get("synthetic_report"):
                        data.append(r)
                except: pass
    return data[-n:]

def build_prompt(report, use_rag=False):
    findings   = report.get("finding_categories", report.get("finding_type", "mass"))
    birads     = report.get("breast_birads", report.get("assessment", ""))
    density    = report.get("breast_density", "")
    laterality = report.get("laterality", "")
    rag_context = ""
    if use_rag:
        context = retrieve(f"BI-RADS {birads} {findings} mammography")
        rag_context = f"\nClinical guidelines:\n{context}\n"
    user_msg = f"Generate a structured mammography report.{rag_context}\nFindings: {findings}\nBI-RADS: {birads}\nDensity: {density}\nLaterality: {laterality}"
    return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def compute_metrics(references, hypotheses, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1,r2,rl = [],[],[]
    for ref,hyp in zip(references,hypotheses):
        s = scorer.score(ref,hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    _,_,F = bert_score(hypotheses, references, lang="en", verbose=False)
    metrics = {
        "bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
        "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
        "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)
    }
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    for k,v in metrics.items():
        print(f"  {k:12s}: {v}")
    return metrics

def main():
    print("Загружаю тестовые данные...")
    test = load_test(n=40)
    refs = [r["synthetic_report"] for r in test]

    print("Загружаю модель...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)

    # 1. Baseline zero-shot
    print("\n[1/3] Baseline zero-shot...")
    hyps_base = [generate(base, tokenizer, build_prompt(r, use_rag=False)) for r in test]
    m1 = compute_metrics(refs, hyps_base, "1. Baseline (zero-shot)")

    # 2. Fine-tuned без RAG
    print("\n[2/3] Fine-tuned без RAG...")
    ft_model = PeftModel.from_pretrained(base, LORA_DIR)
    ft_model.eval()
    hyps_ft = [generate(ft_model, tokenizer, build_prompt(r, use_rag=False)) for r in test]
    m2 = compute_metrics(refs, hyps_ft, "2. Fine-tuned (no RAG)")

    # 3. Fine-tuned + RAG
    print("\n[3/3] Fine-tuned + RAG...")
    hyps_rag = [generate(ft_model, tokenizer, build_prompt(r, use_rag=True)) for r in test]
    m3 = compute_metrics(refs, hyps_rag, "3. Fine-tuned + RAG (ours)")

    # Сохраняем
    results = {"baseline": m1, "finetuned": m2, "finetuned_rag": m3}
    out_path = "./eval_results_final.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены в {out_path}")

if __name__ == "__main__":
    main()
