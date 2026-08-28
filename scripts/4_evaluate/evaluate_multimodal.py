import json
import torch
import numpy as np
import faiss
import pickle
import pandas as pd
from PIL import Image
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID      = "google/medgemma-4b-it"
TEXT_LORA     = "./medgemma_finetuned"
MM_LORA       = "./medgemma_multimodal"
INDEX_FILE    = "./birads_faiss.index"
META_FILE     = "./birads_meta.pkl"
VINDR_JSONL   = "./vindr/synthetic_reports.jsonl"
VINDR_IMG     = "./vindr"

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

def build_text_prompt(r, use_rag=False):
    rag_ctx = ""
    if use_rag:
        ctx = retrieve(f"BI-RADS {r['breast_birads']} {r['finding_categories']} mammography")
        rag_ctx = f"\nClinical guidelines:\n{ctx}\n"
    msg = f"Generate a structured mammography report.{rag_ctx}\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}\nDensity: {r['breast_density']}\nLaterality: {r['laterality']}"
    return f"<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n"

def build_mm_prompt(r, use_rag=False):
    rag_ctx = ""
    if use_rag:
        ctx = retrieve(f"BI-RADS {r['breast_birads']} {r['finding_categories']} mammography")
        rag_ctx = f"\nClinical guidelines:\n{ctx}\n"
    msg = f"Generate a structured mammography report.{rag_ctx}\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}"
    return f"<start_of_turn>user\n<start_of_image>\n{msg}<end_of_turn>\n<start_of_turn>model\n"

def gen_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def gen_mm(model, processor, prompt, image_path):
    image = Image.open(image_path).convert("RGB").resize((448,448))
    inputs = processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

def main():
    print("Загружаю тестовые данные (VinDr с изображениями)...")
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    print(f"Тест: {len(test)} примеров")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_LORA)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Baseline zero-shot (text)
    print("\n[1/4] Baseline zero-shot...")
    base_text = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    hyps1 = [gen_text(base_text, tokenizer, build_text_prompt(r)) for r in test]
    m1 = compute_metrics(refs, hyps1, "1. Baseline (zero-shot, text-only)")
    del base_text; torch.cuda.empty_cache()

    # 2. Text fine-tuned + RAG
    print("\n[2/4] Text fine-tuned + RAG...")
    base2 = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    ft_text = PeftModel.from_pretrained(base2, TEXT_LORA); ft_text.eval()
    hyps2 = [gen_text(ft_text, tokenizer, build_text_prompt(r, use_rag=True)) for r in test]
    m2 = compute_metrics(refs, hyps2, "2. Text FT + RAG")
    del ft_text, base2; torch.cuda.empty_cache()

    # 3. Multimodal fine-tuned (no RAG)
    print("\n[3/4] Multimodal fine-tuned (no RAG)...")
    base3 = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    mm_model = PeftModel.from_pretrained(base3, MM_LORA); mm_model.eval()
    hyps3 = [gen_mm(mm_model, processor, build_mm_prompt(r), r["image_path"]) for r in test]
    m3 = compute_metrics(refs, hyps3, "3. Multimodal FT (no RAG)")

    # 4. Multimodal fine-tuned + RAG
    print("\n[4/4] Multimodal fine-tuned + RAG (ours)...")
    hyps4 = [gen_mm(mm_model, processor, build_mm_prompt(r, use_rag=True), r["image_path"]) for r in test]
    m4 = compute_metrics(refs, hyps4, "4. Multimodal FT + RAG (ours)")

    results = {"baseline": m1, "text_ft_rag": m2, "multimodal_ft": m3, "multimodal_ft_rag": m4}
    out = "./eval_results_multimodal.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены в {out}")

if __name__ == "__main__":
    main()
