import json
import torch
import faiss
import pickle
import requests
from PIL import Image
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID    = "google/medgemma-4b-it"
MM_LORA     = "./medgemma_multimodal"
INDEX_FILE  = "./birads_faiss.index"
META_FILE   = "./birads_meta.pkl"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
VINDR_IMG   = "./vindr"
OLLAMA_MODEL = "llama3.1:8b"

index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=2):
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

def gen_mm(model, processor, r):
    """Генерация без RAG"""
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

def rag_refine(draft, birads, findings):
    """Post-hoc RAG: улучшаем черновик через Ollama + BI-RADS контекст"""
    context = retrieve(f"BI-RADS {birads} {findings} mammography recommendation")
    prompt = f"""You are a radiologist. Refine this mammography report draft using the clinical guidelines below.
Keep the same structure but improve clinical accuracy and terminology.

Clinical guidelines:
{context}

Draft report:
{draft}

Refined report (same length, improved terminology):"""
    
    r = requests.post("http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120)
    return r.json()["response"].strip()

def compute_metrics(refs, hyps, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1,r2,rl = [],[],[]
    for ref,hyp in zip(refs,hyps):
        s = scorer.score(ref,hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    _,_,F = bert_score(hyps, refs, lang="en", verbose=False)
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

def main():
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    print(f"Тест: {len(test)} примеров")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    mm = PeftModel.from_pretrained(base, MM_LORA)
    mm.eval()

    # 1. Multimodal FT без RAG
    print("\n[1/2] Multimodal FT (no RAG)...")
    hyps_mm = []
    for r in test:
        hyps_mm.append(gen_mm(mm, processor, r))
        print(".", end="", flush=True)
    m1 = compute_metrics(refs, hyps_mm, "Multimodal FT (no RAG)")

    # 2. Multimodal FT + Post-hoc RAG (Ollama refinement)
    print("\n[2/2] Multimodal FT + Post-hoc RAG...")
    hyps_rag = []
    for r, draft in zip(test, hyps_mm):
        refined = rag_refine(draft, r["breast_birads"], r["finding_categories"])
        hyps_rag.append(refined)
        print(".", end="", flush=True)
    m2 = compute_metrics(refs, hyps_rag, "Multimodal FT + Post-hoc RAG (ours)")

    results = {"multimodal_ft": m1, "multimodal_ft_posthoc_rag": m2}
    out = "./eval_rag_posthoc.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено в {out}")

if __name__ == "__main__":
    main()
