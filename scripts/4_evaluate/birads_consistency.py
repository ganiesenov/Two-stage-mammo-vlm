import json
import re
import torch
import faiss
import pickle
from PIL import Image
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_ID  = "google/medgemma-4b-it"
MM_LORA   = "./medgemma_multimodal"
INDEX_FILE = "./birads_faiss.index"
META_FILE  = "./birads_meta.pkl"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
VINDR_IMG   = "./vindr"

# RAG с k=1
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=1):
    emb = encoder.encode([query]).astype("float32")
    faiss.normalize_L2(emb)
    _, ids = index.search(emb, k)
    return chunks[ids[0][0]]["text"]

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

def extract_birads(text):
    """Извлекаем BI-RADS категорию из текста"""
    patterns = [
        r'bi-?rads[^\d]*(\d)',
        r'category[^\d]*(\d)',
        r'assessment[^\d]*(\d)',
    ]
    text_lower = text.lower()
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return m.group(1)
    return None

def get_true_birads(birads_str):
    """Извлекаем номер из строки типа 'BI-RADS 4'"""
    m = re.search(r'(\d)', str(birads_str))
    return m.group(1) if m else None

def gen(model, processor, r, use_rag=False):
    rag_ctx = ""
    if use_rag:
        ctx = retrieve(f"BI-RADS {r['breast_birads']} {r['finding_categories']} mammography")
        rag_ctx = f"\nKey guideline: {ctx}\n"  # Короткий контекст!
    prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report.{rag_ctx}\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}<end_of_turn>\n<start_of_turn>model\n"
    image = Image.open(r["image_path"]).convert("RGB").resize((448,448))
    inputs = processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def consistency_score(test, hyps):
    correct = 0
    total = 0
    for r, hyp in zip(test, hyps):
        true_b = get_true_birads(r["breast_birads"])
        pred_b = extract_birads(hyp)
        if true_b and pred_b:
            total += 1
            if true_b == pred_b:
                correct += 1
    return correct / total if total > 0 else 0, correct, total

def main():
    print("Загружаю тестовые данные...")
    test = load_test(n=30)

    print("Загружаю модель...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)

    # Baseline
    print("\n[1/3] Baseline...")
    base.eval()
    hyps_base = [gen(base, processor, r, use_rag=False) for r in test]
    score_base, c, t = consistency_score(test, hyps_base)
    print(f"Baseline BI-RADS consistency: {score_base:.3f} ({c}/{t})")

    # Multimodal FT без RAG
    print("\n[2/3] Multimodal FT (no RAG)...")
    mm = PeftModel.from_pretrained(base, MM_LORA); mm.eval()
    hyps_mm = [gen(mm, processor, r, use_rag=False) for r in test]
    score_mm, c, t = consistency_score(test, hyps_mm)
    print(f"Multimodal FT BI-RADS consistency: {score_mm:.3f} ({c}/{t})")

    # Multimodal FT + RAG (k=1, короткий контекст)
    print("\n[3/3] Multimodal FT + RAG (k=1)...")
    hyps_rag = [gen(mm, processor, r, use_rag=True) for r in test]
    score_rag, c, t = consistency_score(test, hyps_rag)
    print(f"Multimodal FT + RAG BI-RADS consistency: {score_rag:.3f} ({c}/{t})")

    results = {
        "baseline": {"birads_consistency": round(score_base, 4)},
        "multimodal_ft": {"birads_consistency": round(score_mm, 4)},
        "multimodal_ft_rag_k1": {"birads_consistency": round(score_rag, 4)},
    }

    print(f"\n{'='*50}")
    print("BI-RADS Consistency Summary")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:30s}: {v['birads_consistency']:.3f}")

    out = "./birads_consistency.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено в {out}")

if __name__ == "__main__":
    main()
