import json
import torch
import faiss
import pickle
from PIL import Image
from pathlib import Path
from radgraph import F1RadGraph
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_ID    = "google/medgemma-4b-it"
MM_LORA     = "./medgemma_multimodal"
INDEX_FILE  = "./birads_faiss.index"
META_FILE   = "./birads_meta.pkl"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
VINDR_IMG   = "./vindr"

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

def gen(model, processor, r, use_rag=False):
    rag_ctx = ""
    if use_rag:
        ctx = retrieve(f"BI-RADS {r['breast_birads']} {r['finding_categories']} mammography")
        rag_ctx = f"\nKey guideline: {ctx}\n"
    prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report.{rag_ctx}\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}<end_of_turn>\n<start_of_turn>model\n"
    image = Image.open(r["image_path"]).convert("RGB").resize((448,448))
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                      truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

def compute_radgraph(refs, hyps, label):
    f1radgraph = F1RadGraph(reward_level="partial")
    scores = []
    for ref, hyp in zip(refs, hyps):
        try:
            score, _, _ = f1radgraph(hyps=[hyp], refs=[ref])
            scores.append(score)
        except:
            scores.append(0.0)
    mean_score = sum(scores) / len(scores)
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  RadGraph F1 (partial): {mean_score:.4f}")
    return round(mean_score, 4)

def main():
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    print(f"Тест: {len(test)} примеров")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)

    # 1. Baseline
    print("\n[1/3] Baseline...")
    base.eval()
    hyps_base = [gen(base, processor, r) for r in test]
    print(".", end="", flush=True)
    s1 = compute_radgraph(refs, hyps_base, "Baseline (zero-shot)")

    # 2. Multimodal FT
    print("\n[2/3] Multimodal FT...")
    mm = PeftModel.from_pretrained(base, MM_LORA); mm.eval()
    hyps_mm = [gen(mm, processor, r) for r in test]
    print(".", end="", flush=True)
    s2 = compute_radgraph(refs, hyps_mm, "Multimodal FT (no RAG)")

    # 3. Multimodal FT + RAG
    print("\n[3/3] Multimodal FT + RAG...")
    hyps_rag = [gen(mm, processor, r, use_rag=True) for r in test]
    print(".", end="", flush=True)
    s3 = compute_radgraph(refs, hyps_rag, "Multimodal FT + RAG (ours)")

    results = {
        "baseline": {"radgraph_f1": s1},
        "multimodal_ft": {"radgraph_f1": s2},
        "multimodal_ft_rag": {"radgraph_f1": s3}
    }

    print(f"\n{'='*55}")
    print("SUMMARY — RadGraph F1")
    print(f"{'='*55}")
    print(f"  Baseline:           {s1:.4f}")
    print(f"  Multimodal FT:      {s2:.4f}")
    print(f"  Multimodal FT+RAG:  {s3:.4f}")

    out = "./eval_radgraph.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено в {out}")

if __name__ == "__main__":
    main()
