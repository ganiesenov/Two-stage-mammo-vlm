import json
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID  = "google/medgemma-4b-it"
LORA_DIR  = "./medgemma_finetuned"
INDEX_FILE = "./birads_faiss.index"
META_FILE  = "./birads_meta.pkl"

# RAG retriever
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=3):
    emb = encoder.encode([query]).astype("float32")
    faiss.normalize_L2(emb)
    _, ids = index.search(emb, k)
    return " ".join(chunks[i]["text"] for i in ids[0])

def build_prompt(report, use_rag=False):
    findings = report.get("finding_categories", report.get("finding_type", "mass"))
    birads   = report.get("breast_birads", report.get("assessment", ""))
    density  = report.get("breast_density", "")
    laterality = report.get("laterality", "")

    rag_context = ""
    if use_rag:
        query = f"BI-RADS {birads} {findings} mammography"
        context = retrieve(query)
        rag_context = f"\nClinical guidelines:\n{context}\n"

    user_msg = f"""Generate a structured mammography report.{rag_context}
Findings: {findings}
BI-RADS: {birads}
Density: {density}
Laterality: {laterality}"""

    return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

def generate(model, tokenizer, prompt, max_new=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# Тест
if __name__ == "__main__":
    print("Загружаю модель...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    test = {"finding_categories": "['Mass']", "breast_birads": "BI-RADS 4",
            "breast_density": "DENSITY C", "laterality": "L"}

    print("\n--- Fine-tuned БЕЗ RAG ---")
    print(generate(model, tokenizer, build_prompt(test, use_rag=False)))

    print("\n--- Fine-tuned + RAG ---")
    print(generate(model, tokenizer, build_prompt(test, use_rag=True)))
