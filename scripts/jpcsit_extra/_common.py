"""
Common utilities for JPCSIT extra experiments.
Place in: new_article/scripts/jpcsit_extra/_common.py

Reuses logic from rag_postprocess.py.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import faiss
import pickle
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ============================================================
# Paths (match your existing project structure)
# ============================================================
BASE = Path("/mnt/c/Users/juman/hard_ml/rag_mammo/new_article")

MODEL_ID = "google/medgemma-4b-it"
MM_LORA = BASE / "medgemma_multimodal"
CHUNKS_FILE = BASE / "data" / "rag" / "birads_chunks.json"
META_FILE = BASE / "data" / "rag" / "birads_meta.pkl"
VINDR_JSONL = BASE / "vindr/synthetic_reports.jsonl"
VINDR_IMG = BASE / "vindr"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

JPCSIT_RESULTS = BASE / "jpcsit_results"
JPCSIT_RESULTS.mkdir(exist_ok=True)


# ============================================================
# Test data
# ============================================================
def load_test(n=30):
    """Same as in rag_postprocess.py — last n valid VinDr examples."""
    data = []
    with open(VINDR_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("is_valid"):
                    continue
                img_path = VINDR_IMG / r["study_id"] / f"{r['image_id']}.png"
                if img_path.exists():
                    r["image_path"] = str(img_path)
                    data.append(r)
            except Exception:
                pass
    return data[-n:]


# ============================================================
# Knowledge base
# ============================================================
def load_chunks():
    with open(CHUNKS_FILE) as f:
        return json.load(f)


# ============================================================
# Multimodal model (MedGemma + LoRA) — singleton
# ============================================================
_mm_cache = {"model": None, "processor": None}


def get_mm_model():
    if _mm_cache["model"] is None:
        print("Loading MedGemma + multimodal LoRA (one-time)...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, device_map="auto", dtype=torch.bfloat16
        )
        mm = PeftModel.from_pretrained(base, str(MM_LORA))
        mm.eval()
        _mm_cache["model"] = mm
        _mm_cache["processor"] = processor
    return _mm_cache["model"], _mm_cache["processor"]


def gen_mm_draft(model, processor, r):
    """Generate draft report from image (same as rag_postprocess.py)."""
    prompt = (
        f"<start_of_turn>user\n<start_of_image>\n"
        f"Generate a structured mammography report.\n"
        f"Findings: {r['finding_categories']}\n"
        f"BI-RADS: {r['breast_birads']}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    image = Image.open(r["image_path"]).convert("RGB").resize((448, 448))
    inputs = processor(
        text=prompt, images=image, return_tensors="pt",
        truncation=True, max_length=768
    ).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ============================================================
# RAG refine via Ollama (same as rag_postprocess.py)
# ============================================================
def rag_refine(draft, context):
    prompt = (
        "You are a radiologist. Refine this mammography report draft using "
        "the clinical guidelines below.\n"
        "Keep the same structure but improve clinical accuracy and terminology.\n\n"
        f"Clinical guidelines:\n{context}\n\n"
        f"Draft report:\n{draft}\n\n"
        "Refined report (same length, improved terminology):"
    )
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    return r.json()["response"].strip()


# ============================================================
# Metrics (same as rag_postprocess.py)
# ============================================================
def compute_metrics(refs, hyps):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25, .25, .25, .25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    _, _, F = bert_score(hyps, refs, lang="en", verbose=False)
    return {
        "bleu1": round(bleu1, 4),
        "bleu4": round(bleu4, 4),
        "rouge1": round(sum(r1) / len(r1), 4),
        "rouge2": round(sum(r2) / len(r2), 4),
        "rougeL": round(sum(rl) / len(rl), 4),
        "bertscore": round(F.mean().item(), 4),
    }


# ============================================================
# Hallucination rate (simple heuristic — same idea as your existing code)
# ============================================================
def hallucination_rate(hyps, refs):
    """Fraction of generated reports that mention findings not in reference.
    Heuristic: look for finding-keywords in hyp that are absent in ref."""
    keywords = [
        "spiculated", "irregular", "calcification", "mass",
        "asymmetry", "distortion", "lymphadenopathy", "retraction",
        "circumscribed", "lobulated", "fibroadenoma",
    ]
    hallucinated = 0
    for hyp, ref in zip(hyps, refs):
        h_low, r_low = hyp.lower(), ref.lower()
        for kw in keywords:
            if kw in h_low and kw not in r_low:
                hallucinated += 1
                break
    return round(hallucinated / len(hyps), 4)


# ============================================================
# Driver: run RAG on the test set with given (encoder, index, k)
# ============================================================
def run_rag_pipeline(test_data, drafts, encoder, index, chunks, k, refine=True):
    """Returns (hyps, retrieve_latencies_ms, refine_latencies_s)."""
    hyps = []
    retrieve_lat = []
    refine_lat = []
    for r, draft in zip(test_data, drafts):
        # Build query
        query = (
            f"BI-RADS {r['breast_birads']} "
            f"{r['finding_categories']} mammography recommendation"
        )
        # Retrieval
        t0 = time.perf_counter()
        emb = encoder.encode([query]).astype("float32")
        faiss.normalize_L2(emb)
        _, ids = index.search(emb, k)
        retrieve_lat.append((time.perf_counter() - t0) * 1000.0)
        ctx = " ".join(chunks[i]["text"] for i in ids[0] if 0 <= i < len(chunks))

        # Refine
        if refine:
            t0 = time.perf_counter()
            try:
                refined = rag_refine(draft, ctx)
            except Exception as e:
                print(f"[warn] refine failed: {e}; using draft.")
                refined = draft
            refine_lat.append(time.perf_counter() - t0)
            hyps.append(refined)
        else:
            hyps.append(draft)
            refine_lat.append(0.0)
        print(".", end="", flush=True)
    print()
    return hyps, retrieve_lat, refine_lat


# ============================================================
# Save helper
# ============================================================
def save_json(obj, fname):
    path = JPCSIT_RESULTS / fname
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved -> {path}")
