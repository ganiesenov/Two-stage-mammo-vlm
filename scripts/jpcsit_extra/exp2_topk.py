"""
Experiment 2 — Top-k retrieval sweep.
Place in: new_article/scripts/jpcsit_extra/exp2_topk.py

Uses the BEST encoder from Exp1 (default: all-MiniLM-L6-v2 — change after Exp1
finishes if a different encoder wins).

Sweeps k ∈ {1, 2, 3, 5, 10} — measures how the amount of injected BI-RADS
context affects:
  - NLG metrics (BLEU/ROUGE/METEOR/BERTScore)
  - hallucination rate
  - prompt length (context tokens)

Time estimate: 5 values of k × 30 generations × Ollama refine ≈ 30–60 min.
"""
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from _common import (
    load_test, load_chunks, get_mm_model, gen_mm_draft,
    run_rag_pipeline, compute_metrics, hallucination_rate, save_json,
)


# Set to the winner of Exp1; default keeps current baseline encoder
BEST_ENCODER = "pritamdeka/S-PubMedBert-MS-MARCO"
K_VALUES = [1, 2, 3, 5, 10]


def build_index(encoder, chunks):
    texts = [c["text"] for c in chunks]
    emb = encoder.encode(texts, show_progress_bar=False, batch_size=32)
    emb = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


def avg_context_tokens(test, encoder, idx, chunks, k):
    """Estimate average number of whitespace-tokens injected for a given k."""
    lens = []
    for r in test:
        q = (
            f"BI-RADS {r['breast_birads']} "
            f"{r['finding_categories']} mammography recommendation"
        )
        emb = encoder.encode([q]).astype("float32")
        faiss.normalize_L2(emb)
        _, ids = idx.search(emb, k)
        ctx = " ".join(chunks[i]["text"] for i in ids[0] if 0 <= i < len(chunks))
        lens.append(len(ctx.split()))
    return float(np.mean(lens))


def main():
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    chunks = load_chunks()
    print(f"Test set: {len(test)} | Knowledge base: {len(chunks)} fragments")

    # Generate drafts ONCE
    print("\n[Step 0] Generating drafts (multimodal MedGemma+LoRA)...")
    mm, processor = get_mm_model()
    drafts = []
    for r in test:
        drafts.append(gen_mm_draft(mm, processor, r))
        print(".", end="", flush=True)
    print(f"\nGenerated {len(drafts)} drafts.")

    encoder = SentenceTransformer(BEST_ENCODER)
    idx = build_index(encoder, chunks)
    print(f"\nEncoder: {BEST_ENCODER} | dim={idx.d}")

    results = {}
    for k in K_VALUES:
        print(f"\n{'=' * 70}\n  k = {k}\n{'=' * 70}")
        ctx_words = avg_context_tokens(test, encoder, idx, chunks, k)
        print(f"  avg context length: {ctx_words:.1f} words")

        hyps, ret_lat, ref_lat = run_rag_pipeline(
            test, drafts, encoder, idx, chunks, k=k, refine=True
        )
        m = compute_metrics(refs, hyps)
        m["hallucination_rate"] = hallucination_rate(hyps, refs)
        m["avg_context_words"] = round(ctx_words, 1)
        m["retrieve_latency_ms_mean"] = round(float(np.mean(ret_lat)), 3)
        m["refine_latency_s_mean"] = round(float(np.mean(ref_lat)), 3)
        m["k"] = k
        print(f"  Metrics: {m}")

        results[f"k={k}"] = m
        save_json(results, "exp2_topk.json")

    print("\nDone. Final results:")
    print(results)


if __name__ == "__main__":
    main()
