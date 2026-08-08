"""
Experiment 1 — Embedding model comparison for BI-RADS retrieval.
Place in: new_article/scripts/jpcsit_extra/exp1_embeddings.py

Compares 4 sentence encoders:
  - all-MiniLM-L6-v2          (general, current baseline)
  - all-mpnet-base-v2         (general, larger)
  - pritamdeka/S-PubMedBert-MS-MARCO  (biomedical, sentence-level)
  - NeuML/pubmedbert-base-embeddings  (biomedical, alternative)

For each encoder:
  1) Re-encode the BI-RADS knowledge base
  2) Build a FAISS flat IP index
  3) Run the post-hoc RAG pipeline on the same 30 VinDr test cases
  4) Compute NLG metrics + hallucination rate + retrieval latency

Time estimate on RTX 5090: ~4 encoders × (load + 30 generations + 30 Ollama refines)
Expect ~10–20 minutes per encoder depending on Ollama latency.
"""
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from _common import (
    load_test, load_chunks, get_mm_model, gen_mm_draft,
    run_rag_pipeline, compute_metrics, hallucination_rate, save_json,
)


ENCODERS = [
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
    ("S-PubMedBert", "pritamdeka/S-PubMedBert-MS-MARCO"),
    ("PubMedBert-base", "NeuML/pubmedbert-base-embeddings"),
]


def build_index(encoder, chunks):
    texts = [c["text"] for c in chunks]
    emb = encoder.encode(texts, show_progress_bar=False, batch_size=32)
    emb = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


def main():
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    chunks = load_chunks()
    print(f"Test set: {len(test)} | Knowledge base: {len(chunks)} fragments")

    # Generate drafts ONCE — does not depend on the encoder
    print("\n[Step 0] Generating drafts (multimodal MedGemma+LoRA, no RAG)...")
    mm, processor = get_mm_model()
    drafts = []
    for r in test:
        drafts.append(gen_mm_draft(mm, processor, r))
        print(".", end="", flush=True)
    print(f"\nGenerated {len(drafts)} drafts.")

    results = {}
    for nick, hf_id in ENCODERS:
        print(f"\n{'=' * 70}\n  Encoder: {nick}  ({hf_id})\n{'=' * 70}")
        t_start = time.perf_counter()
        encoder = SentenceTransformer(hf_id)
        idx = build_index(encoder, chunks)
        build_t = time.perf_counter() - t_start
        print(f"  Index built in {build_t:.1f}s  | dim={idx.d} | n={idx.ntotal}")

        hyps, ret_lat, ref_lat = run_rag_pipeline(
            test, drafts, encoder, idx, chunks, k=2, refine=True
        )
        m = compute_metrics(refs, hyps)
        m["hallucination_rate"] = hallucination_rate(hyps, refs)
        m["retrieve_latency_ms_mean"] = round(float(np.mean(ret_lat)), 3)
        m["retrieve_latency_ms_p95"] = round(float(np.percentile(ret_lat, 95)), 3)
        m["refine_latency_s_mean"] = round(float(np.mean(ref_lat)), 3)
        m["embedding_dim"] = idx.d
        m["index_build_seconds"] = round(build_t, 2)

        # Free GPU memory between encoders
        del encoder
        import torch
        torch.cuda.empty_cache()

        print(f"  Metrics: {m}")
        results[nick] = m
        # Save partial after each encoder so a crash doesn't lose everything
        save_json(results, "exp1_embeddings.json")

    print("\nDone. Final results:")
    print(results)


if __name__ == "__main__":
    main()
