"""
Experiment 3 — FAISS index comparison.
Place in: new_article/scripts/jpcsit_extra/exp3_faiss.py

Compares three FAISS index types on the BI-RADS knowledge base:
  - IndexFlatIP   (exact, baseline)
  - IndexIVFFlat  (inverted file, approximate)
  - IndexHNSWFlat (hierarchical NSW graph, approximate)

For each index:
  - Build time, index size in memory
  - Retrieval latency (mean, p95) over 1000 random queries
  - Recall@k=2 vs the FlatIP gold standard
  - End-to-end NLG metrics on the 30 VinDr test cases

NOTE: With only ~50–60 fragments in the BI-RADS knowledge base, IVF/HNSW
have NO practical advantage — they exist for million-scale corpora. The
purpose of this experiment is to characterize the speed/recall trade-off
*if* the knowledge base were scaled up (which is exactly what the paper
proposes as future work). This is a methodological experiment, not a
production claim.
"""
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from _common import (
    load_test, load_chunks, get_mm_model, gen_mm_draft,
    run_rag_pipeline, compute_metrics, hallucination_rate, save_json,
)


ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
N_LATENCY_QUERIES = 1000  # synthetic queries for latency measurement


def build_indices(emb):
    """Return dict of index_name -> (faiss_index, build_seconds)."""
    d = emb.shape[1]
    out = {}

    # 1) Flat (exact, baseline)
    t = time.perf_counter()
    flat = faiss.IndexFlatIP(d)
    flat.add(emb)
    out["FlatIP"] = (flat, time.perf_counter() - t)

    # 2) IVFFlat (approximate)
    # nlist must be much smaller than n; for tiny corpora this is degenerate
    nlist = max(2, min(8, emb.shape[0] // 8))
    t = time.perf_counter()
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(emb)
    ivf.add(emb)
    ivf.nprobe = max(1, nlist // 2)
    out["IVFFlat"] = (ivf, time.perf_counter() - t)

    # 3) HNSWFlat (approximate)
    M = 16  # neighbors per layer
    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 40
    hnsw.hnsw.efSearch = 16
    hnsw.add(emb)
    out["HNSWFlat"] = (hnsw, time.perf_counter() - t)

    return out


def measure_latency(idx, emb, n=1000, k=2):
    """Run n random queries from the embedding pool, return mean & p95 in ms."""
    rng = np.random.default_rng(42)
    queries = emb[rng.integers(0, emb.shape[0], size=n)].copy()
    # Warm-up
    idx.search(queries[:5], k)
    times = []
    for q in queries:
        q = q.reshape(1, -1).astype("float32")
        t0 = time.perf_counter()
        idx.search(q, k)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": round(float(np.mean(times)), 4),
        "p50_ms": round(float(np.percentile(times, 50)), 4),
        "p95_ms": round(float(np.percentile(times, 95)), 4),
        "p99_ms": round(float(np.percentile(times, 99)), 4),
    }


def measure_recall(approx_idx, flat_idx, queries, k=2):
    """Recall@k vs FlatIP gold standard."""
    _, gold = flat_idx.search(queries, k)
    _, pred = approx_idx.search(queries, k)
    correct = 0
    total = 0
    for g, p in zip(gold, pred):
        gset = set(int(x) for x in g if x >= 0)
        pset = set(int(x) for x in p if x >= 0)
        correct += len(gset & pset)
        total += len(gset)
    return round(correct / max(total, 1), 4)


def main():
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    chunks = load_chunks()
    print(f"Test set: {len(test)} | Knowledge base: {len(chunks)} fragments")

    # Encode the knowledge base ONCE
    print("\nEncoding knowledge base...")
    encoder = SentenceTransformer(ENCODER_ID)
    texts = [c["text"] for c in chunks]
    emb = encoder.encode(texts, show_progress_bar=False, batch_size=32)
    emb = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(emb)

    # Build all indices
    print("\nBuilding all indices...")
    indices = build_indices(emb)

    # Latency test (synthetic queries from KB itself)
    print("\nMeasuring latency on synthetic queries...")
    flat_for_recall = indices["FlatIP"][0]
    rng = np.random.default_rng(42)
    qs = emb[rng.integers(0, emb.shape[0], size=N_LATENCY_QUERIES)].copy()

    perf = {}
    for name, (idx, build_t) in indices.items():
        lat = measure_latency(idx, emb, n=N_LATENCY_QUERIES, k=2)
        recall = (
            1.0 if name == "FlatIP"
            else measure_recall(idx, flat_for_recall, qs, k=2)
        )
        perf[name] = {
            "build_seconds": round(build_t, 4),
            "n_vectors": idx.ntotal,
            "embedding_dim": emb.shape[1],
            "recall_at_k=2": recall,
            **lat,
        }
        print(f"  {name}: build={build_t:.3f}s, "
              f"latency_mean={lat['mean_ms']:.3f}ms, recall={recall}")

    # End-to-end NLG metrics
    print("\nGenerating drafts (multimodal MedGemma+LoRA)...")
    mm, processor = get_mm_model()
    drafts = []
    for r in test:
        drafts.append(gen_mm_draft(mm, processor, r))
        print(".", end="", flush=True)
    print(f"\nGenerated {len(drafts)} drafts.")

    e2e = {}
    for name, (idx, _) in indices.items():
        print(f"\n{'=' * 70}\n  E2E with {name}\n{'=' * 70}")
        hyps, ret_lat, ref_lat = run_rag_pipeline(
            test, drafts, encoder, idx, chunks, k=2, refine=True
        )
        m = compute_metrics(refs, hyps)
        m["hallucination_rate"] = hallucination_rate(hyps, refs)
        m["retrieve_latency_ms_mean"] = round(float(np.mean(ret_lat)), 3)
        m["refine_latency_s_mean"] = round(float(np.mean(ref_lat)), 3)
        e2e[name] = m
        print(f"  Metrics: {m}")

    out = {"index_performance": perf, "end_to_end_nlg": e2e}
    save_json(out, "exp3_faiss.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
