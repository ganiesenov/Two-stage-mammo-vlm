"""
Инференс с сохранением по-сэмплового вывода (GPU).

Зачем: в исходных прогонах генерации не сохранялись — в results/ лежат только
агрегированные метрики и 5 качественных примеров. Из-за этого нельзя было
посчитать доверительные интервалы (R1.9, R2.2), density/BI-RADS/галлюцинации
на всех 52 случаях (R3.8, R1.3) и вообще ничего перепроверить.

Запуск:
  python 01_infer.py --split test --models zero_shot dmid_only_r16 two_stage_r16 two_stage_r64
  python 01_infer.py --split val  --models ablation_all      # выбор конфига по валидации (R1.13, R3.2)
  python 01_infer.py --split test --models two_stage_r64 --rag   # абляция RAG (R1.7, R3.5)

Все модели гоняются в одинаковых условиях (4-bit NF4, 448px, greedy) — иначе
строки таблицы несопоставимы, что и было главной проблемой Table 4.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(__file__))
import _common
from _common import (ROOT, split_files, load_pairs, load_model, free,
                     generate_all, save_preds, corpus_scores, PROMPTS)

ABL = f"{ROOT}/lora_ablation"

# Лучший чекпойнт каждой конфигурации — по eval_loss из trainer_state.json.
# Верхнеуровневых адаптеров у конфигураций абляции нет: lora_ablation.py
# не вызывал trainer.save_model(), сохранялись только пер-эпоховые чекпойнты.
REGISTRY = {
    # baselines
    "zero_shot":      {"stage1": False, "adapter": None},
    "dmid_only_r16":  {"stage1": False, "adapter": f"{ROOT}/medgemma_dmid_only"},
    "two_stage_r16":  {"stage1": False, "adapter": f"{ROOT}/medgemma_dmid"},
    "dmid_only_r64":  {"stage1": False, "adapter": f"{ROOT}/dmid_only_r64"},   # обучается в 02_
    # конфигурации абляции LoRA (Stage 1 влит в базу, поверх адаптер Stage 2)
    # r16_a32 переобучена тем же механизмом в 15_train_r16_ablation.py: в исходной
    # абляции эта строка была захардкожена из прогона по другому механизму.
    "two_stage_r16_a32":  {"stage1": True, "adapter": f"{ABL}/r16_a32"},
    "two_stage_r8_a16":   {"stage1": True, "adapter": f"{ABL}/r8_a16/checkpoint-204"},
    "two_stage_r8_a32":   {"stage1": True, "adapter": f"{ABL}/r8_a32/checkpoint-204"},
    "two_stage_r32_a32":  {"stage1": True, "adapter": f"{ABL}/r32_a32/checkpoint-204"},
    "two_stage_r32_a64":  {"stage1": True, "adapter": f"{ABL}/r32_a64/checkpoint-204"},
    "two_stage_r64":      {"stage1": True, "adapter": f"{ABL}/r64_a64/checkpoint-204"},
    "two_stage_r64_a128": {"stage1": True, "adapter": f"{ABL}/r64_a128/checkpoint-153"},
}

ABLATION_ALL = ["two_stage_r8_a16", "two_stage_r8_a32", "two_stage_r16_a32",
                "two_stage_r32_a32", "two_stage_r32_a64", "two_stage_r64",
                "two_stage_r64_a128"]


def make_retriever(k=1):
    """BI-RADS RAG, top-k. В статье заявлен top-1 (Figure 1)."""
    import faiss, json as _json
    from sentence_transformers import SentenceTransformer
    idx = faiss.read_index(f"{ROOT}/data/rag/birads_faiss.index")
    chunks = _json.load(open(f"{ROOT}/data/rag/birads_chunks.json"))
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", list(chunks.values()))
    texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    query = ("mammography breast composition ACR density findings "
             "BI-RADS assessment category recommendation")
    emb = enc.encode([query]).astype("float32")
    faiss.normalize_L2(emb)
    _, ids = idx.search(emb, k)
    ctx = "\n".join(texts[i] for i in ids[0] if 0 <= i < len(texts))
    print(f"  RAG: retrieved {k} chunk(s), {len(ctx)} chars")
    return lambda pair: ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rag", action="store_true", help="включить BI-RADS RAG в промпт")
    ap.add_argument("--rag-k", type=int, default=1)
    ap.add_argument("--prompt", choices=["A", "B"], default="B",
                    help="A — промпт обучения моделей r=16; B — промпт Table 4/Table 8")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-тест: только N первых случаев, результат не сохраняется")
    args = ap.parse_args()

    names = []
    for m in args.models:
        names.extend(ABLATION_ALL if m == "ablation_all" else [m])

    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        sys.exit(f"Неизвестные модели: {unknown}\nДоступно: {list(REGISTRY)}")

    pairs = load_pairs(split_files()[args.split])
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Сплит {args.split}: {len(pairs)} случаев"
          + ("  [SMOKE TEST — не сохраняется]" if args.limit else ""))

    _common.PROMPT_TEXT = PROMPTS[args.prompt]
    print(f"Промпт {args.prompt}: {_common.PROMPT_TEXT[:70]}...")

    retriever = make_retriever(args.rag_k) if args.rag else None
    suffix = f"_rag{args.rag_k}" if args.rag else ""
    if args.prompt != "B":
        suffix += f"_prompt{args.prompt}"

    summary = {}
    for name in names:
        spec = REGISTRY[name]
        if spec["adapter"] and not os.path.exists(spec["adapter"]):
            print(f"\n[{name}] пропуск — нет адаптера {spec['adapter']}")
            continue

        print(f"\n{'='*68}\n  {name}{suffix} → {args.split}\n{'='*68}", flush=True)
        model, processor = load_model(spec)
        preds = generate_all(model, processor, pairs, retriever=retriever,
                             log_every=1 if args.limit else 10)
        if args.limit:
            for p in preds:
                print(f"  [{p['id']}] REF: {p['ref'][:110]}")
                print(f"  {' '*(len(p['id'])+3)}HYP: {p['hyp'][:110]}")
            free(model, processor)
            continue
        scores = corpus_scores(preds)
        save_preds(name + suffix, args.split, preds,
                   meta={"spec": spec, "rag": bool(args.rag),
                         "rag_k": args.rag_k if args.rag else None,
                         "corpus_scores": scores})
        summary[name + suffix] = scores
        print(f"  {scores}")
        free(model, processor)

    if args.limit:
        print("\nSmoke test пройден — можно запускать полный прогон.")
        return

    out = f"{ROOT}/results/revision/scores_{args.split}{suffix}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(summary)
    with open(out, "w") as f:
        json.dump(prev, f, indent=2)

    print(f"\n{'='*68}\n  ИТОГ ({args.split}{suffix})\n{'='*68}")
    print(f"  {'model':<24s} {'BLEU-4':>8s} {'R-1':>8s} {'R-L':>8s} {'MET':>8s} {'CIDEr':>8s}")
    for k, v in prev.items():
        cid = v.get("cider")
        print(f"  {k:<24s} {v['bleu4']:>8.4f} {v['rouge1']:>8.4f} {v['rougeL']:>8.4f} "
              f"{v['meteor']:>8.4f} {(f'{cid:.4f}' if cid is not None else '—'):>8s}")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
