"""
BERTScore по сохранённым генерациям (CPU/GPU, после 01_infer.py).

Отвечает на R3.3: в Table 4 колонка BERTScore собрана из разных прогонов —
значение 0.917 в строке two-stage взято от модели r=16, а не от строки r=64,
о чём в подписи таблицы стояла оговорка «BERTScore from prior evaluation runs
where available». Здесь колонка считается одним протоколом для всех строк.

Backbone — roberta-large (умолчание bert_score для lang="en"), rescale_with_baseline
не применяется: исходные числа статьи считались без него, иначе строки разойдутся
не из-за модели, а из-за шкалы.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR

OUT_DIR = f"{ROOT}/results/revision"


def load_pairs(name, split, keep_ids=None):
    path = f"{PRED_DIR}/{name}__{split}.json"
    if not os.path.exists(path):
        return None
    preds = json.load(open(path, encoding="utf-8"))["predictions"]
    if keep_ids is not None:
        preds = [p for p in preds if p["id"] in keep_ids]
    return [p["hyp"] for p in preds], [p["ref"] for p in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    keep, tag = None, ""
    if args.clean_only:
        man = json.load(open(f"{OUT_DIR}/split_manifest.json"))
        keep = set(man["clean_test_ids"])
        tag = "_clean"

    from bert_score import score as bert_score

    out = {"split": args.split, "clean_only": args.clean_only,
           "backbone": "roberta-large", "rescale_with_baseline": False, "scores": {}}

    for m in args.models:
        pair = load_pairs(m, args.split, keep)
        if pair is None:
            print(f"  пропуск {m} — нет предсказаний")
            continue
        hyps, refs = pair
        P, R, F = bert_score(hyps, refs, lang="en", verbose=False)
        out["scores"][m] = {"n": len(hyps),
                            "precision": round(float(P.mean()), 4),
                            "recall": round(float(R.mean()), 4),
                            "f1": round(float(F.mean()), 4),
                            "f1_std": round(float(F.std()), 4)}
        print(f"  {m:<22s} n={len(hyps):<3d} F1={float(F.mean()):.4f}")

    path = f"{OUT_DIR}/bertscore_{args.split}{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
