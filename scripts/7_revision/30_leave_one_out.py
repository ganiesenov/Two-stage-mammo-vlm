"""
Leave-one-out по тестовым случаям (R3.5, R1.2; CPU).

При n=49 один нетипичный случай может заметно сдвинуть корпусную метрику. Для четырёх
fine-tuned моделей Table 4 пересчитываем каждую headline-метрику без i-го случая и
берём max_i |metric(−i) − metric|. То же для парной разности ROUGE-L между схемами
при r=16 и r=64 и для клинических метрик Table 6 (чувствительность, BI-RADS exact).

BERTScore считается заново по-сэмплово на CPU (roberta-large, без rescale — как в
17_bertscore.py): в bertscore_test_clean.json сохранены только средние.
"""
import os, sys, json, importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, load_preds, corpus_scores, per_sample_scores, extract_birads

spec = importlib.util.spec_from_file_location(
    "clin", os.path.join(os.path.dirname(__file__), "04_clinical_metrics.py"))
clin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clin)

MODELS = ["dmid_only_r16", "two_stage_r16", "dmid_only_r64", "two_stage_r64"]
TEXT = ["bleu1", "bleu4", "rouge1", "rougeL", "meteor", "cider"]
OUT = f"{ROOT}/results/revision/leave_one_out.json"


def bertscore_per_sample(preds):
    from bert_score import score as bert_score
    P, R, F = bert_score([p["hyp"] for p in preds], [p["ref"] for p in preds],
                         lang="en", verbose=False, device="cpu", batch_size=16)
    return F.numpy()


def clinical(preds):
    tp = fn = 0
    ex = n = 0
    for p in preds:
        ra, _ = clin.asserts_finding(p["ref"])
        ha, _ = clin.asserts_finding(p["hyp"])
        if ra:
            tp += ha
            fn += not ha
        r, h = extract_birads(p["ref"]), extract_birads(p["hyp"])
        if r is not None and h is not None:
            n += 1
            ex += r == h
    return {"sensitivity": tp / (tp + fn), "birads_exact": ex / n}


def main():
    man = json.load(open(f"{ROOT}/results/revision/split_manifest.json"))
    leaked = set(man["verbatim_duplicate_leakage"]["test_ids"])
    out = {"n": None, "models": {}, "paired_rougeL_delta": {}}
    per_rl = {}
    for m in MODELS:
        preds = [p for p in load_preds(m, "test") if p["id"] not in leaked]
        n = out["n"] = len(preds)
        full = {**corpus_scores(preds), **clinical(preds)}
        bs = bertscore_per_sample(preds)
        full["bertscore"] = float(bs.mean())
        per_rl[m] = {r["id"]: r["rougeL"] for r in per_sample_scores(preds)}

        shifts = {k: [] for k in TEXT + ["sensitivity", "birads_exact", "bertscore"]}
        for i in range(n):
            sub = preds[:i] + preds[i + 1:]
            cs = {**corpus_scores(sub), **clinical(sub)}
            cs["bertscore"] = float(np.delete(bs, i).mean())
            for k in shifts:
                shifts[k].append(abs(cs[k] - full[k]))
            print(f"  {m} {i + 1}/{n}", end="\r", flush=True)
        out["models"][m] = {k: {"full": round(full[k], 6),
                                "max_abs_shift": round(max(v), 4),
                                "argmax_id": preds[int(np.argmax(v))]["id"]}
                            for k, v in shifts.items()}
        print(f"\n{m}: " + "  ".join(f"{k} {d['max_abs_shift']:.4f}"
                                      for k, d in out["models"][m].items()))

    for r in ("r16", "r64"):
        a, b = per_rl[f"two_stage_{r}"], per_rl[f"dmid_only_{r}"]
        d = np.array([a[i] - b[i] for i in a])
        full = d.mean()
        shifts = [abs(np.delete(d, i).mean() - full) for i in range(len(d))]
        out["paired_rougeL_delta"][r] = {"full": round(float(full), 4),
                                         "max_abs_shift": round(float(max(shifts)), 4)}
        print(f"ΔROUGE-L {r}: {full:+.4f}, max LOO shift {max(shifts):.4f}")
    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
