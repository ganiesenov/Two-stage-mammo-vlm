"""
Обработка результатов чтения радиологами (CPU).

Три задачи:
  1. Оценки по осям — то, что просят R1.10 / R2.4 / R3.4.
  2. Межчитательское согласие: взвешенная κ для порядковых осей (1–4)
     и обычная κ для бинарной оси пропуска. Без этого reader study не принимают.
  3. ГЛАВНОЕ — валидация автоматической метрики пропусков человеком.
     Автоматика показала 35% пропусков у лучшей модели; если радиологи размечают
     те же случаи, метрика перестаёт быть эвристикой и становится измеренной.

Принимает и JSON из интерфейса, и заполненные CSV.
Запуск: python 11_reader_agreement.py reader_study/reader_1_results.json ...
"""
import os, sys, json, csv, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR

OUT = f"{ROOT}/results/revision"
KEY = f"{ROOT}/reader_study/_key.json"
AXES = ["grammar", "findings", "impression", "birads", "overall"]


def load_results(paths):
    readers = defaultdict(dict)
    for p in paths:
        if p.endswith(".json"):
            for row in json.load(open(p, encoding="utf-8")):
                readers[str(row.get("reader", os.path.basename(p)))][row["uid"]] = row
        else:
            rid = os.path.basename(p).split(".")[0]
            for row in csv.DictReader(open(p, encoding="utf-8")):
                if any(row.get(a) for a in AXES):
                    readers[rid][row["uid"]] = row
    return readers


def num(v):
    try:
        return int(v)
    except Exception:
        return None


def kappa(a, b, weighted=False):
    """Cohen's κ; weighted=True — квадратично взвешенная (для порядковых шкал)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None, 0
    labels = sorted({v for p in pairs for v in p})
    n = len(pairs)
    if len(labels) < 2:
        return None, n
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    O = [[0] * k for _ in range(k)]
    for x, y in pairs:
        O[idx[x]][idx[y]] += 1
    ra = [sum(O[i]) for i in range(k)]
    ca = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[ra[i] * ca[j] / n for j in range(k)] for i in range(k)]

    def w(i, j):
        if not weighted:
            return 0.0 if i == j else 1.0
        return ((labels[i] - labels[j]) ** 2) / ((labels[-1] - labels[0]) ** 2)

    num_ = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * E[i][j] for i in range(k) for j in range(k))
    return (None if den == 0 else 1 - num_ / den), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--auto-model", default="two_stage_r64",
                    help="модель, чью автоматическую метрику пропусков сверяем")
    args = ap.parse_args()

    if not os.path.exists(KEY):
        sys.exit(f"Нет {KEY} — сначала 10_reader_study.py")
    key = json.load(open(KEY, encoding="utf-8"))["mapping"]
    readers = load_results(args.results)
    if not readers:
        sys.exit("Пустые результаты")

    print("=" * 74)
    print(f"  Reader study: читателей {len(readers)}")
    print("=" * 74)

    out = {"n_readers": len(readers), "by_reader": {}, "agreement": {}, "omission_validation": {}}

    # ── 1. оценки по осям ──
    for rid, rows in readers.items():
        done = [r for r in rows.values() if num(r.get("overall")) is not None]
        print(f"\n  Читатель {rid}: оценено {len(done)} из {len(key)}")
        stats = {}
        for a in AXES:
            vals = [num(r.get(a)) for r in rows.values() if num(r.get(a)) is not None]
            if vals:
                m = sum(vals) / len(vals)
                sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** .5
                stats[a] = {"mean": round(m, 2), "sd": round(sd, 2), "n": len(vals)}
                print(f"    {a:<11s} {m:.2f} ± {sd:.2f}  (n={len(vals)})")
        om = [str(r.get("omission", "")).lower() for r in rows.values()]
        y = sum(1 for v in om if v == "yes")
        tot = sum(1 for v in om if v in ("yes", "no"))
        if tot:
            stats["omission_rate"] = round(y / tot, 4)
            print(f"    пропуск находки: {y}/{tot} = {y/tot*100:.1f}%")
        out["by_reader"][rid] = stats

    # ── 2. межчитательское согласие ──
    rids = list(readers)
    if len(rids) >= 2:
        a_id, b_id = rids[0], rids[1]
        common = sorted(set(readers[a_id]) & set(readers[b_id]))
        print(f"\n{'='*74}\n  Согласие между читателями {a_id} и {b_id} (общих позиций {len(common)})\n{'='*74}")
        for a in AXES:
            va = [num(readers[a_id][u].get(a)) for u in common]
            vb = [num(readers[b_id][u].get(a)) for u in common]
            kw, n = kappa(va, vb, weighted=True)
            exact = sum(1 for x, y in zip(va, vb) if x is not None and x == y)
            within1 = sum(1 for x, y in zip(va, vb)
                          if x is not None and y is not None and abs(x - y) <= 1)
            print(f"    {a:<11s} взвеш. κ {('%.3f' % kw) if kw is not None else '—':>7s}"
                  f"   точное {exact}/{n}   ±1 {within1}/{n}")
            out["agreement"][a] = {"weighted_kappa": round(kw, 3) if kw is not None else None,
                                   "exact": exact, "within_1": within1, "n": n}
        oa = [str(readers[a_id][u].get("omission", "")).lower() or None for u in common]
        ob = [str(readers[b_id][u].get("omission", "")).lower() or None for u in common]
        ko, n = kappa(oa, ob)
        print(f"    {'omission':<11s} κ {('%.3f' % ko) if ko is not None else '—':>7s}  (n={n})")
        out["agreement"]["omission"] = {"kappa": round(ko, 3) if ko is not None else None, "n": n}

    # ── 2b. согласие разметки снимка с эталонным отчётом DMID ──
    # Показывает, сколько «пропусков» на деле является расхождением между радиологами,
    # а не ошибкой модели: автоматическая метрика считает эталоном ОТЧЁТ автора DMID,
    # тогда как читатель размечает СНИМОК независимо и до показа заключения.
    auto_path = f"{ROOT}/results/revision/clinical_metrics_test.json"
    ref_pos = set()
    if os.path.exists(auto_path):
        cm_all = json.load(open(auto_path, encoding="utf-8")).get(args.auto_model, {})
        f_ = cm_all.get("findings", {})
        # изображения, где эталонный отчёт утверждает находку
        ref_pos = ({c["id"] for c in f_.get("omission_cases", [])}
                   | {c["id"] for c in f_.get("hallucination_cases", [])})
    if ref_pos:
        print(f"\n{'='*74}\n  Разметка снимка читателями против эталонного отчёта DMID\n{'='*74}")
        a_ = []
        b_ = []
        for uid, meta in key.items():
            votes = [str(readers[r][uid].get("s1_finding") or readers[r][uid].get("s1finding") or "").lower()
                     for r in readers if uid in readers[r]]
            votes = [v for v in votes if v in ("yes", "no")]
            if not votes:
                continue
            a_.append("yes" if votes.count("yes") * 2 >= len(votes) else "no")
            b_.append("yes" if meta["image"] in ref_pos else "no")
        if a_:
            kk, nn = kappa(a_, b_)
            agree = sum(1 for x, y in zip(a_, b_) if x == y)
            print(f"    совпадений {agree}/{nn}   κ {('%.3f' % kk) if kk is not None else '—'}")
            print("    Низкое согласие здесь означает, что часть «пропусков» —")
            print("    расхождение между радиологами, а не ошибка модели.")
            out["reader_vs_dmid_reference"] = {
                "n": nn, "agree": agree,
                "kappa": round(kk, 3) if kk is not None else None}

    # ── 3. валидация автоматической метрики пропусков ──
    if os.path.exists(auto_path):
        cm = json.load(open(auto_path, encoding="utf-8")).get(args.auto_model, {})
        auto_missed = {c["id"] for c in cm.get("findings", {}).get("omission_cases", [])}
        if auto_missed:
            print(f"\n{'='*74}\n  Валидация автоматической метрики пропусков "
                  f"({args.auto_model})\n{'='*74}")
            print("    Золотой стандарт — независимая разметка снимка (шаг 1), сделанная")
            print("    ДО показа заключения; суждение о пропуске (шаг 2) берётся как")
            print("    подтверждение. Эталонный отчёт DMID золотым стандартом НЕ является.")
            tp = fp = fn = tn = 0
            for uid, meta in key.items():
                if meta["model"] != args.auto_model:
                    continue
                # человеческий пропуск = читатель увидел находку на снимке (шаг 1)
                # И отметил её отсутствие в заключении (шаг 2)
                hv = []
                for r in readers:
                    if uid not in readers[r]:
                        continue
                    row = readers[r][uid]
                    s1 = str(row.get("s1_finding") or row.get("s1finding") or "").lower()
                    om = str(row.get("omission", "")).lower()
                    if s1 in ("yes", "no") and om in ("yes", "no"):
                        hv.append("yes" if (s1 == "yes" and om == "yes") else "no")
                if not hv:
                    continue
                human = "yes" if hv.count("yes") * 2 >= len(hv) else "no"
                auto = meta["image"] in auto_missed
                if human == "yes" and auto: tp += 1
                elif human == "no" and auto: fp += 1
                elif human == "yes" and not auto: fn += 1
                else: tn += 1
            n = tp + fp + fn + tn
            if n:
                acc = (tp + tn) / n
                sens = tp / (tp + fn) if (tp + fn) else None
                spec = tn / (tn + fp) if (tn + fp) else None
                kv, _ = kappa(["yes"] * tp + ["no"] * fp + ["yes"] * fn + ["no"] * tn,
                              ["yes"] * tp + ["yes"] * fp + ["no"] * fn + ["no"] * tn)
                print(f"    согласовано {n} случаев")
                print(f"    TP {tp}  FP {fp}  FN {fn}  TN {tn}")
                print(f"    точность {acc:.3f}   чувствительность "
                      f"{('%.3f' % sens) if sens is not None else '—'}   "
                      f"специфичность {('%.3f' % spec) if spec is not None else '—'}")
                print(f"    κ между автоматикой и радиологами: "
                      f"{('%.3f' % kv) if kv is not None else '—'}")
                out["omission_validation"] = {
                    "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "accuracy": round(acc, 3),
                    "sensitivity": round(sens, 3) if sens is not None else None,
                    "specificity": round(spec, 3) if spec is not None else None,
                    "kappa_auto_vs_human": round(kv, 3) if kv is not None else None,
                }

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/reader_study_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  → {OUT}/reader_study_results.json")


if __name__ == "__main__":
    main()
