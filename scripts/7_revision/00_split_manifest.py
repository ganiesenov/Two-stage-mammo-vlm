"""
Манифест сплита DMID + анализ утечек (CPU).

Отвечает на:
  R1.1 — «дайте прямые ссылки на точные парные отчёты» → полный список Img-ID по сплитам
  R3.1 — «сплит по пациенту или по изображению?» → честный ответ + масштаб проблемы
  R1.5 — «пересечение пациентов синтетика/реальные» → синтетика из VinDr/CBIS,
         пересечение с DMID невозможно; внутри DMID меряем дословные дубликаты

DMID не публикует идентификаторов пациентов, связывающих 510 изображений с 225 случаями,
поэтому «сплит по пациентам» технически недостижим. Ближайший измеримый прокси —
группы изображений с дословно совпадающим текстом отчёта.

Результат: results/revision/split_manifest.json + supplementary_split.csv
"""
import os, re, json, csv
from collections import defaultdict, Counter

import sys
sys.path.insert(0, os.path.dirname(__file__))
from _common import REPS_DIR, ROOT, split_files

OUT_DIR = f"{ROOT}/results/revision"


def norm(t):
    return re.sub(r'\s+', ' ', t).strip().lower()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    splits = split_files()
    files = sorted(os.listdir(REPS_DIR))

    texts = {}
    for f in files:
        with open(os.path.join(REPS_DIR, f), encoding='utf-8', errors='ignore') as fh:
            texts[f] = norm(fh.read())

    where = {f: s for s, fs in splits.items() for f in fs}

    groups = defaultdict(list)
    for f, t in texts.items():
        groups[t].append(f)
    groups = {i: sorted(g) for i, g in enumerate(groups.values())}

    print("=" * 68)
    print("  DMID: структура сплита")
    print("=" * 68)
    print(f"  всего файлов отчётов : {len(files)}  (Img001..Img510, 1 отчёт на изображение)")
    print(f"  train / val / test   : {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"  уникальных текстов   : {len(groups)}")
    print(f"  сплит                : по изображениям (протокол AMRG), НЕ по пациентам")
    print(f"  причина              : DMID не публикует patient ID для 225 случаев")

    sizes = Counter(len(g) for g in groups.values())
    print(f"\n  распределение размеров групп дословных дубликатов:")
    for k in sorted(sizes):
        print(f"    группы из {k:2d} отчёт(ов): {sizes[k]}")

    # утечка: группа, чьи файлы попали и в test, и в train/val
    leaks = []
    for gid, g in groups.items():
        s = {where[f] for f in g}
        if "test" in s and len(s) > 1:
            leaks.append({
                "group_id": gid,
                "test_files": [f.replace('.txt', '') for f in g if where[f] == "test"],
                "train_files": [f.replace('.txt', '') for f in g if where[f] == "train"],
                "val_files": [f.replace('.txt', '') for f in g if where[f] == "val"],
                "report_preview": texts[g[0]][:160],
            })

    contaminated = sorted({f for L in leaks for f in L["test_files"]})

    print("\n" + "=" * 68)
    print("  Утечка: тестовые отчёты, чей текст дословно есть в train/val")
    print("=" * 68)
    print(f"  затронуто тестовых случаев: {len(contaminated)}/{len(splits['test'])} "
          f"({len(contaminated)/len(splits['test'])*100:.1f}%)")
    for L in leaks:
        print(f"    {', '.join(L['test_files'])}  ←  train: {', '.join(L['train_files']) or '—'}"
              f"  val: {', '.join(L['val_files']) or '—'}")
        print(f"      «{L['report_preview'][:110]}...»")

    print(f"\n  Для sensitivity analysis: чистое подмножество теста = "
          f"{len(splits['test']) - len(contaminated)} случаев")

    # Table 7 статьи показывает Img462 как «near-perfect generation»
    flagged = [c for c in contaminated if c in {"Img461", "Img462", "Img463"}]
    if flagged:
        print(f"\n  ВНИМАНИЕ: качественные примеры Table 7 в списке заражённых: {flagged}")
        print(f"  Их нужно заменить примерами из чистого подмножества.")

    manifest = {
        "dataset": "DMID",
        "split_protocol": "image-level, following AMRG (arXiv:2508.09225)",
        "patient_level_possible": False,
        "patient_id_note": ("DMID publishes 225 cases / 510 images but no image→patient "
                            "mapping; patient-level splitting is not reconstructable."),
        "counts": {k: len(v) for k, v in splits.items()},
        "n_unique_report_texts": len(groups),
        "verbatim_duplicate_leakage": {
            "n_test_affected": len(contaminated),
            "test_ids": contaminated,
            "groups": leaks,
        },
        "clean_test_ids": [f.replace('.txt', '') for f in splits["test"]
                           if f.replace('.txt', '') not in contaminated],
        "synthetic_overlap": {
            "stage1_sources": ["VinDr-Mammo", "CBIS-DDSM"],
            "stage2_source": "DMID",
            "overlap_possible": False,
            "rationale": ("Stage 1 uses only VinDr-Mammo and CBIS-DDSM; Stage 2 uses only DMID. "
                          "These are independent collections from different institutions, so "
                          "patient overlap between synthetic pretraining and real fine-tuning "
                          "data is structurally impossible."),
        },
        "splits": {k: [f.replace('.txt', '') for f in v] for k, v in splits.items()},
    }
    with open(f"{OUT_DIR}/split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(f"{OUT_DIR}/supplementary_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "report_file", "split", "duplicate_group", "leaks_into_test"])
        gid_of = {fl: gid for gid, g in groups.items() for fl in g}
        for fl in files:
            gid = gid_of[fl]
            w.writerow([fl.replace('.txt', ''), f"Reports/{fl}", where[fl], gid,
                        "yes" if fl.replace('.txt', '') in contaminated else ""])

    print(f"\n  → {OUT_DIR}/split_manifest.json")
    print(f"  → {OUT_DIR}/supplementary_split.csv  (в Supplementary для R1.1)")


if __name__ == "__main__":
    main()
