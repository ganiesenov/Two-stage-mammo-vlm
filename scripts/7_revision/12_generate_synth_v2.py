"""
Генерация синтетического корпуса v2 для дозозависимой абляции Stage 1.

Отличия от v1 (scripts/1_generate/generate_ollama.py, generate_cbis.py):

  1. ЯЗЫКОВОЙ GUARD В ЦИКЛЕ. v1 генерировал один раз и проверял три строковых
     индикатора; 21 из 200 отчётов VinDr вышли на русском при англоязычном промпте
     и прошли валидацию (5.4% корпуса Stage 1, см. FINDINGS.md п. 6). Здесь наличие
     кириллицы — брак: отчёт перегенерируется с усиленной инструкцией, и только при
     исчерпании попыток кандидат отбрасывается и берётся следующий. Постфильтрация
     не используется намеренно: она уменьшала бы N вместе с чисткой и путала бы
     два фактора в одном сравнении.

  2. ВАЛИДНОСТЬ ТОЖЕ ЧЕРЕЗ ПЕРЕГЕНЕРАЦИЮ. В v1 брак просто оставался в файле с
     is_valid=False (10 из 400). Здесь принимаются только валидные отчёты, поэтому
     размер корпуса — ровно заданное N, а не «сколько получилось».

  3. ВЛОЖЕННЫЕ ДОЗЫ. Кандидаты чередуются VinDr/CBIS, генерация идёт строго по
     порядку, дозы берутся ПРЕФИКСАМИ пула: 369 ⊂ 800 ⊂ 1550. Любые две дозы
     отличаются только объёмом — источники 50/50 и доля находок одинаковы во всех.

  Промпты скопированы из v1 дословно (кроме добавки при перегенерации), иначе дозы
  были бы несопоставимы с исходным корпусом.

ПОТОЛОК ОБЪЁМА. В training-сплите VinDr ровно 775 уникальных study×laterality с
находками, а текущий корпус на стороне VinDr состоит из находок на 100% (No Finding = 0:
исходный скрипт делал .head(200) после конкатенации, и нормальные случаи не попали).
Поэтому при фиксированном составе 50/50 максимум — 775+775 = 1550. Заявленные в спеке
1600 потребовали бы добора нормальными случаями, то есть одновременного изменения
N и баланса классов — ровно того спутывания, которое эта работа устраняет.

Запуск:
    nohup python3 scripts/7_revision/12_generate_synth_v2.py > logs/gen_v2.log 2>&1 &
Возобновление: повторный запуск дочитывает готовое из выходного файла и продолжает.
"""
import os, sys, json, re, time, argparse
import pandas as pd
import requests

ROOT = "."
OUT = f"{ROOT}/data/synthetic_v2.jsonl"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TARGET_TOTAL = 1444          # 722 VinDr + 722 CBIS-mass (потолок при составе v1)
DOSES = [369, 800, 1444]     # вложенные префиксы пула
MAX_RETRIES = 4              # попыток на кандидата до отбрасывания

CYR = re.compile(r'[а-яА-ЯёЁ]')
DENSITY_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


# ─────────────────────── промпты (дословно из v1) ───────────────────────

def prompt_vindr(row):
    return f"""You are an experienced radiologist. Write a short structured mammography report.

Study data:
- Side: {'left' if str(row['laterality']).upper().startswith('L') else 'right'} breast, view: {row['view_position']}
- Breast density: ACR category {str(row['breast_density']).replace('DENSITY ', '')}
- Findings: {row['finding_categories']}
- BI-RADS category: {row['breast_birads']}

Requirements:
1. Structure: Density → Findings → BI-RADS → Recommendation
2. Standard radiological terminology
3. 4-6 sentences
4. Report text only, no extra words"""


def _density(row):
    """Устойчиво к обоим написаниям колонки: mass — 'breast_density', calc — 'breast density'."""
    for col in ("breast_density", "breast density"):
        v = row.get(col)
        if v is not None and pd.notna(v):
            try:
                return DENSITY_MAP.get(int(v), "B")
            except (TypeError, ValueError):
                pass
    return "B"


def prompt_cbis_mass(row):
    return f"""You are an experienced radiologist. Write a short structured mammography report.

Study data:
- Side: {'left' if str(row.get('left or right breast', '')).upper() == 'LEFT' else 'right'} breast, view: {row.get('image view', 'CC')}
- Breast density: ACR category {_density(row)}
- Finding: mass, shape: {str(row.get('mass shape', '')).lower()}, margins: {str(row.get('mass margins', '')).lower()}
- Pathology: {str(row.get('pathology', '')).lower()}
- BI-RADS category: {row.get('assessment', '')}

Requirements:
1. Structure: Density → Findings → BI-RADS → Recommendation
2. Standard radiological terminology
3. 4-6 sentences
4. Report text only"""


def prompt_cbis_calc(row):
    return f"""You are an experienced radiologist. Write a short structured mammography report.

Study data:
- Side: {'left' if str(row.get('left or right breast', '')).upper() == 'LEFT' else 'right'} breast, view: {row.get('image view', 'CC')}
- Breast density: ACR category {_density(row)}
- Finding: calcification, morphology: {str(row.get('calc type', '')).lower()}, distribution: {str(row.get('calc distribution', '')).lower()}
- BI-RADS category: {row.get('assessment', '')}

Requirements:
1. Structure: Density → Findings → BI-RADS → Recommendation
2. Standard radiological terminology
3. 4-6 sentences
4. Report text only"""


LANG_REINFORCE = ("\n\nIMPORTANT: Write the report in ENGLISH only. "
                  "Do not use Russian or any other language.")


# ─────────────────────── проверки ───────────────────────

def is_english(text):
    """Кириллица в отчёте — брак. Именно эта проверка отсутствовала в v1."""
    return CYR.search(text) is None


def is_valid(text):
    """Три индикатора v1 — сохранены дословно ради сопоставимости корпусов."""
    t = text.lower()
    if "bi-rads" not in t:
        return False, "missing_birads"
    if not any(k in t for k in ["recommend", "follow", "biopsy", "routine", "additional"]):
        return False, "missing_recommendation"
    if len(text.split()) < 25:
        return False, "too_short"
    return True, None


def call_ollama(prompt):
    r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt,
                                        "stream": False}, timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()


# ─────────────────────── кандидаты ───────────────────────

def vindr_candidates():
    """Уникальные study×laterality с находками из training-сплита, с проверкой файла."""
    df = pd.read_csv(f"{ROOT}/vindr/finding_annotations.csv")
    df = df[(df["split"] == "training") & (df["finding_categories"] != "['No Finding']")]
    df = df.drop_duplicates(subset=["study_id", "laterality"])
    df = df.sort_values(["study_id", "laterality"])          # детерминированный порядок
    out = []
    for _, r in df.iterrows():
        p = f"{ROOT}/vindr/{r['study_id']}/{r['image_id']}.png"
        if os.path.exists(p):
            out.append({"source": "vindr", "img": p, "row": r.to_dict(),
                        "key": f"vindr:{r['study_id']}:{r['laterality']}"})
    return out


def cbis_candidates():
    """Только mass из train-сплита, уникальные patient×laterality, с проверкой jpeg.

    Почему не добавлены кальцинаты. Опубликованный корпус v1 на стороне CBIS состоит
    из mass на 100% (200/200): generate_cbis.py обрабатывает mass первым и упирается
    в MAX_REPORTS=200 до того, как доходит до calc. Добавление кальцинатов в v2 меняло бы
    вместе с объёмом ещё и тип находок, то есть снова смешивало бы два фактора.

    Побочно обнаружен латентный баг v1: в calc-таблице колонка называется
    'breast density' (с пробелом), а generate_cbis.py:20 читает 'breast_density',
    поэтому для кальцинатов сработал бы дефолт 2 → 'B' в 68.6% случаев вопреки
    истинной плотности. В опубликованный корпус это не попало ровно потому, что
    кальцинаты не генерировались. Чтение колонки ниже сделано устойчивым к обоим
    вариантам названия — на случай, если calc-ветка когда-нибудь понадобится.
    """
    cm = pd.read_csv(f"{ROOT}/cbis-ddsm/image_mapping.csv")
    df = pd.read_csv(f"{ROOT}/cbis-ddsm/csv/mass_case_description_train_set.csv")
    df["_kind"] = "mass"
    df = df.drop_duplicates(subset=["patient_id", "left or right breast"])
    df = df.sort_values(["patient_id", "left or right breast", "_kind"])
    out = []
    for _, r in df.iterrows():
        lat = str(r["left or right breast"]).strip().upper()[:1]
        view = str(r.get("image view", "CC")).strip().upper()
        m = cm[(cm["patient_id"] == r["patient_id"])
               & (cm["laterality"].astype(str).str.upper().str.startswith(lat))
               & (cm["view"].astype(str).str.upper() == view)]
        if len(m) and os.path.exists(str(m.iloc[0]["jpeg_path"])):
            out.append({"source": "cbis", "img": str(m.iloc[0]["jpeg_path"]),
                        "row": r.to_dict(), "kind": r["_kind"],
                        "key": f"cbis:{r['patient_id']}:{lat}"})
    return out


def interleave(v, c, total):
    """Чередование источников: любой префикс пула остаётся близким к 50/50."""
    half = total // 2
    v, c = v[:half], c[:total - half]
    out = []
    for i in range(max(len(v), len(c))):
        if i < len(v):
            out.append(v[i])
        if i < len(c):
            out.append(c[i])
    return out[:total]


# ─────────────────────── основной цикл ───────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_TOTAL)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)

    print("Сбор кандидатов…", flush=True)
    vs, cs = vindr_candidates(), cbis_candidates()
    print(f"  VinDr с находками и картинкой: {len(vs)}")
    print(f"  CBIS с картинкой:              {len(cs)}")

    ceiling = 2 * min(len(vs), len(cs))
    target = min(args.target, ceiling)
    if target < args.target:
        print(f"  ПОТОЛОК: при составе 50/50 доступно {ceiling}; цель снижена "
              f"{args.target} → {target}", flush=True)

    pool = interleave(vs, cs, target)
    print(f"  пул кандидатов: {len(pool)} "
          f"(vindr {sum(1 for p in pool if p['source']=='vindr')}, "
          f"cbis {sum(1 for p in pool if p['source']=='cbis')})", flush=True)

    done = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done[r["key"]] = r
            except Exception:
                pass
        print(f"  возобновление: уже готово {len(done)}", flush=True)

    fh = open(OUT, "a")
    n_ok = len(done)
    n_lang_retry = n_val_retry = n_dropped = 0
    t0 = time.time()

    for i, cand in enumerate(pool, 1):
        if cand["key"] in done:
            continue
        if cand["source"] == "vindr":
            base = prompt_vindr(cand["row"])
        else:
            base = (prompt_cbis_mass if cand["kind"] == "mass" else prompt_cbis_calc)(cand["row"])

        text = None
        for attempt in range(MAX_RETRIES):
            p = base + (LANG_REINFORCE if attempt else "")
            try:
                t = call_ollama(p)
            except Exception as e:
                print(f"    ollama error: {e}", flush=True)
                time.sleep(5)
                continue
            if not is_english(t):
                n_lang_retry += 1
                continue
            ok, _ = is_valid(t)
            if not ok:
                n_val_retry += 1
                continue
            text = t
            break

        if text is None:
            n_dropped += 1
            continue

        rec = {"key": cand["key"], "source": cand["source"], "img": cand["img"],
               "synthetic_report": text, "order": i}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        n_ok += 1
        if n_ok % 25 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{n_ok}/{target}] {el:.0f} мин | перегенераций: язык "
                  f"{n_lang_retry}, валидность {n_val_retry} | отброшено {n_dropped}",
                  flush=True)

    fh.close()

    # ── сводка и определение доз ──
    recs = [json.loads(l) for l in open(OUT)]
    recs.sort(key=lambda r: r["order"])
    n_cyr = sum(1 for r in recs if not is_english(r["synthetic_report"]))
    summary = {
        "generator": OLLAMA_MODEL,
        "n_total": len(recs),
        "n_vindr": sum(1 for r in recs if r["source"] == "vindr"),
        "n_cbis": sum(1 for r in recs if r["source"] == "cbis"),
        "n_russian_remaining": n_cyr,
        "regenerations_language": n_lang_retry,
        "regenerations_validity": n_val_retry,
        "dropped_after_retries": n_dropped,
        "doses": {},
    }
    for d in DOSES:
        if d <= len(recs):
            sub = recs[:d]
            summary["doses"][str(d)] = {
                "n": d,
                "vindr": sum(1 for r in sub if r["source"] == "vindr"),
                "cbis": sum(1 for r in sub if r["source"] == "cbis"),
            }
    with open(f"{ROOT}/results/revision/synth_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 60)
    if n_cyr:
        print(f"ВНИМАНИЕ: в итоговом корпусе осталось {n_cyr} русскоязычных — это баг guard'а")
    else:
        print("Языковая контаминация: 0 (было 21/200 в v1)")


if __name__ == "__main__":
    main()
