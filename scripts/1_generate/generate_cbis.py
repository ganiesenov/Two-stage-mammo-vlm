import pandas as pd
import json
import requests
from pathlib import Path

MASS_CSV = "./cbis-ddsm/csv/mass_case_description_train_set.csv"
CALC_CSV = "./cbis-ddsm/csv/calc_case_description_train_set.csv"
OUTPUT_FILE = "./cbis-ddsm/synthetic_reports_cbis.jsonl"
OLLAMA_MODEL = "llama3.1:8b"
MAX_REPORTS = 200

DENSITY_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}

def build_prompt_mass(row):
    birads = str(row.get("assessment", ""))
    density = DENSITY_MAP.get(int(row.get("breast_density", 2)), "B")
    laterality = "left" if str(row.get("left or right breast", "")).upper() == "LEFT" else "right"
    view = str(row.get("image view", "CC"))
    shape = str(row.get("mass shape", "")).lower()
    margins = str(row.get("mass margins", "")).lower()
    pathology = str(row.get("pathology", "")).lower()

    return f"""You are an experienced radiologist. Write a short structured mammography report.

Study data:
- Side: {laterality} breast, view: {view}
- Breast density: ACR category {density}
- Finding: mass, shape: {shape}, margins: {margins}
- Pathology: {pathology}
- BI-RADS category: {birads}

Requirements:
1. Structure: Density → Findings → BI-RADS → Recommendation
2. Standard radiological terminology
3. 4-6 sentences
4. Report text only"""

def build_prompt_calc(row):
    birads = str(row.get("assessment", ""))
    density = DENSITY_MAP.get(int(row.get("breast_density", 2)), "B")
    laterality = "left" if str(row.get("left or right breast", "")).upper() == "LEFT" else "right"
    view = str(row.get("image view", "CC"))
    morphology = str(row.get("calc type", "")).lower()
    distribution = str(row.get("calc distribution", "")).lower()

    return f"""You are an experienced radiologist. Write a short structured mammography report.

Study data:
- Side: {laterality} breast, view: {view}
- Breast density: ACR category {density}
- Finding: calcification, morphology: {morphology}, distribution: {distribution}
- BI-RADS category: {birads}

Requirements:
1. Structure: Density → Findings → BI-RADS → Recommendation
2. Standard radiological terminology
3. 4-6 sentences
4. Report text only"""

def generate(prompt):
    r = requests.post("http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120)
    return r.json()["response"].strip()

def validate(text):
    issues = []
    t = text.lower()
    if "bi-rads" not in t: issues.append("missing_birads")
    if not any(k in t for k in ["recommend","follow","biopsy","routine","additional"]):
        issues.append("missing_recommendation")
    if len(text.split()) < 25: issues.append("too_short")
    return len(issues) == 0, issues

def load_existing():
    existing = set()
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing.add(r["patient_id"] + r["image_view"])
                except: pass
    return existing

def main():
    print("="*50)
    print("CBIS-DDSM Report Generator")
    print("="*50)

    existing = load_existing()
    if existing:
        print(f"Resume: {len(existing)} готовых отчётов")

    mass_df = pd.read_csv(MASS_CSV).dropna(subset=["assessment","breast_density"])
    calc_df = pd.read_csv(CALC_CSV).rename(columns={"breast density": "breast_density"}).dropna(subset=["assessment","breast_density"])

    mass_df["_type"] = "mass"
    calc_df["_type"] = "calc"

    # Унифицируем колонки
    mass_df["patient_id"] = mass_df["patient_id"]
    mass_df["image_view"] = mass_df["image view"]
    calc_df["patient_id"] = calc_df["patient_id"]
    calc_df["image_view"] = calc_df["image view"]

    df = pd.concat([mass_df, calc_df]).drop_duplicates(
        subset=["patient_id","image_view"]).head(MAX_REPORTS)
    df = df[~df.apply(lambda r: r["patient_id"]+str(r["image_view"]) in existing, axis=1)]

    print(f"Осталось: {len(df)} отчётов\n")

    count = len(existing)
    for _, row in df.iterrows():
        count += 1
        print(f"[{count:3d}/{MAX_REPORTS}] BI-RADS {row['assessment']} | {row['_type']:4s} | {str(row.get('mass shape', row.get('calc type','')))[:20]:20s}", end=" → ")
        try:
            prompt = build_prompt_mass(row) if row["_type"] == "mass" else build_prompt_calc(row)
            report = generate(prompt)
            is_valid, issues = validate(report)
            result = {
                "patient_id": row["patient_id"],
                "image_view": str(row["image_view"]),
                "laterality": str(row.get("left or right breast","")),
                "breast_density": str(row.get("breast_density","")),
                "assessment": str(row.get("assessment","")),
                "finding_type": row["_type"],
                "pathology": str(row.get("pathology","")),
                "synthetic_report": report,
                "is_valid": is_valid,
                "validation_issues": issues,
                "dataset": "cbis-ddsm"
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print("✓" if is_valid else f"⚠ {issues}")
        except Exception as e:
            print(f"ОШИБКА: {e}")

    print("\nГОТОВО!")

if __name__ == "__main__":
    main()
