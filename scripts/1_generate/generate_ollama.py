import pandas as pd
import json
import time
import ast
import requests
from pathlib import Path

FINDING_CSV = "./vindr/finding_annotations.csv"
OUTPUT_FILE = "./vindr/synthetic_reports.jsonl"
MAX_REPORTS = 200
OLLAMA_MODEL = "llama3.1:8b"

def parse_finding_categories(cat_str):
    try:
        return ast.literal_eval(cat_str)
    except:
        return []

def build_prompt(row):
    birads = str(row.get("breast_birads","")).replace("BI-RADS ","")
    density = str(row.get("breast_density","")).replace("DENSITY ","")
    laterality = "left" if row.get("laterality") == "L" else "right"
    view = "craniocaudal (CC)" if row.get("view_position") == "CC" else "mediolateral oblique (MLO)"
    findings = parse_finding_categories(str(row.get("finding_categories","[]")))
    findings_str = ", ".join(findings) if findings and findings != ["No Finding"] else "no pathological findings"

    return f"""You are an experienced radiologist. Write a short structured mammography report.

Study data:
- Side: {laterality} breast, view: {view}
- Breast density: ACR category {density}
- Findings: {findings_str}
- BI-RADS category: {birads}

Requirements:
1. Structure: Density → Findings → BI-RADS → Recommendation
2. Standard radiological terminology
3. 4-6 sentences
4. Report text only, no extra words"""

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
                    existing.add(r["study_id"] + r["laterality"])
                except: pass
    return existing

def main():
    print("="*50)
    print(f"Ollama Report Generator — {OLLAMA_MODEL}")
    print("="*50)

    existing = load_existing()
    if existing:
        print(f"Resume: найдено {len(existing)} готовых отчётов")

    df = pd.read_csv(FINDING_CSV)
    df = df[df["split"] == "training"]
    df_findings = df[df["finding_categories"] != "['No Finding']"]
    df_normal = df[df["finding_categories"] == "['No Finding']"].sample(min(30, len(df)), random_state=42)
    df_todo = pd.concat([df_findings, df_normal])\
                .drop_duplicates(subset=["study_id","laterality"])\
                .head(MAX_REPORTS)
    df_todo = df_todo[~df_todo.apply(lambda r: r["study_id"]+r["laterality"] in existing, axis=1)]

    print(f"Осталось: {len(df_todo)} отчётов\n")

    for count, (_, row) in enumerate(df_todo.iterrows(), start=len(existing)+1):
        print(f"[{count:3d}/{MAX_REPORTS}] {row['breast_birads']:10s} | {str(row['finding_categories'])[:30]:30s}", end=" → ")
        try:
            report = generate(build_prompt(row))
            is_valid, issues = validate(report)
            result = {
                "study_id": row["study_id"], "image_id": row["image_id"],
                "laterality": row["laterality"], "view_position": row["view_position"],
                "breast_birads": row["breast_birads"], "breast_density": row["breast_density"],
                "finding_categories": row["finding_categories"],
                "synthetic_report": report, "is_valid": is_valid,
                "validation_issues": issues, "dataset": "vindr"
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print("✓" if is_valid else f"⚠ {issues}")
        except Exception as e:
            print(f"ОШИБКА: {e}")

    print("\nГОТОВО!")

if __name__ == "__main__":
    main()
