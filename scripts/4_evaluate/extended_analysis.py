"""
Extended analysis: per-BIRADS breakdown, density accuracy, 
report length analysis — computed from existing predictions
"""
import os, json, re
import numpy as np

RESULTS = "./results/eval_dmid_full.json"
DMID_REPS = "./dmid/Reports/Reports/"

def extract_birads(text):
    patterns = [r'BI-?RADS[\s:]*(\d)', r'BIRADS[\s:]*(\d)', r'CATEGORY\s*(\d)']
    for p in patterns:
        m = re.search(p, text.upper())
        if m: return int(m.group(1))
    return None

def extract_density(text):
    patterns = [r'ACR\s*([A-D])', r'acr\s*([a-dA-D])']
    for p in patterns:
        m = re.search(p, text)
        if m: return m.group(1).upper()
    return None

def main():
    with open(RESULTS) as f:
        data = json.load(f)
    
    examples = data["qualitative_examples"]
    
    # Load ALL test reports (last 52)
    report_files = sorted(os.listdir(DMID_REPS))
    test_files = report_files[-52:]
    
    all_refs = []
    for rf in test_files:
        with open(os.path.join(DMID_REPS, rf), encoding='utf-8', errors='ignore') as f:
            all_refs.append(f.read().strip())
    
    # We need the generated reports too — get from examples if available
    # Only 5 examples saved, so we work with what we have for qualitative,
    # but compute stats on all refs for BI-RADS distribution
    
    # ── 1. BI-RADS Distribution in test set ──
    print("="*60)
    print("  BI-RADS Distribution in DMID Test Set (n=52)")
    print("="*60)
    birads_dist = {}
    for ref in all_refs:
        b = extract_birads(ref)
        birads_dist[b] = birads_dist.get(b, 0) + 1
    for k in sorted(birads_dist.keys(), key=lambda x: x or 99):
        print(f"  BI-RADS {k}: {birads_dist[k]} ({birads_dist[k]/52*100:.1f}%)")
    
    # ── 2. Density Distribution ──
    print("\n" + "="*60)
    print("  Density Distribution in DMID Test Set")
    print("="*60)
    density_dist = {}
    for ref in all_refs:
        d = extract_density(ref)
        density_dist[d] = density_dist.get(d, 0) + 1
    for k in sorted(density_dist.keys(), key=lambda x: x or 'Z'):
        print(f"  ACR {k}: {density_dist[k]} ({density_dist[k]/52*100:.1f}%)")
    
    # ── 3. Density accuracy from qualitative examples ──
    print("\n" + "="*60)
    print("  Density Prediction (from 5 qualitative examples)")
    print("="*60)
    for variant in ["dmid_only", "two_stage"]:
        correct = 0
        total = 0
        for ex in examples:
            ref_d = extract_density(ex["real_report"])
            gen_d = extract_density(ex[variant])
            if ref_d:
                total += 1
                if ref_d == gen_d:
                    correct += 1
                    print(f"  {variant:12s} | {ex['img_id']}: {gen_d} == {ref_d} ✓")
                else:
                    print(f"  {variant:12s} | {ex['img_id']}: {gen_d} != {ref_d} ✗")
        print(f"  {variant}: {correct}/{total} correct")
    
    # ── 4. Report length analysis ──
    print("\n" + "="*60)
    print("  Report Length Analysis (words)")
    print("="*60)
    ref_lens = [len(r.split()) for r in all_refs]
    print(f"  Real reports:    mean={np.mean(ref_lens):.1f}, std={np.std(ref_lens):.1f}, "
          f"min={min(ref_lens)}, max={max(ref_lens)}")
    
    for variant in ["zero_shot", "dmid_only", "two_stage"]:
        if variant in ["zero_shot"]:
            # estimate from examples
            lens = [len(ex[variant].split()) for ex in examples]
        else:
            lens = [len(ex[variant].split()) for ex in examples]
        print(f"  {variant:15s}: mean={np.mean(lens):.1f}, std={np.std(lens):.1f}, "
              f"min={min(lens)}, max={max(lens)}")
    
    # ── 5. BI-RADS prediction from examples ──
    print("\n" + "="*60)
    print("  BI-RADS Prediction (from 5 qualitative examples)")
    print("="*60)
    for variant in ["zero_shot", "dmid_only", "two_stage"]:
        correct = 0
        total = 0
        for ex in examples:
            ref_b = extract_birads(ex["real_report"])
            gen_b = extract_birads(ex[variant])
            if ref_b is not None:
                total += 1
                match = "✓" if ref_b == gen_b else "✗"
                print(f"  {variant:12s} | {ex['img_id']}: pred={gen_b} real={ref_b} {match}")
                if ref_b == gen_b:
                    correct += 1
        print(f"  {variant}: {correct}/{total}\n")
    
    # ── 6. Full dataset stats for paper ──
    print("\n" + "="*60)
    print("  Dataset Statistics for Paper")
    print("="*60)
    
    # VinDr stats (from paper)
    print("  VinDr-Mammo: 5,000 exams, 20,000 images")
    print("    BI-RADS: 1-5, Density: ACR A-D")
    print("    200 synthetic reports generated")
    
    print("  CBIS-DDSM: 1,566 patients")
    print("    Finding types: mass, calcification")
    print("    200 synthetic reports generated")
    
    print(f"  DMID: 225 cases, 510 images")
    print(f"    Split: 407 train / 51 val / 52 test")
    print(f"    BI-RADS distribution (test):")
    for k in sorted(birads_dist.keys(), key=lambda x: x or 99):
        print(f"      BI-RADS {k}: {birads_dist[k]}")
    
    # Save extended results
    extended = {
        "birads_distribution_test": {str(k): v for k, v in birads_dist.items()},
        "density_distribution_test": density_dist,
        "report_length_real": {"mean": round(np.mean(ref_lens), 1), "std": round(np.std(ref_lens), 1)},
    }
    
    out = "./results/extended_analysis.json"
    with open(out, "w") as f:
        json.dump(extended, f, indent=2)
    print(f"\nСохранено в {out}")

if __name__ == "__main__":
    main()
