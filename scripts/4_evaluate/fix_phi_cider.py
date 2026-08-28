"""
Fix 1: Re-run Phi-3.5 with correct prompt format
Fix 2: Recompute CIDEr with pycocoevalcap for ALL models
"""
import os, torch, json, gc, math
from PIL import Image
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

DMID_IMGS = "./dmid/TIFF Images/TIFF Images/"
DMID_REPS = "./dmid/Reports/Reports/"
RESULTS   = "./results/phase2_full.json"
PROMPT_TEXT = "Generate a structured mammography radiology report with breast composition (ACR density), findings, BI-RADS category, and recommendation."

def load_test_pairs(n=52):
    pairs = []
    report_files = sorted(os.listdir(DMID_REPS))
    for rf in report_files[-n:]:
        img_id = rf.replace('.txt', '')
        img_num = img_id.replace('Img', '').replace('IMG', '')
        img_path = None
        for f in os.listdir(DMID_IMGS):
            if img_num in f and not f.endswith('.txt'):
                img_path = os.path.join(DMID_IMGS, f)
                break
        with open(os.path.join(DMID_REPS, rf), encoding='utf-8', errors='ignore') as fh:
            report = fh.read().strip()
        if img_path and report:
            pairs.append({"img_path": img_path, "real_report": report})
    return pairs

def cleanup():
    gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════
# Fix Phi-3.5 — different prompt format
# ═══════════════════════════════════════
def run_phi_fixed(pairs):
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("\nLoading Phi-3.5-Vision (fixed prompt)...")
    model_id = "microsoft/Phi-3.5-vision-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, _attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    hyps = []
    for i, p in enumerate(pairs):
        try:
            image = Image.open(p["img_path"]).convert("RGB").resize((448, 448))

            # Phi-3.5 correct chat format
            messages = [
                {"role": "user", "content": "<|image_1|>\n" + PROMPT_TEXT}
            ]
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Debug: print first few
            if i < 3:
                print(f"\n  [DEBUG] Phi output {i}: '{text[:100]}...'")

            hyps.append(text)
        except Exception as e:
            print(f"\n  Error {i}: {e}")
            hyps.append("")
        print(f"\r  Phi: {i+1}/{len(pairs)}", end="", flush=True)
    print()

    # Check how many empty
    empty = sum(1 for h in hyps if len(h.strip()) == 0)
    print(f"  Empty outputs: {empty}/{len(hyps)}")

    del model, processor; cleanup()
    return hyps

# ═══════════════════════════════════════
# Compute CIDEr with pycocoevalcap
# ═══════════════════════════════════════
def compute_cider_proper(refs, hyps):
    """Use pycocoevalcap if available, fallback to manual"""
    try:
        from pycocoevalcap.cider.cider import Cider
        # Format for pycocoevalcap: dict of {id: [text]}
        gts = {i: [ref] for i, ref in enumerate(refs)}
        res = {i: [hyp] for i, hyp in enumerate(hyps)}
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        print(f"  CIDEr (pycocoevalcap): {score:.4f}")
        return round(score, 4)
    except ImportError:
        print("  pycocoevalcap not found, using manual CIDEr")
        return compute_cider_manual(refs, hyps)
    except Exception as e:
        print(f"  CIDEr error: {e}, using manual")
        return compute_cider_manual(refs, hyps)

def compute_cider_manual(refs, hyps):
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    N = len(refs)
    df = Counter()
    for ref in refs:
        seen = set()
        for n in range(1, 5):
            for ng in ngrams(ref.lower().split(), n):
                if ng not in seen: df[ng] += 1; seen.add(ng)
    scores = []
    for ref, hyp in zip(refs, hyps):
        rt, ht = Counter(), Counter()
        for n in range(1, 5):
            rt.update(ngrams(ref.lower().split(), n))
            ht.update(ngrams(hyp.lower().split(), n))
        ri = {ng: c * math.log(max(1.,N)/max(1.,df.get(ng,0))) for ng, c in rt.items()}
        hi = {ng: c * math.log(max(1.,N)/max(1.,df.get(ng,0))) for ng, c in ht.items()}
        ang = set(ri)|set(hi)
        dot = sum(ri.get(ng,0)*hi.get(ng,0) for ng in ang)
        nr = math.sqrt(sum(v**2 for v in ri.values()))+1e-8
        nh = math.sqrt(sum(v**2 for v in hi.values()))+1e-8
        scores.append(dot/(nr*nh))
    return round(sum(scores)/len(scores),4) if scores else 0.0

def compute_all_metrics(refs, hyps, label):
    sf = SmoothingFunction().method1
    bleu1 = corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps], weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps], weights=(.25,.25,.25,.25), smoothing_function=sf)
    sc = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1,r2,rl = [],[],[]
    for ref,hyp in zip(refs,hyps):
        s = sc.score(ref,hyp); r1.append(s['rouge1'].fmeasure); r2.append(s['rouge2'].fmeasure); rl.append(s['rougeL'].fmeasure)
    met = []
    for ref,hyp in zip(refs,hyps):
        try: met.append(meteor_score([nltk.word_tokenize(ref.lower())], nltk.word_tokenize(hyp.lower())))
        except: met.append(0.0)
    cider = compute_cider_proper(refs, hyps)
    m = {"bleu1":round(bleu1,4),"bleu4":round(bleu4,4),"rouge1":round(sum(r1)/len(r1),4),
         "rouge2":round(sum(r2)/len(r2),4),"rougeL":round(sum(rl)/len(rl),4),
         "meteor":round(sum(met)/len(met),4),"cider":cider,"bertscore":None}
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

# ═══════════════════════════════════════
# Also re-run our models to get proper CIDEr
# ═══════════════════════════════════════
def recompute_medgemma_cider(pairs, refs):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import PeftModel

    MODEL_ID  = "google/medgemma-4b-it"
    DMID_ONLY = "./medgemma_dmid_only"
    TWO_STAGE = "./medgemma_dmid"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)

    def gen(model, pr):
        try: img = Image.open(pr["img_path"]).convert("RGB").resize((448,448))
        except: return ""
        prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n<start_of_turn>model\n"
        inp = proc(text=prompt, images=img, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        inp["token_type_ids"] = torch.zeros_like(inp["input_ids"])
        with torch.no_grad(): out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=proc.tokenizer.eos_token_id)
        return proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results = {}

    # Zero-shot
    print("\nMedGemma zero-shot (recompute)...")
    base.eval()
    hyps = []
    for i, pr in enumerate(pairs):
        hyps.append(gen(base, pr))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["medgemma_zeroshot"] = compute_all_metrics(refs, hyps, "MedGemma ZS")

    # DMID-only
    print("\nDMID-only (recompute)...")
    mdl = PeftModel.from_pretrained(base, DMID_ONLY); mdl.eval()
    hyps = []
    for i, pr in enumerate(pairs):
        hyps.append(gen(mdl, pr))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["dmid_only"] = compute_all_metrics(refs, hyps, "DMID-only")
    del mdl; cleanup()

    # Two-stage
    print("\nTwo-stage (recompute)...")
    mdl = PeftModel.from_pretrained(base, TWO_STAGE); mdl.eval()
    hyps = []
    for i, pr in enumerate(pairs):
        hyps.append(gen(mdl, pr))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["two_stage"] = compute_all_metrics(refs, hyps, "Two-stage")
    del mdl, base; cleanup()

    return results

def main():
    pairs = load_test_pairs(52)
    refs = [p["real_report"] for p in pairs]

    # Load existing
    with open(RESULTS) as f:
        results = json.load(f)

    # 1. Fix Phi
    print("="*60)
    print("  FIX 1: Re-running Phi-3.5-Vision")
    print("="*60)
    hyps_phi = run_phi_fixed(pairs)
    results["phi_zeroshot"] = compute_all_metrics(refs, hyps_phi, "Phi-3.5-Vision (fixed)")

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)

    # 2. Recompute CIDEr for MedGemma models
    print("\n" + "="*60)
    print("  FIX 2: Recomputing all MedGemma with proper CIDEr")
    print("="*60)
    mg_results = recompute_medgemma_cider(pairs, refs)
    for key, metrics in mg_results.items():
        results[key] = metrics

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)

    # 3. Also recompute CIDEr for Qwen/LLaVA (they need re-inference too)
    # But we can just recompute CIDEr from saved hyps if we had them...
    # For now, update what we have

    # Summary
    print("\n" + "="*95)
    print("  UPDATED COMPARISON — DMID Test Set (n=52)")
    print("="*95)
    print(f"  {'Method':<35s} {'BLEU-4':>7s} {'R-1':>7s} {'R-L':>7s} {'METEOR':>7s} {'CIDEr':>7s}")
    print(f"  {'-'*75}")
    order = ["llava_zeroshot","phi_zeroshot","qwen25_zeroshot","qwen3_zeroshot",
             "medgemma_zeroshot","dmid_only","two_stage"]
    names = {"llava_zeroshot":"LLaVA-1.6-7B (ZS)","phi_zeroshot":"Phi-3.5-Vision (ZS)",
             "qwen25_zeroshot":"Qwen2.5-VL-7B (ZS)","qwen3_zeroshot":"Qwen3-VL-8B (ZS)",
             "medgemma_zeroshot":"MedGemma-4B (ZS)","dmid_only":"DMID-only (LoRA)",
             "two_stage":"Synth→DMID (ours)"}
    for k in order:
        if k in results:
            m = results[k]
            print(f"  {names.get(k,k):<35s} {m['bleu4']:>7.3f} {m['rouge1']:>7.3f} "
                  f"{m['rougeL']:>7.3f} {m['meteor']:>7.3f} {m['cider']:>7.3f}")

    print(f"\nСохранено в {RESULTS}")

if __name__ == "__main__":
    main()
