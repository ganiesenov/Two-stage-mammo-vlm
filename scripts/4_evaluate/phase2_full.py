"""
Phase 2 Full: All VLM baselines on DMID test + METEOR/CIDEr
1. Qwen2.5-VL-7B (zero-shot)
2. LLaVA-1.6-7B (zero-shot)
3. Phi-3.5-Vision (zero-shot)
4. Qwen3-VL-8B (zero-shot) — latest
5. MedGemma-4B (zero-shot)
6. MedGemma DMID-only
7. MedGemma Two-stage (ours)
"""
import os, torch, json, gc, math
from PIL import Image
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
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
    test_files = report_files[-n:]
    for rf in test_files:
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
    print(f"Test pairs: {len(pairs)}/{n}")
    return pairs

# ── Metrics ──
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_cider(refs, hyps):
    N = len(refs)
    df = Counter()
    for ref in refs:
        seen = set()
        tokens = ref.lower().split()
        for n in range(1, 5):
            for ng in ngrams(tokens, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    scores = []
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        ref_tf, hyp_tf = Counter(), Counter()
        for n in range(1, 5):
            ref_tf.update(ngrams(ref_tokens, n))
            hyp_tf.update(ngrams(hyp_tokens, n))
        ref_tfidf, hyp_tfidf = {}, {}
        for ng, c in ref_tf.items():
            ref_tfidf[ng] = c * math.log(max(1., N) / max(1., df.get(ng, 0)))
        for ng, c in hyp_tf.items():
            hyp_tfidf[ng] = c * math.log(max(1., N) / max(1., df.get(ng, 0)))
        all_ng = set(ref_tfidf) | set(hyp_tfidf)
        dot = sum(ref_tfidf.get(ng, 0) * hyp_tfidf.get(ng, 0) for ng in all_ng)
        nr = math.sqrt(sum(v**2 for v in ref_tfidf.values())) + 1e-8
        nh = math.sqrt(sum(v**2 for v in hyp_tfidf.values())) + 1e-8
        scores.append(dot / (nr * nh))
    return sum(scores) / len(scores) if scores else 0.0

def compute_all_metrics(refs, hyps, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure); r2.append(s['rouge2'].fmeasure); rl.append(s['rougeL'].fmeasure)
    meteor_scores = []
    for ref, hyp in zip(refs, hyps):
        try:
            ms = meteor_score([nltk.word_tokenize(ref.lower())], nltk.word_tokenize(hyp.lower()))
        except:
            ms = 0.0
        meteor_scores.append(ms)
    cider = compute_cider(refs, hyps)
    _, _, F = bert_score(hyps, refs, lang="en", verbose=False)
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "meteor": round(sum(meteor_scores)/len(meteor_scores),4),
         "cider": round(cider,4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k, v in m.items():
        print(f"  {k:12s}: {v}")
    return m

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# ══════════════════════════════════════════════
# 1. Qwen2.5-VL-7B
# ══════════════════════════════════════════════
def run_qwen25(pairs):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("\n[1/7] Loading Qwen2.5-VL-7B-Instruct...")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    hyps = []
    for i, p in enumerate(pairs):
        try:
            image = Image.open(p["img_path"]).convert("RGB").resize((448, 448))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            gen_ids = out[0][inputs.input_ids.shape[1]:]
            hyps.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        except Exception as e:
            print(f"\n  Error on {i}: {e}")
            hyps.append("")
        print(f"\r  Qwen2.5: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    del model, processor; cleanup()
    return hyps

# ══════════════════════════════════════════════
# 2. LLaVA-1.6-7B
# ══════════════════════════════════════════════
def run_llava(pairs):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    print("\n[2/7] Loading LLaVA-1.6-Mistral-7B...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    hyps = []
    for i, p in enumerate(pairs):
        try:
            image = Image.open(p["img_path"]).convert("RGB").resize((448, 448))
            prompt = f"[INST] <image>\n{PROMPT_TEXT} [/INST]"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            hyps.append(processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        except Exception as e:
            print(f"\n  Error on {i}: {e}")
            hyps.append("")
        print(f"\r  LLaVA: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    del model, processor; cleanup()
    return hyps

# ══════════════════════════════════════════════
# 3. Phi-3.5-Vision
# ══════════════════════════════════════════════
def run_phi(pairs):
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("\n[3/7] Loading Phi-3.5-Vision...")
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
            messages = [{"role": "user", "content": f"<|image_1|>\n{PROMPT_TEXT}"}]
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                     eos_token_id=processor.tokenizer.eos_token_id)
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            hyps.append(processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        except Exception as e:
            print(f"\n  Error on {i}: {e}")
            hyps.append("")
        print(f"\r  Phi: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    del model, processor; cleanup()
    return hyps

# ══════════════════════════════════════════════
# 4. Qwen3-VL-8B (latest)
# ══════════════════════════════════════════════
def run_qwen3(pairs):
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("\n[4/7] Qwen3-VL requires transformers>=4.57.0, skipping...")
        return None
    print("\n[4/7] Loading Qwen3-VL-8B-Instruct...")
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    hyps = []
    for i, p in enumerate(pairs):
        try:
            image = Image.open(p["img_path"]).convert("RGB").resize((448, 448))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            gen_ids = out[0][inputs.input_ids.shape[1]:]
            hyps.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        except Exception as e:
            print(f"\n  Error on {i}: {e}")
            hyps.append("")
        print(f"\r  Qwen3: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    del model, processor; cleanup()
    return hyps

# ══════════════════════════════════════════════
# 5-7. Our MedGemma models
# ══════════════════════════════════════════════
def run_medgemma_models(pairs):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import PeftModel

    MODEL_ID  = "google/medgemma-4b-it"
    DMID_ONLY = "./medgemma_dmid_only"
    TWO_STAGE = "./medgemma_dmid"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)

    def gen(model, p):
        try:
            image = Image.open(p["img_path"]).convert("RGB").resize((448, 448))
        except: return ""
        prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n<start_of_turn>model\n"
        inputs = processor(text=prompt, images=image, return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 pad_token_id=processor.tokenizer.eos_token_id)
        return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results = {}

    # 5. Zero-shot
    print("\n[5/7] MedGemma zero-shot...")
    base.eval()
    hyps_zs = []
    for i, p in enumerate(pairs):
        hyps_zs.append(gen(base, p))
        print(f"\r  ZS: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["medgemma_zeroshot"] = hyps_zs

    # 6. DMID-only
    print("\n[6/7] MedGemma DMID-only...")
    model_do = PeftModel.from_pretrained(base, DMID_ONLY)
    model_do.eval()
    hyps_do = []
    for i, p in enumerate(pairs):
        hyps_do.append(gen(model_do, p))
        print(f"\r  DO: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["dmid_only"] = hyps_do
    del model_do; cleanup()

    # 7. Two-stage
    print("\n[7/7] MedGemma Two-stage...")
    model_2s = PeftModel.from_pretrained(base, TWO_STAGE)
    model_2s.eval()
    hyps_2s = []
    for i, p in enumerate(pairs):
        hyps_2s.append(gen(model_2s, p))
        print(f"\r  2S: {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["two_stage"] = hyps_2s
    del model_2s, base; cleanup()

    return results

# ══════════════════════════════════════════════
def main():
    pairs = load_test_pairs(52)
    refs = [p["real_report"] for p in pairs]

    all_results = {}

    # 1. Qwen2.5-VL
    hyps = run_qwen25(pairs)
    all_results["qwen25_zeroshot"] = compute_all_metrics(refs, hyps, "Qwen2.5-VL-7B (zero-shot)")
    with open(RESULTS, "w") as f: json.dump(all_results, f, indent=2)

    # 2. LLaVA
    hyps = run_llava(pairs)
    all_results["llava_zeroshot"] = compute_all_metrics(refs, hyps, "LLaVA-1.6-7B (zero-shot)")
    with open(RESULTS, "w") as f: json.dump(all_results, f, indent=2)

    # 3. Phi
    hyps = run_phi(pairs)
    all_results["phi_zeroshot"] = compute_all_metrics(refs, hyps, "Phi-3.5-Vision (zero-shot)")
    with open(RESULTS, "w") as f: json.dump(all_results, f, indent=2)

    # 4. Qwen3-VL
    hyps = run_qwen3(pairs)
    if hyps is not None:
        all_results["qwen3_zeroshot"] = compute_all_metrics(refs, hyps, "Qwen3-VL-8B (zero-shot)")
        with open(RESULTS, "w") as f: json.dump(all_results, f, indent=2)

    # 5-7. MedGemma models
    mg_results = run_medgemma_models(pairs)
    for key, hyps in mg_results.items():
        all_results[key] = compute_all_metrics(refs, hyps, key)
        with open(RESULTS, "w") as f: json.dump(all_results, f, indent=2)

    # ── Final Summary ──
    print("\n" + "="*95)
    print("  FULL COMPARISON — DMID Test Set (n=52, real radiologist reports)")
    print("="*95)
    print(f"  {'Method':<35s} {'BLEU-1':>7s} {'BLEU-4':>7s} {'R-1':>7s} {'R-L':>7s} {'METEOR':>7s} {'CIDEr':>7s} {'BERTSc':>7s}")
    print(f"  {'-'*90}")

    order = ["llava_zeroshot", "phi_zeroshot", "qwen25_zeroshot", "qwen3_zeroshot",
             "medgemma_zeroshot", "dmid_only", "two_stage"]
    labels = {
        "llava_zeroshot":      "LLaVA-1.6-7B (zero-shot)",
        "phi_zeroshot":        "Phi-3.5-Vision (zero-shot)",
        "qwen25_zeroshot":     "Qwen2.5-VL-7B (zero-shot)",
        "qwen3_zeroshot":      "Qwen3-VL-8B (zero-shot)",
        "medgemma_zeroshot":   "MedGemma-4B (zero-shot)",
        "dmid_only":           "DMID-only (fresh LoRA)",
        "two_stage":           "Synthetic→DMID (ours)",
    }

    for key in order:
        if key in all_results:
            m = all_results[key]
            label = labels.get(key, key)
            print(f"  {label:<35s} {m['bleu1']:>7.3f} {m['bleu4']:>7.3f} {m['rouge1']:>7.3f} "
                  f"{m['rougeL']:>7.3f} {m['meteor']:>7.3f} {m['cider']:>7.3f} {m['bertscore']:>7.3f}")

    # AMRG + CLIP references
    print(f"  {'─'*90}")
    print(f"  {'AMRG (Sung et al.)':<35s} {'0.308':>7s} {'0.308':>7s} {'0.575':>7s} "
          f"{'0.569':>7s} {'0.615':>7s} {'0.582':>7s} {'—':>7s}")

    print(f"\nСохранено в {RESULTS}")

if __name__ == "__main__":
    main()
