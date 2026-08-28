"""
Resume Phase 2 — skip already computed models
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

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_cider(refs, hyps):
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
        scores.append(dot / ((math.sqrt(sum(v**2 for v in ri.values()))+1e-8) * (math.sqrt(sum(v**2 for v in hi.values()))+1e-8)))
    return sum(scores)/len(scores) if scores else 0.0

def compute_metrics(refs, hyps, label):
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
    cider = compute_cider(refs, hyps)
    m = {"bleu1":round(bleu1,4),"bleu4":round(bleu4,4),"rouge1":round(sum(r1)/len(r1),4),
         "rouge2":round(sum(r2)/len(r2),4),"rougeL":round(sum(rl)/len(rl),4),
         "meteor":round(sum(met)/len(met),4),"cider":round(cider,4),"bertscore":None}
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

def cleanup():
    gc.collect(); torch.cuda.empty_cache()

# Load existing results
def load_existing():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f: return json.load(f)
    return {}

def save_results(r):
    with open(RESULTS, "w") as f: json.dump(r, f, indent=2)

def main():
    pairs = load_test_pairs(52)
    refs = [p["real_report"] for p in pairs]
    results = load_existing()
    print(f"Already computed: {list(results.keys())}")

    # ── 1. Qwen2.5-VL ──
    if "qwen25_zeroshot" not in results:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print("\n[1/7] Qwen2.5-VL-7B...")
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        p = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct"); m.eval()
        hyps = []
        for i, pr in enumerate(pairs):
            try:
                img = Image.open(pr["img_path"]).convert("RGB").resize((448,448))
                msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT_TEXT}]}]
                t = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = p(text=[t], images=[img], return_tensors="pt", padding=True).to(m.device)
                with torch.no_grad(): out = m.generate(**inp, max_new_tokens=200, do_sample=False)
                hyps.append(p.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip())
            except: hyps.append("")
            print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
        print()
        results["qwen25_zeroshot"] = compute_metrics(refs, hyps, "Qwen2.5-VL-7B"); save_results(results)
        del m,p; cleanup()
    else:
        print("Qwen2.5 — already done, skipping")

    # ── 2. LLaVA ──
    if "llava_zeroshot" not in results:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        print("\n[2/7] LLaVA-1.6-7B...")
        p = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        m = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"); m.eval()
        hyps = []
        for i, pr in enumerate(pairs):
            try:
                img = Image.open(pr["img_path"]).convert("RGB").resize((448,448))
                inp = p(text=f"[INST] <image>\n{PROMPT_TEXT} [/INST]", images=img, return_tensors="pt").to(m.device)
                with torch.no_grad(): out = m.generate(**inp, max_new_tokens=200, do_sample=False)
                hyps.append(p.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            except: hyps.append("")
            print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
        print()
        results["llava_zeroshot"] = compute_metrics(refs, hyps, "LLaVA-1.6-7B"); save_results(results)
        del m,p; cleanup()
    else:
        print("LLaVA — already done, skipping")

    # ── 3. Phi-3.5 ──
    if "phi_zeroshot" not in results:
        from transformers import AutoModelForCausalLM, AutoProcessor
        print("\n[3/7] Phi-3.5-Vision...")
        m = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, _attn_implementation="eager")
        p = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True); m.eval()
        hyps = []
        for i, pr in enumerate(pairs):
            try:
                img = Image.open(pr["img_path"]).convert("RGB").resize((448,448))
                msgs = [{"role":"user","content":f"<|image_1|>\n{PROMPT_TEXT}"}]
                prompt = p.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = p(prompt, images=[img], return_tensors="pt").to(m.device)
                with torch.no_grad(): out = m.generate(**inp, max_new_tokens=200, do_sample=False, eos_token_id=p.tokenizer.eos_token_id)
                hyps.append(p.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            except: hyps.append("")
            print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
        print()
        results["phi_zeroshot"] = compute_metrics(refs, hyps, "Phi-3.5-Vision"); save_results(results)
        del m,p; cleanup()
    else:
        print("Phi — already done, skipping")

    # ── 4. Qwen3-VL ──
    if "qwen3_zeroshot" not in results:
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            print("\n[4/7] Qwen3-VL-8B...")
            m = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
            p = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct"); m.eval()
            hyps = []
            for i, pr in enumerate(pairs):
                try:
                    img = Image.open(pr["img_path"]).convert("RGB").resize((448,448))
                    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT_TEXT}]}]
                    t = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inp = p(text=[t], images=[img], return_tensors="pt", padding=True).to(m.device)
                    with torch.no_grad(): out = m.generate(**inp, max_new_tokens=200, do_sample=False)
                    hyps.append(p.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip())
                except Exception as e:
                    print(f"\n  Err {i}: {e}"); hyps.append("")
                print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
            print()
            results["qwen3_zeroshot"] = compute_metrics(refs, hyps, "Qwen3-VL-8B"); save_results(results)
            del m,p; cleanup()
        except ImportError:
            print("\n[4/7] Qwen3 needs transformers>=4.57, skipping")
    else:
        print("Qwen3 — already done, skipping")

    # ── 5-7. MedGemma models ──
    medgemma_keys = ["medgemma_zeroshot", "dmid_only", "two_stage"]
    if not all(k in results for k in medgemma_keys):
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        proc = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        base = AutoModelForImageTextToText.from_pretrained("google/medgemma-4b-it", quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        
        def gen(model, pr):
            try: img = Image.open(pr["img_path"]).convert("RGB").resize((448,448))
            except: return ""
            prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n<start_of_turn>model\n"
            inp = proc(text=prompt, images=img, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            inp["token_type_ids"] = torch.zeros_like(inp["input_ids"])
            with torch.no_grad(): out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=proc.tokenizer.eos_token_id)
            return proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if "medgemma_zeroshot" not in results:
            print("\n[5/7] MedGemma zero-shot...")
            base.eval()
            hyps = [gen(base, pr) for i, pr in enumerate(pairs) if not print(f"\r  {i+1}/{len(pairs)}", end="", flush=True) or True]
            print()
            results["medgemma_zeroshot"] = compute_metrics(refs, hyps, "MedGemma zero-shot"); save_results(results)
        else: print("MedGemma ZS — done")

        if "dmid_only" not in results:
            print("\n[6/7] MedGemma DMID-only...")
            mdl = PeftModel.from_pretrained(base, "./medgemma_dmid_only"); mdl.eval()
            hyps = []
            for i, pr in enumerate(pairs): hyps.append(gen(mdl, pr)); print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
            print()
            results["dmid_only"] = compute_metrics(refs, hyps, "DMID-only"); save_results(results)
            del mdl; cleanup()
        else: print("DMID-only — done")

        if "two_stage" not in results:
            print("\n[7/7] MedGemma Two-stage...")
            mdl = PeftModel.from_pretrained(base, "./medgemma_dmid"); mdl.eval()
            hyps = []
            for i, pr in enumerate(pairs): hyps.append(gen(mdl, pr)); print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
            print()
            results["two_stage"] = compute_metrics(refs, hyps, "Two-stage"); save_results(results)
            del mdl; cleanup()
        else: print("Two-stage — done")

        del base; cleanup()
    else:
        print("All MedGemma — already done")

    # ── Summary ──
    print("\n" + "="*95)
    print("  FULL COMPARISON — DMID Test Set (n=52)")
    print("="*95)
    print(f"  {'Method':<35s} {'BLEU-4':>7s} {'R-1':>7s} {'R-L':>7s} {'METEOR':>7s} {'CIDEr':>7s}")
    print(f"  {'-'*75}")
    order = ["llava_zeroshot","phi_zeroshot","qwen25_zeroshot","qwen3_zeroshot","medgemma_zeroshot","dmid_only","two_stage"]
    names = {"llava_zeroshot":"LLaVA-1.6-7B","phi_zeroshot":"Phi-3.5-Vision","qwen25_zeroshot":"Qwen2.5-VL-7B",
             "qwen3_zeroshot":"Qwen3-VL-8B","medgemma_zeroshot":"MedGemma-4B (ZS)","dmid_only":"DMID-only","two_stage":"Synth→DMID (ours)"}
    for k in order:
        if k in results:
            m = results[k]
            print(f"  {names[k]:<35s} {m['bleu4']:>7.3f} {m['rouge1']:>7.3f} {m['rougeL']:>7.3f} {m['meteor']:>7.3f} {m['cider']:>7.3f}")
    print(f"\nСохранено в {RESULTS}")

if __name__ == "__main__":
    main()
