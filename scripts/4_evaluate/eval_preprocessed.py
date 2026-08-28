"""
Evaluate preprocessed model on preprocessed test images
"""
import os, torch, json, math
from PIL import Image
from collections import Counter
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

MODEL_ID  = "google/medgemma-4b-it"
PREPROC   = "./medgemma_dmid_preprocessed"
ORIG      = "./lora_ablation/r64_a64"
IMGS_RAW  = "./dmid/TIFF Images/TIFF Images/"
IMGS_PREP = "./dmid/preprocessed/"
REPS_DIR  = "./dmid/Reports/Reports/"
RESULTS   = "./results/eval_preprocessed.json"
PROMPT    = "Generate a structured mammography report with breast composition, findings, BI-RADS category and recommendation."

def load_test(n=52):
    pairs = []
    report_files = sorted(os.listdir(REPS_DIR))
    for rf in report_files[-n:]:
        img_id = rf.replace('.txt', '')
        img_num = img_id.replace('Img','').replace('IMG','')
        # Find in both dirs
        raw_path, prep_path = None, None
        for f in os.listdir(IMGS_RAW):
            if img_num in f and not f.endswith('.txt'):
                raw_path = os.path.join(IMGS_RAW, f); break
        for f in os.listdir(IMGS_PREP):
            if img_num in f:
                prep_path = os.path.join(IMGS_PREP, f); break
        with open(os.path.join(REPS_DIR, rf), encoding='utf-8', errors='ignore') as fh:
            report = fh.read().strip()
        if report and (raw_path or prep_path):
            pairs.append({"raw": raw_path, "prep": prep_path, "report": report})
    return pairs

def generate(model, processor, img_path):
    try: image = Image.open(img_path).convert("RGB").resize((448,448))
    except: return ""
    prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT}<end_of_turn>\n<start_of_turn>model\n"
    inp = processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    inp["token_type_ids"] = torch.zeros_like(inp["input_ids"])
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def compute_cider(refs, hyps):
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i:[r] for i,r in enumerate(refs)}
        res = {i:[h] for i,h in enumerate(hyps)}
        score, _ = Cider().compute_score(gts, res)
        return round(score, 4)
    except: return 0.0

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
         "meteor":round(sum(met)/len(met),4),"cider":cider}
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

def main():
    pairs = load_test(52)
    refs = [p["report"] for p in pairs]
    print(f"Test: {len(pairs)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)

    results = {}

    # 1. Preprocessed model + preprocessed images
    print("\n[1/3] Preprocessed model + preprocessed images...")
    model = PeftModel.from_pretrained(base, PREPROC); model.eval()
    hyps = []
    for i,p in enumerate(pairs):
        img = p["prep"] or p["raw"]
        hyps.append(generate(model, processor, img))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["preprocessed_model_prep_imgs"] = compute_metrics(refs, hyps, "Preprocessed model + prep images")
    del model; torch.cuda.empty_cache()

    # 2. Preprocessed model + raw images (how it handles unprocessed input)
    print("\n[2/3] Preprocessed model + raw images...")
    model = PeftModel.from_pretrained(base, PREPROC); model.eval()
    hyps = []
    for i,p in enumerate(pairs):
        hyps.append(generate(model, processor, p["raw"]))
        print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
    print()
    results["preprocessed_model_raw_imgs"] = compute_metrics(refs, hyps, "Preprocessed model + raw images")
    del model; torch.cuda.empty_cache()

    # 3. Original best model (r64) + raw images (for comparison)
    if os.path.exists(ORIG):
        print("\n[3/3] Original r64 model + raw images...")
        model = PeftModel.from_pretrained(base, ORIG); model.eval()
        hyps = []
        for i,p in enumerate(pairs):
            hyps.append(generate(model, processor, p["raw"]))
            print(f"\r  {i+1}/{len(pairs)}", end="", flush=True)
        print()
        results["original_r64_raw_imgs"] = compute_metrics(refs, hyps, "Original r64 + raw images")
        del model; torch.cuda.empty_cache()

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS}")

if __name__ == "__main__":
    main()
