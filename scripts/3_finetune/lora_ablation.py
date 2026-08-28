"""
LoRA Rank Ablation: r=8, r=16 (done), r=32, r=64
All two-stage: synthetic pretrain → DMID
"""
import os, torch, json, gc, math
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

MODEL_ID   = "google/medgemma-4b-it"
SYNTH_LORA = "./medgemma_multimodal"
IMGS_DIR   = "./dmid/TIFF Images/TIFF Images/"
REPS_DIR   = "./dmid/Reports/Reports/"
BASE_OUT   = "./lora_ablation"
RESULTS    = "./results/lora_ablation.json"
PROMPT_TEXT = "Generate a structured mammography radiology report with breast composition (ACR density), findings, BI-RADS category, and recommendation."

os.makedirs(BASE_OUT, exist_ok=True)

def find_image(img_id):
    img_num = img_id.replace('Img','').replace('IMG','')
    for f in os.listdir(IMGS_DIR):
        if img_num in f and not f.endswith('.txt'):
            return os.path.join(IMGS_DIR, f)
    return None

def load_pairs(files):
    pairs = []
    for rf in files:
        img_id = rf.replace('.txt','')
        img_path = find_image(img_id)
        if not img_path: continue
        with open(os.path.join(REPS_DIR, rf), encoding='utf-8', errors='ignore') as f:
            report = f.read().strip()
        if report:
            pairs.append({"img_path": img_path, "report": report})
    return pairs

class DMIDDataset(Dataset):
    def __init__(self, pairs, processor, max_length=512):
        self.pairs, self.processor, self.max_length = pairs, processor, max_length
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        try: image = Image.open(item["img_path"]).convert("RGB").resize((448,448))
        except: image = Image.new("RGB", (448,448), 128)
        prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n<start_of_turn>model\n{item['report']}<end_of_turn>"
        enc = self.processor(text=prompt, images=image, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = enc["input_ids"].squeeze()
        labels = input_ids.clone(); labels[:] = -100
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        for i in range(len(input_ids) - len(sep)):
            if input_ids[i:i+len(sep)].tolist() == sep:
                labels[i+len(sep):] = input_ids[i+len(sep):]; break
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"].squeeze(),
                "pixel_values": enc["pixel_values"].squeeze(),
                "token_type_ids": torch.zeros_like(input_ids), "labels": labels}

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_cider(refs, hyps):
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [h] for i, h in enumerate(hyps)}
        score, _ = Cider().compute_score(gts, res)
        return round(score, 4)
    except:
        return 0.0

def evaluate(model, processor, test_pairs):
    refs = [p["report"] for p in test_pairs]
    hyps = []
    model.eval()
    for p in test_pairs:
        try: img = Image.open(p["img_path"]).convert("RGB").resize((448,448))
        except: hyps.append(""); continue
        prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n<start_of_turn>model\n"
        inp = processor(text=prompt, images=img, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        inp["token_type_ids"] = torch.zeros_like(inp["input_ids"])
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        hyps.append(processor.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())

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
    return {"bleu1":round(bleu1,4),"bleu4":round(bleu4,4),"rouge1":round(sum(r1)/len(r1),4),
            "rouge2":round(sum(r2)/len(r2),4),"rougeL":round(sum(rl)/len(rl),4),
            "meteor":round(sum(met)/len(met),4),"cider":cider}

def main():
    all_files = sorted(os.listdir(REPS_DIR))
    train_pairs = load_pairs(all_files[:407])
    val_pairs = load_pairs(all_files[407:458])
    test_pairs = load_pairs(all_files[-52:])
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # Load existing results
    if os.path.exists(RESULTS):
        with open(RESULTS) as f: results = json.load(f)
    else:
        results = {}

    # r=16, alpha=32 already done — add from existing
    results["r16_a32"] = {"bleu4":0.2784,"rouge1":0.7193,"rougeL":0.6587,"meteor":0.657,"cider":0.6392,
                          "config":"r=16, α=32 (default)"}

    configs = [
        {"r": 8,  "alpha": 16, "key": "r8_a16"},
        {"r": 8,  "alpha": 32, "key": "r8_a32"},
        {"r": 32, "alpha": 32, "key": "r32_a32"},
        {"r": 32, "alpha": 64, "key": "r32_a64"},
        {"r": 64, "alpha": 64, "key": "r64_a64"},
        {"r": 64, "alpha": 128, "key": "r64_a128"},
    ]

    for cfg in configs:
        key = cfg["key"]
        if key in results:
            print(f"\n{key} — already done, skipping")
            continue

        r, alpha = cfg["r"], cfg["alpha"]
        print(f"\n{'='*60}")
        print(f"  Training: r={r}, alpha={alpha}")
        print(f"{'='*60}")

        out_dir = f"{BASE_OUT}/{key}"
        os.makedirs(out_dir, exist_ok=True)

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        base.config.use_cache = False

        # Stage 1 weights (synthetic pretrain with default r=16)
        # For fair comparison, we retrain Stage 2 with different LoRA configs
        # But Stage 1 pretrain stays the same (r=16)
        # Load synthetic pretrained base, then apply new LoRA for Stage 2
        model = PeftModel.from_pretrained(base, SYNTH_LORA)
        model = model.merge_and_unload()  # Merge Stage 1 into base

        # Apply new LoRA config for Stage 2
        lora_config = LoraConfig(
            r=r, lora_alpha=alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params:,}")

        train_ds = DMIDDataset(train_pairs, processor)
        val_ds = DMIDDataset(val_pairs, processor)

        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=5,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=20,
            logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, bf16=True, report_to="none",
            optim="paged_adamw_8bit", remove_unused_columns=False)

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()

        # Eval
        metrics = evaluate(model, processor, test_pairs)
        metrics["config"] = f"r={r}, α={alpha}"
        metrics["trainable_params"] = n_params
        print(f"\n  Results: {metrics}")
        results[key] = metrics

        # Save after each
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)

        del model, base, trainer, processor
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("  LoRA ABLATION SUMMARY")
    print("="*80)
    print(f"  {'Config':<20s} {'BLEU-4':>8s} {'R-1':>8s} {'R-L':>8s} {'METEOR':>8s} {'CIDEr':>8s}")
    print(f"  {'-'*60}")
    for key in ["r8_a16","r8_a32","r16_a32","r32_a32","r32_a64","r64_a64","r64_a128"]:
        if key in results:
            m = results[key]
            print(f"  {m.get('config','?'):<20s} {m['bleu4']:>8.3f} {m['rouge1']:>8.3f} "
                  f"{m['rougeL']:>8.3f} {m['meteor']:>8.3f} {m['cider']:>8.3f}")

    print(f"\nСохранено в {RESULTS}")

if __name__ == "__main__":
    main()
