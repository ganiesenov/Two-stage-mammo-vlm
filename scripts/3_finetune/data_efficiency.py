"""
Data Efficiency Experiment
Train DMID-only and Two-stage with N = 50, 100, 200 real examples
(N=407 already done)
"""
import os, sys, torch, json
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText, 
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID   = "google/medgemma-4b-it"
SYNTH_LORA = "./medgemma_multimodal"
IMGS_DIR   = "./dmid/TIFF Images/TIFF Images/"
REPS_DIR   = "./dmid/Reports/Reports/"
BASE_OUT   = "./data_efficiency"
RESULTS    = "./results/data_efficiency.json"

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
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        try: image = Image.open(item["img_path"]).convert("RGB").resize((448,448))
        except: image = Image.new("RGB", (448,448), 128)
        prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report with breast composition, findings, BI-RADS category and recommendation.<end_of_turn>\n<start_of_turn>model\n{item['report']}<end_of_turn>"
        encoding = self.processor(text=prompt, images=image, return_tensors="pt",
                                  truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[:] = -100
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        for i in range(len(input_ids) - len(sep)):
            if input_ids[i:i+len(sep)].tolist() == sep:
                labels[i+len(sep):] = input_ids[i+len(sep):]
                break
        return {"input_ids": input_ids, "attention_mask": encoding["attention_mask"].squeeze(),
                "pixel_values": encoding["pixel_values"].squeeze(),
                "token_type_ids": torch.zeros_like(input_ids), "labels": labels}

def generate(model, processor, img_path):
    try: image = Image.open(img_path).convert("RGB").resize((448,448))
    except: return ""
    prompt = "<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report with breast composition, findings, BI-RADS category and recommendation.<end_of_turn>\n<start_of_turn>model\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def evaluate(model, processor, test_pairs):
    refs = [p["report"] for p in test_pairs]
    hyps = []
    for p in test_pairs:
        hyps.append(generate(model, processor, p["img_path"]))
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    r1, rl = [], []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure); rl.append(s['rougeL'].fmeasure)
    _, _, F = bert_score(hyps, refs, lang="en", verbose=False)
    return {"bleu4": round(bleu4,4), "rouge1": round(sum(r1)/len(r1),4),
            "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)}

def train_and_eval(n_train, variant, processor, base_model, all_files, test_pairs):
    """variant: 'dmid_only' or 'two_stage'"""
    out_dir = f"{BASE_OUT}/{variant}_n{n_train}"
    print(f"\n{'='*60}")
    print(f"  Training: {variant}, N={n_train}")
    print(f"{'='*60}")

    train_pairs = load_pairs(all_files[:n_train])
    val_pairs = load_pairs(all_files[n_train:n_train+51])
    if len(val_pairs) < 10:
        val_pairs = load_pairs(all_files[max(0, n_train-51):n_train])
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    if variant == "two_stage":
        model = PeftModel.from_pretrained(base_model, SYNTH_LORA, is_trainable=True)
    else:
        lora_config = LoraConfig(r=16, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)

    model.config.use_cache = False
    train_ds = DMIDDataset(train_pairs, processor)
    val_ds = DMIDDataset(val_pairs, processor)

    # Scale epochs inversely with data size
    n_epochs = 5 if n_train >= 200 else 8 if n_train >= 100 else 10

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=n_epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=10,
        logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, bf16=True, report_to="none",
        optim="paged_adamw_8bit", remove_unused_columns=False)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model(out_dir)
    
    # Evaluate
    model.eval()
    metrics = evaluate(model, processor, test_pairs)
    print(f"  Results: {metrics}")
    
    # Cleanup for next run
    del model, trainer
    torch.cuda.empty_cache()
    
    return metrics

def main():
    os.makedirs(BASE_OUT, exist_ok=True)
    
    all_files = sorted(os.listdir(REPS_DIR))
    test_files = all_files[-52:]
    train_val_files = all_files[:-52]  # first 458 for train+val
    
    test_pairs = load_pairs(test_files)
    print(f"Test: {len(test_pairs)} pairs")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    results = {}
    
    # Already have N=407 results
    results["dmid_only_407"] = {"bleu4": 0.2205, "rouge1": 0.6858, "rougeL": 0.6148, "bertscore": 0.9093}
    results["two_stage_407"] = {"bleu4": 0.3065, "rouge1": 0.7241, "rougeL": 0.6620, "bertscore": 0.9171}
    
    for n_train in [50, 100, 200]:
        for variant in ["dmid_only", "two_stage"]:
            # Reload base model fresh each time
            base = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
            base.config.use_cache = False
            
            metrics = train_and_eval(n_train, variant, processor, base, train_val_files, test_pairs)
            results[f"{variant}_{n_train}"] = metrics
            
            del base
            torch.cuda.empty_cache()
            
            # Save after each run
            with open(RESULTS, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to {RESULTS}")

    # Final summary
    print("\n" + "="*60)
    print("  DATA EFFICIENCY SUMMARY (ROUGE-L)")
    print("="*60)
    print(f"  {'N':>5s}  {'DMID-only':>10s}  {'Two-stage':>10s}  {'Δ':>8s}")
    print(f"  {'-'*40}")
    for n in [50, 100, 200, 407]:
        do = results.get(f"dmid_only_{n}", {}).get("rougeL", "?")
        ts = results.get(f"two_stage_{n}", {}).get("rougeL", "?")
        if isinstance(do, float) and isinstance(ts, float):
            delta = f"+{(ts-do)/do*100:.1f}%"
        else:
            delta = "?"
        print(f"  {n:>5d}  {str(do):>10s}  {str(ts):>10s}  {delta:>8s}")

if __name__ == "__main__":
    main()
