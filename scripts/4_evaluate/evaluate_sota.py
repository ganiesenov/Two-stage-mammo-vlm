import json
import torch
from PIL import Image
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
VINDR_IMG   = "./vindr"

def load_test(n=30):
    data = []
    with open(VINDR_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("is_valid"): continue
                img_path = f"{VINDR_IMG}/{r['study_id']}/{r['image_id']}.png"
                if Path(img_path).exists():
                    r["image_path"] = img_path
                    data.append(r)
            except: pass
    return data[-n:]

def compute_metrics(references, hypotheses, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1,r2,rl = [],[],[]
    for ref,hyp in zip(references,hypotheses):
        s = scorer.score(ref,hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    _,_,F = bert_score(hypotheses, references, lang="en", verbose=False)
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

def run_qwen(test):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("\n[1/2] Загружаю Qwen2.5-VL-7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto", dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model.eval()

    hyps = []
    for r in test:
        image = Image.open(r["image_path"]).convert("RGB").resize((448,448))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Generate a structured mammography radiology report.\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}\nDensity: {r['breast_density']}\nStructure: Density → Findings → BI-RADS → Recommendation"}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        hyps.append(processor.tokenizer.decode(gen, skip_special_tokens=True).strip())
        print(".", end="", flush=True)

    del model; torch.cuda.empty_cache()
    return hyps

def run_llava(test):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    print("\n[2/2] Загружаю LLaVA-1.6-Mistral-7B...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", device_map="auto", dtype=torch.bfloat16)
    model.eval()

    hyps = []
    for r in test:
        image = Image.open(r["image_path"]).convert("RGB").resize((448,448))
        prompt = f"[INST] <image>\nGenerate a structured mammography radiology report.\nFindings: {r['finding_categories']}\nBI-RADS: {r['breast_birads']}\nDensity: {r['breast_density']}\nStructure: Density → Findings → BI-RADS → Recommendation [/INST]"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        hyps.append(processor.tokenizer.decode(gen, skip_special_tokens=True).strip())
        print(".", end="", flush=True)

    del model; torch.cuda.empty_cache()
    return hyps

def main():
    print("Загружаю тестовые данные...")
    test = load_test(n=30)
    refs = [r["synthetic_report"] for r in test]
    print(f"Тест: {len(test)} примеров")

    results = {}

    # Qwen2.5-VL
    hyps_qwen = run_qwen(test)
    results["qwen2.5-vl-7b"] = compute_metrics(refs, hyps_qwen, "Qwen2.5-VL-7B (zero-shot)")

    # LLaVA
    hyps_llava = run_llava(test)
    results["llava-1.6-7b"] = compute_metrics(refs, hyps_llava, "LLaVA-1.6-Mistral-7B (zero-shot)")

    out = "./eval_results_sota.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nГотово! Результаты в {out}")

if __name__ == "__main__":
    main()
