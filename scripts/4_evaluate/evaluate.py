import json
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

nltk.download('punkt', quiet=True)

MODEL_ID   = "google/medgemma-4b-it"
LORA_DIR   = "./medgemma_finetuned"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
CBIS_JSONL  = "./cbis-ddsm/synthetic_reports_cbis.jsonl"

def load_test_reports(*paths, n=40):
    data = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("is_valid") and r.get("synthetic_report"):
                        data.append(r)
                except: pass
    # Берём последние n как тест (не видели при обучении ~10%)
    return data[-n:]

def format_prompt(report):
    findings = report.get("finding_categories", report.get("finding_type", "mass"))
    birads = report.get("breast_birads", report.get("assessment", ""))
    density = report.get("breast_density", "")
    laterality = report.get("laterality", "")
    return f"<start_of_turn>user\nGenerate a structured mammography report.\nFindings: {findings}\nBI-RADS: {birads}\nDensity: {density}\nLaterality: {laterality}<end_of_turn>\n<start_of_turn>model\n"

def generate(model, tokenizer, prompt, max_new=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # token_type_ids нули
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def compute_metrics(references, hypotheses):
    # BLEU
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]
    sf = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=sf)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)

    # BERTScore
    P, R, F = bert_score(hypotheses, references, lang="en", verbose=False)

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ EVALUATION")
    print("="*50)
    print(f"BLEU-1:    {bleu1:.4f}")
    print(f"BLEU-4:    {bleu4:.4f}")
    print(f"ROUGE-1:   {sum(r1)/len(r1):.4f}")
    print(f"ROUGE-2:   {sum(r2)/len(r2):.4f}")
    print(f"ROUGE-L:   {sum(rl)/len(rl):.4f}")
    print(f"BERTScore: {F.mean().item():.4f}")
    print("="*50)

    return {
        "bleu1": bleu1, "bleu4": bleu4,
        "rouge1": sum(r1)/len(r1), "rouge2": sum(r2)/len(r2), "rougeL": sum(rl)/len(rl),
        "bertscore": F.mean().item()
    }

def main():
    print("Загружаю тестовые отчёты...")
    test_reports = load_test_reports(VINDR_JSONL, CBIS_JSONL, n=40)
    print(f"Тест: {len(test_reports)} отчётов")

    print("\nЗагружаю модель...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    
    print("\n--- Baseline (zero-shot) ---")
    refs, hyps_baseline = [], []
    for r in test_reports:
        prompt = format_prompt(r)
        hyp = generate(base_model, tokenizer, prompt)
        refs.append(r["synthetic_report"])
        hyps_baseline.append(hyp)
        print(".", end="", flush=True)
    baseline_metrics = compute_metrics(refs, hyps_baseline)

    print("\n--- Fine-tuned + LoRA ---")
    ft_model = PeftModel.from_pretrained(base_model, LORA_DIR)
    ft_model.eval()
    hyps_ft = []
    for r in test_reports:
        prompt = format_prompt(r)
        hyp = generate(ft_model, tokenizer, prompt)
        hyps_ft.append(hyp)
        print(".", end="", flush=True)
    ft_metrics = compute_metrics(refs, hyps_ft)

    # Сохраняем результаты
    results = {"baseline": baseline_metrics, "finetuned": ft_metrics}
    with open("./eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nРезультаты сохранены в eval_results.json")

if __name__ == "__main__":
    main()
