import os
import torch
import json
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)

MODEL_ID   = "google/medgemma-4b-it"
MM_LORA    = "./medgemma_multimodal"
DMID_IMGS  = "./dmid/TIFF Images/TIFF Images/"
DMID_REPS  = "./dmid/Reports/Reports/"
DMID_META  = "./dmid/Metadata.xlsx"

def load_dmid_pairs(n=52):
    """Загружаем пары изображение-отчёт из DMID (тест сет AMRG: последние 52)"""
    import pandas as pd
    pairs = []

    # Читаем метаданные
    meta = pd.read_excel(DMID_META)
    print(f"Metadata columns: {meta.columns.tolist()}")
    print(meta.head(3))

    # Берём все отчёты
    report_files = sorted(os.listdir(DMID_REPS))
    test_files = report_files[-n:]  # последние 52 как в AMRG

    for rf in test_files:
        img_id = rf.replace('.txt', '')
        img_num = img_id.replace('Img', '').replace('IMG', '')

        # Ищем изображение
        img_path = None
        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
            p = os.path.join(DMID_IMGS, f"{img_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break
            # Пробуем с номером
            p2 = os.path.join(DMID_IMGS, f"IMG{img_num}{ext}")
            if os.path.exists(p2):
                img_path = p2
                break

        if img_path is None:
            # Ищем любой файл с этим номером
            for f in os.listdir(DMID_IMGS):
                if img_num in f:
                    img_path = os.path.join(DMID_IMGS, f)
                    break

        # Читаем отчёт
        with open(os.path.join(DMID_REPS, rf), encoding='utf-8', errors='ignore') as f:
            report = f.read().strip()

        if img_path and report:
            pairs.append({
                "img_id": img_id,
                "img_path": img_path,
                "real_report": report
            })

    print(f"Найдено пар: {len(pairs)} из {n}")
    return pairs

def generate(model, processor, img_path):
    try:
        image = Image.open(img_path).convert("RGB").resize((448,448))
    except:
        return ""
    prompt = "<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report with breast composition, findings, BI-RADS category and recommendation.<end_of_turn>\n<start_of_turn>model\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()

def compute_metrics(refs, hyps, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1,r2,rl = [],[],[]
    for ref,hyp in zip(refs,hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)
    _,_,F = bert_score(hyps, refs, lang="en", verbose=False)
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    for k,v in m.items(): print(f"  {k:12s}: {v}")
    return m

def main():
    print("Загружаю DMID тест сет...")
    pairs = load_dmid_pairs(n=52)
    if not pairs:
        print("Не найдено пар! Проверь пути к файлам.")
        return

    refs = [p["real_report"] for p in pairs]

    print("\nЗагружаю модели...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)

    # 1. Baseline zero-shot
    print("\n[1/2] Baseline (zero-shot)...")
    base.eval()
    hyps_base = []
    for p in pairs:
        hyp = generate(base, processor, p["img_path"])
        hyps_base.append(hyp)
        print(".", end="", flush=True)
    m1 = compute_metrics(refs, hyps_base, "Baseline MedGemma (zero-shot) on DMID")

    # 2. Fine-tuned
    print("\n[2/2] Fine-tuned (ours)...")
    ft = PeftModel.from_pretrained(base, MM_LORA)
    ft.eval()
    hyps_ft = []
    for p in pairs:
        hyp = generate(ft, processor, p["img_path"])
        hyps_ft.append(hyp)
        print(".", end="", flush=True)
    m2 = compute_metrics(refs, hyps_ft, "Multimodal FT (ours) on DMID")

    # Сохраняем
    results = {
        "dataset": "DMID",
        "n_test": len(pairs),
        "note": "Evaluated against REAL radiologist reports",
        "baseline": m1,
        "multimodal_ft": m2
    }
    out = "./results/eval_dmid.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено в {out}")

    # Показываем пример
    print("\n--- Пример отчёта ---")
    print(f"РЕАЛЬНЫЙ:\n{refs[0][:300]}")
    print(f"\nНАШ:\n{hyps_ft[0][:300]}")

if __name__ == "__main__":
    main()
