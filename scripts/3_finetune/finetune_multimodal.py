import json
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_ID   = "google/medgemma-4b-it"
VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
CBIS_JSONL  = "./cbis-ddsm/synthetic_reports_cbis.jsonl"
CBIS_MAP   = "./cbis-ddsm/image_mapping.csv"
VINDR_IMG  = "./vindr"
OUTPUT_DIR = "./medgemma_multimodal"

def load_paired_data():
    pairs = []

    # VinDr
    with open(VINDR_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("is_valid"): continue
                img_path = f"{VINDR_IMG}/{r['study_id']}/{r['image_id']}.png"
                if Path(img_path).exists():
                    pairs.append({"image_path": img_path, "report": r["synthetic_report"],
                                  "findings": r["finding_categories"],
                                  "birads": r["breast_birads"], "dataset": "vindr"})
            except: pass

    # CBIS
    cbis_map = pd.read_csv(CBIS_MAP)
    cbis_map["laterality_short"] = cbis_map["laterality"].str[:1]  # L/R
    with open(CBIS_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("is_valid"): continue
                pid = r["patient_id"]
                lat = r["laterality"].strip().upper()[:1]
                view = r["image_view"].strip().upper()
                match = cbis_map[
                    (cbis_map["patient_id"] == pid) &
                    (cbis_map["laterality"].str.startswith(lat)) &
                    (cbis_map["view"] == view)
                ]
                if len(match) > 0:
                    img_path = match.iloc[0]["jpeg_path"]
                    if Path(img_path).exists():
                        pairs.append({"image_path": img_path, "report": r["synthetic_report"],
                                      "findings": r["finding_type"],
                                      "birads": r["assessment"], "dataset": "cbis"})
            except: pass

    print(f"VinDr пар: {sum(1 for p in pairs if p['dataset']=='vindr')}")
    print(f"CBIS пар:  {sum(1 for p in pairs if p['dataset']=='cbis')}")
    print(f"Итого:     {len(pairs)}")
    return pairs

class MammoDataset(Dataset):
    def __init__(self, pairs, processor, max_length=512):
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB").resize((448, 448))
        except:
            image = Image.new("RGB", (448, 448), color=128)

        prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report.\nFindings: {item['findings']}\nBI-RADS: {item['birads']}<end_of_turn>\n<start_of_turn>model\n{item['report']}<end_of_turn>"

        encoding = self.processor(
            text=prompt, images=image,
            return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length"
        )
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        # Маскируем промпт — учим только на отчёте
        sep = "<start_of_turn>model\n"
        sep_ids = self.processor.tokenizer.encode(sep, add_special_tokens=False)
        for i in range(len(input_ids) - len(sep_ids)):
            if input_ids[i:i+len(sep_ids)].tolist() == sep_ids:
                labels[:i+len(sep_ids)] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "pixel_values": encoding["pixel_values"].squeeze(),
            "token_type_ids": torch.zeros_like(input_ids),
            "labels": labels,
        }

def main():
    print("="*50)
    print("Multimodal Fine-tuning MedGemma-4B")
    print("="*50)

    pairs = load_paired_data()
    import random; random.shuffle(pairs)
    n_val = max(5, int(len(pairs)*0.1))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print("\nЗагружаю модель...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = MammoDataset(train_pairs, processor)
    val_dataset   = MammoDataset(val_pairs, processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\nНачинаю обучение...")
    trainer.train()

    print("\nСохраняю модель...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Готово! Модель в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
