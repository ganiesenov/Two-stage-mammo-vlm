import os, json, torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import BitsAndBytesConfig

MODEL_ID  = "google/medgemma-4b-it"
BASE_LORA = "./medgemma_multimodal"
OUTPUT    = "./medgemma_dmid"
IMGS_DIR  = "./dmid/TIFF Images/TIFF Images/"
REPS_DIR  = "./dmid/Reports/Reports/"

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
        try:
            image = Image.open(item["img_path"]).convert("RGB").resize((448,448))
        except:
            image = Image.new("RGB", (448,448), 128)

        prompt = f"<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report with breast composition, findings, BI-RADS category and recommendation.<end_of_turn>\n<start_of_turn>model\n{item['report']}<end_of_turn>"

        encoding = self.processor(
            text=prompt, images=image,
            return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length"
        )
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[:] = -100

        # Маскируем всё кроме ответа модели
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        for i in range(len(input_ids) - len(sep)):
            if input_ids[i:i+len(sep)].tolist() == sep:
                labels[i+len(sep):] = input_ids[i+len(sep):]
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
    print("Fine-tuning on DMID (real reports)")
    print("="*50)

    files = sorted(os.listdir(REPS_DIR))
    train_pairs = load_pairs(files[:407])
    val_pairs   = load_pairs(files[407:458])
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print("Загружаю модель...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    base.config.use_cache = False

    # Загружаем наши веса и дообучаем дальше на DMID
    print("Загружаю наши LoRA веса...")
    model = PeftModel.from_pretrained(base, BASE_LORA, is_trainable=True)
    model.print_trainable_parameters()

    train_ds = DMIDDataset(train_pairs, processor)
    val_ds   = DMIDDataset(val_pairs, processor)

    args = TrainingArguments(
        output_dir=OUTPUT,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
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
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
    )

    print("Обучаю...")
    trainer.train()
    trainer.save_model(OUTPUT)
    processor.save_pretrained(OUTPUT)
    print(f"Готово! Модель в {OUTPUT}")

if __name__ == "__main__":
    main()
