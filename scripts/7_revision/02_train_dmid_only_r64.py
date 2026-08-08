"""
DMID-only при r=64, α=64 — бейзлайн с той же ёмкостью адаптера (GPU).

Зачем (главная методологическая правка ревизии, см. REVISION_PLAN.md п.1.1):
в Table 4 строка «Ours (two-stage)» — это модель r=64/α=64 из абляции,
а строка «DMID-only (LoRA)» — модель r=16/α=32. Заявленный выигрыш смешивает
двухэтапное обучение с четырёхкратным ростом ёмкости адаптера. Рецензент 1 (п.11)
и рецензент 3 (пп. 2, 7) подошли к этому вплотную.

Этот прогон даёт недостающую клетку: обучение ТОЛЬКО на DMID, но при r=64/α=64,
всё остальное идентично. После него Table 4 можно строить парами при равном ранге.

Гиперпараметры совпадают с lora_ablation.py, единственное отличие —
базовая модель не получает веса Stage 1.
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, MODEL_ID, IMG_SIZE, PROMPT_TEXT, split_files, load_pairs, _bnb

from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model

OUT = f"{ROOT}/dmid_only_r64"
R, ALPHA = 64, 64


class DMIDDataset(Dataset):
    def __init__(self, pairs, processor, max_length=512):
        self.pairs, self.processor, self.max_length = pairs, processor, max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            image = Image.open(item["img_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 128)
        prompt = (f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n"
                  f"<start_of_turn>model\n{item['report']}<end_of_turn>")
        enc = self.processor(text=prompt, images=image, return_tensors="pt",
                             truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = enc["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[:] = -100
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        for i in range(len(input_ids) - len(sep)):
            if input_ids[i:i + len(sep)].tolist() == sep:
                labels[i + len(sep):] = input_ids[i + len(sep):]
                break
        return {"input_ids": input_ids,
                "attention_mask": enc["attention_mask"].squeeze(),
                "pixel_values": enc["pixel_values"].squeeze(),
                "token_type_ids": torch.zeros_like(input_ids),
                "labels": labels}


def main():
    torch.manual_seed(42)
    splits = split_files()
    train_pairs = load_pairs(splits["train"])
    val_pairs = load_pairs(splits["val"])
    print(f"Train {len(train_pairs)} / Val {len(val_pairs)}  —  DMID-only, r={R}, α={ALPHA}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=_bnb(), device_map="auto", dtype=torch.bfloat16)
    model.config.use_cache = False

    # Ключевое отличие от two-stage: Stage 1 НЕ загружается и не мержится.
    lora = LoraConfig(r=R, lora_alpha=ALPHA,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=5,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=20,
        logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, bf16=True, report_to="none",
        optim="paged_adamw_8bit", remove_unused_columns=False, seed=42)

    trainer = Trainer(model=model, args=args,
                      train_dataset=DMIDDataset(train_pairs, processor),
                      eval_dataset=DMIDDataset(val_pairs, processor))
    trainer.train()

    # В отличие от lora_ablation.py сохраняем верхнеуровневый адаптер явно,
    # чтобы модель можно было загрузить без выбора чекпойнта вручную.
    trainer.save_model(OUT)
    processor.save_pretrained(OUT)
    print(f"\nГотово → {OUT}")
    print("Дальше: python 01_infer.py --split test --models dmid_only_r64")


if __name__ == "__main__":
    main()
