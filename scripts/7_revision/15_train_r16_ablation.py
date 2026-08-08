"""
Дообучение конфигурации r=16 α=32 тем же механизмом, что и остальные строки Table 8 (GPU).

Зачем: в исходной абляции строка r16_a32 была ЗАХАРДКОЖЕНА (lora_ablation.py:126-127)
числами более раннего прогона, полученного другим механизмом переноса Stage 1 → Stage 2
(продолжение обучения того же адаптера r=16 вместо слияния Stage 1 в базовые веса).
Из-за этого строка несопоставима с остальными шестью, что и есть та самая неравномерность
условий, за которую зацепились R1.13 и R3.2.

Здесь конфигурация обучается ровно так же, как остальные шесть: адаптер Stage 1 мержится
в базовую модель, поверх ставится новый адаптер r=16 α=32. Все гиперпараметры совпадают
с lora_ablation.py дословно (5 эпох, bs=1, grad_accum=8, lr=1e-4, cosine, warmup 20,
отбор лучшего чекпойнта по eval_loss).

Результат сохраняется в lora_ablation/r16_a32/ верхним уровнем (лучший чекпойнт),
после чего оценивается через 01_infer.py на валидации и тесте.
"""
import os, sys, json, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, MODEL_ID, STAGE1, REPS_DIR, split_files, load_pairs, PROMPTS

OUT_DIR = f"{ROOT}/lora_ablation/r16_a32"
# Промпт B — тот же, которым обучались все конфигурации Table 8.
PROMPT_TEXT = PROMPTS["B"]
R, ALPHA = 16, 32


class DMIDDataset(Dataset):
    """Копия датасета из lora_ablation.py: те же 448 px, max_length 512, padding max_length
    и то же маскирование префикса, иначе обучение не будет тождественным."""

    def __init__(self, pairs, processor, max_length=512):
        self.pairs, self.processor, self.max_length = pairs, processor, max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            image = Image.open(item["img_path"]).convert("RGB").resize((448, 448))
        except Exception:
            image = Image.new("RGB", (448, 448), 128)
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
    if os.path.exists(f"{OUT_DIR}/adapter_model.safetensors"):
        print(f"Адаптер уже обучен: {OUT_DIR}")
        return

    sp = split_files()
    train_pairs = load_pairs(sp["train"])
    val_pairs = load_pairs(sp["val"])
    print(f"Train {len(train_pairs)} / Val {len(val_pairs)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    base.config.use_cache = False

    # Тот же механизм, что у остальных шести конфигураций: Stage 1 вливается в базовые веса.
    model = PeftModel.from_pretrained(base, STAGE1)
    model = model.merge_and_unload()

    model = get_peft_model(model, LoraConfig(
        r=R, lora_alpha=ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {n_params:,}")

    args = TrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=5,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=20,
        logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, bf16=True, report_to="none",
        optim="paged_adamw_8bit", remove_unused_columns=False)

    trainer = Trainer(model=model, args=args,
                      train_dataset=DMIDDataset(train_pairs, processor),
                      eval_dataset=DMIDDataset(val_pairs, processor))
    trainer.train()

    # load_best_model_at_end=True — в памяти лучший по eval_loss чекпойнт, сохраняем его
    # верхним уровнем, чтобы 01_infer.py брал адаптер без угадывания номера чекпойнта.
    trainer.save_model(OUT_DIR)
    print(f"\nЛучший чекпойнт сохранён → {OUT_DIR}")

    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    for h in hist:
        print(f"  эпоха {h['epoch']:.0f}: eval_loss {h['eval_loss']:.4f}")
    with open(f"{ROOT}/results/revision/r16_a32_train_log.json", "w") as f:
        json.dump({"config": f"r={R}, alpha={ALPHA}", "mechanism": "merge_and_unload",
                   "trainable_params": n_params, "eval_history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
