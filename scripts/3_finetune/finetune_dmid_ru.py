"""
Stage 3 — Cross-lingual fine-tuning on Russian mammography reports (DMID-RU).

Continues from the two-stage English checkpoint (medgemma_dmid) and adapts the
model to generate Russian radiology reports. Per-study format: 4 views
(RCC, LCC, RMLO, LMLO) -> 1 report.

    Stage 1 (synthetic, VinDr+CBIS)  ->  medgemma_multimodal
    Stage 2 (real English, DMID)     ->  medgemma_dmid
    Stage 3 (real Russian, DMID-RU)  ->  medgemma_dmid_ru   <-- this script
"""
import os, json, random, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import PeftModel

MODEL_ID  = "google/medgemma-4b-it"
BASE_LORA = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/medgemma_dmid"   # Stage 2 (English)
OUTPUT    = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/medgemma_dmid_ru"
DATA_DIR  = "/mnt/d/dmid_ru"
PAIRS     = os.path.join(DATA_DIR, "pairs_study.json")
SPLIT_OUT = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/scripts/3_finetune/split_ru.json"

SEED = 42
VIEW_RU = {"RCC": "Правая КК", "LCC": "Левая КК", "RMLO": "Правая МЛК", "LMLO": "Левая МЛК"}
INSTRUCTION = ("Составь структурированное заключение по маммографии обеих молочных желёз "
               "на русском языке: рентгенологический тип плотности по ACR, описание (протокол), "
               "заключение, категорию BI-RADS и рекомендации.")
IMG_SIZE = 448
MAX_LEN  = 2048


def make_split(pairs):
    """Deterministic 80/10/10 split BY STUDY (no patient leakage). Saved to disk."""
    ids = [p["patient_id"] for p in pairs]
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * 0.10))
    n_val  = max(1, round(n * 0.10))
    test, val, train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    split = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    with open(SPLIT_OUT, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"Split saved -> {SPLIT_OUT}  (train {len(train)}, val {len(val)}, test {len(test)})")
    return set(train), set(val), set(test)


def build_messages(pair):
    """4 labelled views + Russian instruction as a single user turn."""
    content = []
    for img_rel, view in zip(pair["images"], pair["views"]):
        content.append({"type": "text", "text": f"{VIEW_RU.get(view, view)} проекция:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


class DMIDRuDataset(Dataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def _load_images(self, pair):
        # Pass near-native resolution; the processor downsizes to 896 once (sharper than pre-shrinking).
        imgs = []
        for rel in pair["images"]:
            try:
                im = Image.open(os.path.join(DATA_DIR, rel)).convert("RGB")
            except Exception:
                im = Image.new("RGB", (896, 896), 128)
            imgs.append(im)
        return imgs

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        images = self._load_images(pair)
        messages = build_messages(pair)

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        full = prompt + pair["report"].strip() + "<end_of_turn>\n"

        enc = self.processor(text=full, images=images, return_tensors="pt",
                             truncation=True, max_length=MAX_LEN)
        input_ids = enc["input_ids"].squeeze(0)
        attn      = enc["attention_mask"].squeeze(0)

        # Mask everything up to and including the model-turn marker; supervise only the report.
        labels = input_ids.clone()
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        cut = 0
        for i in range(len(input_ids) - len(sep)):
            if input_ids[i:i + len(sep)].tolist() == sep:
                cut = i + len(sep)
                break
        labels[:cut] = -100
        # Never compute loss on image placeholder tokens.
        img_tok = self.processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        if img_tok is not None and img_tok >= 0:
            labels[input_ids == img_tok] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pixel_values": enc["pixel_values"],          # [num_images, 3, H, W]
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(input_ids).unsqueeze(0)).squeeze(0),
            "labels": labels,
        }


class Collator:
    """Pad input_ids/attention/labels to longest in batch; concat pixel_values along image dim."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(b["input_ids"].size(0) for b in batch)
        ids, attn, tti, lab = [], [], [], []
        for b in batch:
            n = maxlen - b["input_ids"].size(0)
            pad = lambda t, v: torch.cat([t, torch.full((n,), v, dtype=t.dtype)]) if n else t
            ids.append(pad(b["input_ids"], self.pad_id))
            attn.append(pad(b["attention_mask"], 0))
            tti.append(pad(b["token_type_ids"], 0))
            lab.append(pad(b["labels"], -100))
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(attn),
            "token_type_ids": torch.stack(tti),
            "labels": torch.stack(lab),
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        }


def main():
    print("=" * 60)
    print("Stage 3 — Russian fine-tuning (DMID-RU), per-study 4-view")
    print("=" * 60)

    pairs = json.load(open(PAIRS, encoding="utf-8"))
    train_ids, val_ids, _ = make_split(pairs)
    train_pairs = [p for p in pairs if p["patient_id"] in train_ids]
    val_pairs   = [p for p in pairs if p["patient_id"] in val_ids]
    print(f"Train studies: {len(train_pairs)}, Val studies: {len(val_pairs)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print("Загружаю processor и базовую модель...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    base.config.use_cache = False

    print(f"Загружаю LoRA из {BASE_LORA} и продолжаю обучение...")
    model = PeftModel.from_pretrained(base, BASE_LORA, is_trainable=True)
    model.print_trainable_parameters()
    # Required for gradient checkpointing on a quantized base with LoRA adapters.
    model.enable_input_require_grads()

    train_ds = DMIDRuDataset(train_pairs, processor)
    val_ds   = DMIDRuDataset(val_pairs, processor)
    collator = Collator(processor.tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=OUTPUT,
        num_train_epochs=6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                     eval_dataset=val_ds, data_collator=collator)

    print("Обучаю...")
    trainer.train()
    trainer.save_model(OUTPUT)
    processor.save_pretrained(OUTPUT)
    print(f"Готово! Модель в {OUTPUT}")


if __name__ == "__main__":
    main()
