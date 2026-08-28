"""
Stage 3 v2 — Russian fine-tune with DIAGNOSIS-FOCUSED loss.

Same data/format as finetune_dmid_ru.py, but addresses the mode-collapse where the
model defaults BI-RADS to "2" and ACR to "В": the standard uniform per-token loss
dilutes the single category token across an ~80-word report, so the model never
prioritises it. Two honest, data-supported changes (train has 51/172 suspicious
studies and a well-spread ACR distribution, so the signal exists):

  1. TOKEN-WEIGHTED LOSS — up-weight the tokens carrying the BI-RADS category and
     the ACR density letter (and their markers) by CAT_WEIGHT.
  2. OVERSAMPLING — duplicate suspicious studies (BI-RADS 4-6) OVERSAMPLE_SUSP x.

Trains to a NEW dir (medgemma_dmid_ru_v2); the original model is left untouched.
Evaluate with:  python scripts/4_evaluate/evaluate_dmid_ru.py --lora .../medgemma_dmid_ru_v2
"""
import os, re, json, random, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import PeftModel

MODEL_ID  = "google/medgemma-4b-it"
BASE_LORA = "./medgemma_dmid"   # Stage 2 (English)
OUTPUT    = "./medgemma_dmid_ru_v2"
DATA_DIR  = "/mnt/d/dmid_ru"
PAIRS     = os.path.join(DATA_DIR, "pairs_study.json")
SPLIT     = "./scripts/3_finetune/split_ru.json"

SEED = 42
CAT_WEIGHT      = 6.0   # loss multiplier on BI-RADS / ACR category tokens
OVERSAMPLE_SUSP = 2     # how many times to repeat BI-RADS 4-6 studies in train
VIEW_RU = {"RCC": "Правая КК", "LCC": "Левая КК", "RMLO": "Правая МЛК", "LMLO": "Левая МЛК"}
INSTRUCTION = ("Составь структурированное заключение по маммографии обеих молочных желёз "
               "на русском языке: рентгенологический тип плотности по ACR, описание (протокол), "
               "заключение, категорию BI-RADS и рекомендации.")
MAX_LEN  = 2048


def birads_of(text):
    m = re.search(r"BI-?RADS\s*[:\-]?\s*([0-6])(?:\s*/\s*([0-6]))?", text, re.I)
    if not m: return None
    return max([int(m.group(1))] + ([int(m.group(2))] if m.group(2) else []))


def build_messages(pair):
    content = []
    for img_rel, view in zip(pair["images"], pair["views"]):
        content.append({"type": "text", "text": f"{VIEW_RU.get(view, view)} проекция:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


def find_subseq(hay, needle):
    """Yield start indices where list `needle` occurs in list `hay`."""
    if not needle: return
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            yield i


class DMIDRuDataset(Dataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor
        tok = processor.tokenizer
        # marker token sequences whose following tokens carry the category value.
        # include leading-space variants — SentencePiece tokenises " категория"
        # (mid-text) differently from "категория" (start), so both are needed.
        bases = ["BI-RADS", "категория", "категории", "ACR", "категория -", "категория –"]
        seen = set()
        self.markers = []
        for b in bases:
            for variant in (b, " " + b):
                ids = tok.encode(variant, add_special_tokens=False)
                key = tuple(ids)
                if ids and key not in seen:
                    seen.add(key); self.markers.append(ids)

    def __len__(self):
        return len(self.pairs)

    def _load_images(self, pair):
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

        labels = input_ids.clone()
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        cut = 0
        for i in range(len(input_ids) - len(sep)):
            if input_ids[i:i + len(sep)].tolist() == sep:
                cut = i + len(sep); break
        labels[:cut] = -100
        img_tok = self.processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        if img_tok is not None and img_tok >= 0:
            labels[input_ids == img_tok] = -100

        # per-token loss weights: 1.0 default, CAT_WEIGHT on marker + next few tokens
        weights = torch.ones_like(input_ids, dtype=torch.float32)
        ids_list = input_ids.tolist()
        for mk in self.markers:
            for s in find_subseq(ids_list, mk):
                if s < cut:  # only inside the report
                    continue
                lo, hi = s, min(len(weights), s + len(mk) + 4)  # marker + value tokens
                weights[lo:hi] = CAT_WEIGHT
        weights[labels == -100] = 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pixel_values": enc["pixel_values"],
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(input_ids).unsqueeze(0)).squeeze(0),
            "labels": labels,
            "loss_weight": weights,
        }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(b["input_ids"].size(0) for b in batch)
        ids, attn, tti, lab, lw = [], [], [], [], []
        for b in batch:
            n = maxlen - b["input_ids"].size(0)
            pad = lambda t, v: torch.cat([t, torch.full((n,), v, dtype=t.dtype)]) if n else t
            ids.append(pad(b["input_ids"], self.pad_id))
            attn.append(pad(b["attention_mask"], 0))
            tti.append(pad(b["token_type_ids"], 0))
            lab.append(pad(b["labels"], -100))
            lw.append(pad(b["loss_weight"], 0.0))
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(attn),
            "token_type_ids": torch.stack(tti),
            "labels": torch.stack(lab),
            "loss_weight": torch.stack(lw),
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        }


class WeightedTrainer(Trainer):
    """Per-token weighted cross-entropy so category tokens dominate the gradient."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("loss_weight")
        labels  = inputs["labels"]
        outputs = model(**inputs)
        logits  = outputs.logits
        # shift
        sl = logits[:, :-1, :].contiguous()
        sl_lab = labels[:, 1:].contiguous()
        sl_w   = weights[:, 1:].contiguous()
        loss_tok = torch.nn.functional.cross_entropy(
            sl.view(-1, sl.size(-1)).float(),
            sl_lab.view(-1),
            ignore_index=-100, reduction="none",
        ).view(sl_lab.size())
        w = sl_w
        denom = w.sum().clamp_min(1.0)
        loss = (loss_tok * w).sum() / denom
        return (loss, outputs) if return_outputs else loss


def main():
    print("=" * 60)
    print("Stage 3 v2 — diagnosis-focused (weighted loss + oversampling)")
    print("=" * 60)

    split = json.load(open(SPLIT, encoding="utf-8"))
    train_ids, val_ids = set(split["train"]), set(split["val"])
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    train_pairs = [p for p in pairs if p["patient_id"] in train_ids]
    val_pairs   = [p for p in pairs if p["patient_id"] in val_ids]

    # oversample suspicious (BI-RADS 4-6)
    extra = []
    for p in train_pairs:
        if (birads_of(p["report"]) or 0) >= 4:
            extra += [p] * (OVERSAMPLE_SUSP - 1)
    random.Random(SEED).shuffle(extra)
    train_pairs = train_pairs + extra
    random.Random(SEED).shuffle(train_pairs)
    print(f"Train studies: {len(train_pairs)} (incl. +{len(extra)} oversampled suspicious), Val: {len(val_pairs)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    base.config.use_cache = False

    print(f"Continue from Stage-2 LoRA: {BASE_LORA}")
    model = PeftModel.from_pretrained(base, BASE_LORA, is_trainable=True)
    model.print_trainable_parameters()
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

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds,
                             eval_dataset=val_ds, data_collator=collator)
    print("Training...")
    trainer.train()
    trainer.save_model(OUTPUT)
    processor.save_pretrained(OUTPUT)
    print(f"Done -> {OUTPUT}")


if __name__ == "__main__":
    main()
