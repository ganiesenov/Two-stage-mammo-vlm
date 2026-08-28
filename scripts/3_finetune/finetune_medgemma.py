import json
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

VINDR_JSONL = "./vindr/synthetic_reports.jsonl"
CBIS_JSONL  = "./cbis-ddsm/synthetic_reports_cbis.jsonl"
OUTPUT_DIR  = "./medgemma_finetuned"
MODEL_ID    = "google/medgemma-4b-it"
MAX_LENGTH  = 512

def load_reports(*paths):
    data = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("is_valid") and r.get("synthetic_report"):
                        data.append(r)
                except: pass
    print(f"Загружено отчётов: {len(data)}")
    return data

def format_prompt(report):
    findings = report.get("finding_categories", report.get("finding_type", "mass"))
    birads = report.get("breast_birads", report.get("assessment", ""))
    density = report.get("breast_density", "")
    laterality = report.get("laterality", "")
    user_msg = f"Generate a structured mammography report.\nFindings: {findings}\nBI-RADS: {birads}\nDensity: {density}\nLaterality: {laterality}"
    assistant_msg = report["synthetic_report"]
    return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n{assistant_msg}<end_of_turn>"

def tokenize(batch, tokenizer):
    out = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    # token_type_ids — все нули (текст, не изображение)
    out["token_type_ids"] = [[0]*MAX_LENGTH for _ in batch["text"]]
    return out

def main():
    print("="*50)
    print("Fine-tuning MedGemma-4B with LoRA")
    print("="*50)

    reports = load_reports(VINDR_JSONL, CBIS_JSONL)
    texts = [format_prompt(r) for r in reports]
    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=0.1, seed=42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nЗагружаю модель...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nТокенизирую датасет...")
    tok_train = split["train"].map(lambda b: tokenize(b, tokenizer), batched=True, remove_columns=["text"])
    tok_val   = split["test"].map(lambda b: tokenize(b, tokenizer), batched=True, remove_columns=["text"])
    tok_train.set_format("torch")
    tok_val.set_format("torch")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\nНачинаю обучение...")
    trainer.train()

    print("\nСохраняю модель...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Готово! Модель в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
