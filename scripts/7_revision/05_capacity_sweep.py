"""
Свип по ёмкости адаптера × две ветви × seed'ы (GPU, длинный прогон).

Проверяет утверждение о ВЗАИМОДЕЙСТВИИ: выигрыш синтетического претрейна велик при малой
ёмкости адаптера и стремится к нулю при большой. Условия и правила решения зафиксированы
заранее в results/revision/PREREGISTRATION.md (порог эквивалентности Δ_eq = 0.02 ROUGE-L).

Дизайн:
  ранги 8/16/32/64, α = 2r, ветви {two_stage, dmid_only}, seeds {42,43,44}
  Stage 1 переобучается ПОД КАЖДЫЙ SEED — иначе seed варьировал бы только Stage 2,
  и все two-stage прогоны стартовали бы с одного чекпойнта.

Порядок исполнения выбран так, чтобы сначала закрылись опорные точки утверждения
(r=16 и r=64 по всем seed'ам), а форма кривой (r=8, r=32) достраивалась после.
Результаты пишутся инкрементально: прогон можно прервать в любой момент,
всё уже посчитанное останется пригодным.

Запуск:  nohup $PY scripts/7_revision/05_capacity_sweep.py > logs/sweep.log 2>&1 &
Возобновление: повторный запуск пропускает уже сделанные ячейки.
"""
import os, sys, json, gc, time, argparse

sys.path.insert(0, os.path.dirname(__file__))
from _common import (ROOT, MODEL_ID, IMG_SIZE, PROMPT_TEXT, split_files, load_pairs,
                     _bnb, load_pairs as _lp, generate_all, save_preds, corpus_scores)

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, PeftModel

SWEEP_DIR = f"{ROOT}/capacity_sweep"
RESULTS = f"{ROOT}/results/revision/capacity_sweep.json"
VINDR_JSONL = f"{ROOT}/vindr/synthetic_reports.jsonl"
CBIS_JSONL = f"{ROOT}/cbis-ddsm/synthetic_reports_cbis.jsonl"

# Конфигурации задаются явными парами (r, α), а не правилом α=2r.
# Причина: ключевой результат ревизии (Δ=+0.0009 при r=64) получен на α=64, тогда как
# правило α=2r дало бы при r=64 значение α=128 — в Table 8 это другая конфигурация
# с измеримо иным качеством (0.6663 против 0.6705 ROUGE-L). Поэтому опубликованная
# точка (64, 64) включена отдельно, вне ряда постоянного отношения α/r.
#
# Порядок: сначала обе опорные точки утверждения, затем форма кривой при α=2r.
CONFIGS = [
    (16, 32),    # якорь: совпадает и с опубликованным two-stage r=16, и с рядом α=2r
    (64, 64),    # якорь: опубликованная конфигурация и точка ключевого результата
    (64, 128),   # ряд α=2r
    (32, 64),    # ряд α=2r
    (8, 16),     # ряд α=2r
]
SEEDS = [42, 43, 44]
ARMS = ["two_stage", "dmid_only"]


# ─────────────────────── данные ───────────────────────

class PairDataset(Dataset):
    def __init__(self, pairs, processor, key_img, key_txt, max_length=512):
        self.pairs, self.processor = pairs, processor
        self.ki, self.kt, self.max_length = key_img, key_txt, max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            image = Image.open(item[self.ki]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 128)
        prompt = (f"<start_of_turn>user\n<start_of_image>\n{PROMPT_TEXT}<end_of_turn>\n"
                  f"<start_of_turn>model\n{item[self.kt]}<end_of_turn>")
        enc = self.processor(text=prompt, images=image, return_tensors="pt",
                             truncation=True, max_length=self.max_length, padding="max_length")
        ids = enc["input_ids"].squeeze()
        labels = ids.clone()
        labels[:] = -100
        sep = self.processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        for i in range(len(ids) - len(sep)):
            if ids[i:i + len(sep)].tolist() == sep:
                labels[i + len(sep):] = ids[i + len(sep):]
                break
        return {"input_ids": ids, "attention_mask": enc["attention_mask"].squeeze(),
                "pixel_values": enc["pixel_values"].squeeze(),
                "token_type_ids": torch.zeros_like(ids), "labels": labels}


def load_synthetic_v2(dose):
    """Корпус v2 (12_generate_synth_v2.py): префикс длины `dose` из общего пула.

    Дозы вложены по построению (369 ⊂ 800 ⊂ 1444) и одинаковы по составу:
    источники 50/50, VinDr — только находки, CBIS — только mass, кириллицы нет.
    Между дозами меняется единственный фактор — N.
    """
    path = f"{ROOT}/data/synthetic_v2.jsonl"
    if not os.path.exists(path):
        sys.exit(f"Корпус v2 не найден: {path} — сначала 12_generate_synth_v2.py")
    recs = [json.loads(l) for l in open(path)]
    recs.sort(key=lambda r: r["order"])
    if dose > len(recs):
        sys.exit(f"Запрошена доза {dose}, а в корпусе {len(recs)} записей")
    recs = recs[:dose]
    pairs = [{"img": r["img"], "txt": r["synthetic_report"]}
             for r in recs if os.path.exists(r["img"])]
    n_v = sum(1 for r in recs if r["source"] == "vindr")
    print(f"  синтетика v2: доза {dose} (vindr {n_v}, cbis {len(recs)-n_v}), "
          f"пар с картинкой {len(pairs)}")
    return pairs


def load_synthetic(drop_russian=False):
    """Синтетические пары Stage 1 (VinDr + CBIS), только прошедшие валидацию.

    drop_russian=True убирает 21 отчёт, сгенерированный Llama-3.1-8B на русском языке
    вопреки англоязычному промпту (все прошли автоматическую валидацию). Это ветка
    абляции контаминации — EXPLORATORY, в PREREGISTRATION.md её нет.

    Логика сопоставления изображений повторяет scripts/3_finetune/finetune_multimodal.py:
    у VinDr путь собирается из study_id/image_id, у CBIS требуется image_mapping.csv
    (в самом jsonl пути к изображению нет — только patient_id, laterality, view).

    Из 400 сгенерированных отчётов is_valid=True имеют 390 (196 VinDr + 194 CBIS);
    в статье заявлено 400 — расхождение зафиксировано в REVISION_PLAN.md п. 1.5.
    """
    import pandas as pd
    import re as _re
    CYR = _re.compile(r'[а-яА-ЯёЁ]')
    pairs = []

    n_v = n_dropped = 0
    for line in open(VINDR_JSONL):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("is_valid"):
            continue
        txt = r["synthetic_report"]
        if drop_russian and CYR.search(txt):
            n_dropped += 1
            continue
        p = f"{ROOT}/vindr/{r['study_id']}/{r['image_id']}.png"
        if os.path.exists(p):
            pairs.append({"img": p, "txt": txt})
            n_v += 1

    n_c = 0
    cbis_map_path = f"{ROOT}/cbis-ddsm/image_mapping.csv"
    if os.path.exists(cbis_map_path):
        cm = pd.read_csv(cbis_map_path)
        for line in open(CBIS_JSONL):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("is_valid"):
                continue
            lat = str(r["laterality"]).strip().upper()[:1]
            view = str(r["image_view"]).strip().upper()
            m = cm[(cm["patient_id"] == r["patient_id"])
                   & (cm["laterality"].astype(str).str.upper().str.startswith(lat))
                   & (cm["view"].astype(str).str.upper() == view)]
            if len(m) and os.path.exists(str(m.iloc[0]["jpeg_path"])):
                pairs.append({"img": str(m.iloc[0]["jpeg_path"]), "txt": r["synthetic_report"]})
                n_c += 1

    tag = f"  (отброшено русскоязычных: {n_dropped})" if drop_russian else ""
    print(f"  синтетика: VinDr {n_v} + CBIS {n_c} = {len(pairs)} пар{tag}")
    return pairs


# ─────────────────────── обучение ───────────────────────

def base_model():
    m = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=_bnb(), device_map="auto", dtype=torch.bfloat16)
    m.config.use_cache = False
    return m


def targs(out, seed, epochs, lr, warmup, val_ds=None):
    """Аргументы обучения.

    Для Stage 2 обязательно load_best_model_at_end: опубликованный протокол
    (lora_ablation.py) отбирает лучшую эпоху по валидации, и лучшей везде оказывается
    4-я, а не последняя (val loss 0.6645 против 0.711). Взятие финальной эпохи
    систематически занижало бы качество в обеих ветвях и делало бы свип
    несопоставимым с Table 8.
    """
    use_val = val_ds is not None
    return TrainingArguments(
        output_dir=out, num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=warmup,
        logging_steps=50,
        eval_strategy="epoch" if use_val else "no",
        save_strategy="epoch" if use_val else "no",
        load_best_model_at_end=use_val,
        save_total_limit=1,
        bf16=True, report_to="none", optim="paged_adamw_8bit",
        remove_unused_columns=False, seed=seed, data_seed=seed)


# Метка корпуса Stage 1: "" — опубликованный корпус как есть, "_clean" — без
# русскоязычной контаминации. Входит в имена каталогов и ключи результатов,
# чтобы две ветки не смешивались.
CORPUS_TAG = ""


def train_stage1(seed, processor, synth):
    """Stage 1 под конкретный seed. Ровно 3 минуты, поэтому переобучаем, а не переиспользуем."""
    out = f"{SWEEP_DIR}/stage1{CORPUS_TAG}_seed{seed}"
    if os.path.exists(f"{out}/adapter_model.safetensors"):
        print(f"  Stage 1 (seed {seed}) уже есть")
        return out
    print(f"  Stage 1 (seed {seed}): {len(synth)} синтетических пар", flush=True)
    torch.manual_seed(seed)
    m = base_model()
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    tr = Trainer(model=m, args=targs(out, seed, 3, 2e-4, 100),
                 train_dataset=PairDataset(synth, processor, "img", "txt"))
    tr.train()
    tr.save_model(out)
    del m, tr
    gc.collect(); torch.cuda.empty_cache()
    return out


def cell_key(arm, r, alpha, seed):
    """Ключ обязан содержать α: при r=64 конфигурации α=64 и α=128 различны.

    CORPUS_TAG добавляется ТОЛЬКО для two_stage. У dmid_only нет Stage 1, поэтому
    от корпуса и от дозы синтетики он не зависит вообще — его ячейки переиспользуются
    как общий компаратор для всех доз. Иначе сетка 3 ранга × 3 дозы × 3 сида
    насчитала бы 27 ячеек dmid_only вместо 9, то есть ~12 лишних часов GPU
    на переобучение идентичных моделей.
    """
    tag = CORPUS_TAG if arm == "two_stage" else ""
    return f"{arm}{tag}_r{r}_a{alpha}_seed{seed}"


def train_cell(arm, r, alpha, seed, processor, train_pairs, val_pairs, stage1_dir):
    out = f"{SWEEP_DIR}/{cell_key(arm, r, alpha, seed)}"
    if os.path.exists(f"{out}/adapter_model.safetensors"):
        print(f"  {arm} r={r} α={alpha} seed={seed} уже обучен")
        return out
    print(f"  обучение: {arm} r={r} α={alpha} seed={seed}", flush=True)
    torch.manual_seed(seed)
    m = base_model()
    if arm == "two_stage":
        # Веса Stage 1 вливаются в базу, поверх ставится новый адаптер Stage 2.
        m = PeftModel.from_pretrained(m, stage1_dir)
        m = m.merge_and_unload()
    m = get_peft_model(m, LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    val_ds = PairDataset(val_pairs, processor, "img_path", "report")
    tr = Trainer(model=m, args=targs(out, seed, 5, 1e-4, 20, val_ds=val_ds),
                 train_dataset=PairDataset(train_pairs, processor, "img_path", "report"),
                 eval_dataset=val_ds)
    tr.train()
    tr.save_model(out)
    del m, tr
    gc.collect(); torch.cuda.empty_cache()
    return out


def eval_cell(adapter_dir, arm, r, alpha, seed, processor, test_pairs, stage1_dir):
    m = base_model()
    m.config.use_cache = True
    if arm == "two_stage":
        m = PeftModel.from_pretrained(m, stage1_dir)
        m = m.merge_and_unload()
    m = PeftModel.from_pretrained(m, adapter_dir)
    m.eval()
    preds = generate_all(m, processor, test_pairs, log_every=26)
    sc = corpus_scores(preds)
    save_preds(f"sweep_{cell_key(arm, r, alpha, seed)}", "test", preds,
               meta={"arm": arm, "r": r, "alpha": alpha, "seed": seed, "corpus_scores": sc})
    del m
    gc.collect(); torch.cuda.empty_cache()
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None,
                    help="пары r:alpha, напр. 16:32 64:64")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=["two_stage", "dmid_only"])
    ap.add_argument("--clean-corpus", action="store_true",
                    help="EXPLORATORY: Stage 1 без 21 русскоязычного отчёта")
    ap.add_argument("--synth-v2", action="store_true",
                    help="EXPLORATORY: корпус v2 с языковым guard'ом (12_generate_synth_v2.py)")
    ap.add_argument("--dose", type=int, default=None,
                    help="размер префикса корпуса v2: 369 / 800 / 1444")
    args = ap.parse_args()

    global CORPUS_TAG
    if args.clean_corpus:
        CORPUS_TAG = "_clean"
        print("ВЕТКА: чистый корпус Stage 1 (EXPLORATORY, вне пререгистрации)")
    if args.synth_v2:
        if args.dose is None:
            sys.exit("--synth-v2 требует --dose")
        CORPUS_TAG = f"_v2d{args.dose}"
        print(f"ВЕТКА: корпус v2, доза {args.dose} (EXPLORATORY, вне пререгистрации)")

    configs = ([tuple(int(x) for x in c.split(":")) for c in args.configs]
               if args.configs else CONFIGS)

    os.makedirs(SWEEP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    splits = split_files()
    train_pairs = load_pairs(splits["train"])
    val_pairs = load_pairs(splits["val"])
    test_pairs = load_pairs(splits["test"])
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    synth = (load_synthetic_v2(args.dose) if args.synth_v2
             else load_synthetic(drop_russian=args.clean_corpus))
    print(f"DMID train {len(train_pairs)} / test {len(test_pairs)}   синтетика {len(synth)}")
    if not synth:
        sys.exit("Синтетические пары не найдены — Stage 1 обучить нельзя")

    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}

    # Stage 1 нужен только ветви two_stage; при прогоне одного dmid_only не тратим GPU.
    stage1 = {}
    if "two_stage" in args.arms:
        for seed in args.seeds:
            stage1[seed] = train_stage1(seed, processor, synth)
    else:
        stage1 = {s: None for s in args.seeds}

    t0 = time.time()
    for r, alpha in configs:
        for seed in args.seeds:
            for arm in args.arms:
                key = cell_key(arm, r, alpha, seed)
                if key in results:
                    print(f"[{key}] готово, пропуск")
                    continue
                print(f"\n{'='*70}\n  {key}   (прошло {(time.time()-t0)/60:.0f} мин)\n{'='*70}", flush=True)
                d = train_cell(arm, r, alpha, seed, processor, train_pairs, val_pairs, stage1[seed])
                sc = eval_cell(d, arm, r, alpha, seed, processor, test_pairs, stage1[seed])
                results[key] = {"arm": arm, "r": r, "alpha": alpha, "seed": seed, **sc}
                with open(RESULTS, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  {sc}", flush=True)

        # промежуточная сводка по конфигурации
        sel = lambda a: [v["rougeL"] for v in results.values()
                         if v["r"] == r and v["alpha"] == alpha and v["arm"] == a]
        ts, do = sel("two_stage"), sel("dmid_only")
        if ts and do:
            print(f"\n  >>> r={r} α={alpha}: two-stage {sum(ts)/len(ts):.4f} (n={len(ts)})  "
                  f"dmid-only {sum(do)/len(do):.4f} (n={len(do)})  "
                  f"Δ={sum(ts)/len(ts)-sum(do)/len(do):+.4f}", flush=True)

    print(f"\nГотово за {(time.time()-t0)/60:.0f} мин → {RESULTS}")
    print("Дальше: 06_equivalence.py — TOST по заранее объявленному порогу")


if __name__ == "__main__":
    main()
