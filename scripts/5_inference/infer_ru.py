"""
Инференс русской модели маммографических заключений (Stage 3, medgemma_dmid_ru).

Использование:
    # один пациент (папка с PNG-снимками):
    python infer_ru.py /path/to/study_folder

    # пакетно (родительская папка, в каждой подпапке — снимки одного пациента):
    python infer_ru.py /path/to/parent_folder --batch

В папке должно быть до 4 снимков. Проекции определяются по имени файла
(RCC / LCC / RMLO / LMLO, регистр не важен). Если метки не найдены — берутся
до 4 изображений в алфавитном порядке без подписи проекции.

Результат печатается в консоль и сохраняется как report_ru.txt в папке исследования.
"""
import os, sys, argparse, glob, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "google/medgemma-4b-it"
LORA     = "./medgemma_dmid_ru"

VIEW_RU = {"RCC": "Правая КК", "LCC": "Левая КК", "RMLO": "Правая МЛК", "LMLO": "Левая МЛК"}
VIEW_ORDER = ["RCC", "LCC", "RMLO", "LMLO"]
INSTRUCTION = ("Составь структурированное заключение по маммографии обеих молочных желёз "
               "на русском языке: рентгенологический тип плотности по ACR, описание (протокол), "
               "заключение, категорию BI-RADS и рекомендации.")
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.tif", "*.tiff")


def detect_view(filename):
    name = filename.upper()
    # порядок проверки важен: MLO до CC, чтобы 'RMLO' не схватился как 'R..CC'
    for v in ["RMLO", "LMLO", "RCC", "LCC"]:
        if v in name:
            return v
    return None


def collect_images(folder):
    """Вернуть [(view|None, path)] в порядке RCC,LCC,RMLO,LMLO; неразмеченные — в конце."""
    files = []
    for pat in EXTS:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files = sorted(set(files))
    tagged = {}
    untagged = []
    for f in files:
        v = detect_view(os.path.basename(f))
        if v and v not in tagged:
            tagged[v] = f
        else:
            untagged.append(f)
    ordered = [(v, tagged[v]) for v in VIEW_ORDER if v in tagged]
    for f in untagged:
        if len(ordered) >= 4:
            break
        ordered.append((None, f))
    return ordered[:4]


def build_messages(views):
    content = []
    for v, _ in views:
        if v:
            content.append({"type": "text", "text": f"{VIEW_RU[v]} проекция:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, LORA)
    model.eval()
    return model, processor


def infer_folder(model, processor, folder):
    views = collect_images(folder)
    if not views:
        return None, "нет снимков в папке"
    images = [Image.open(p).convert("RGB") for _, p in views]
    prompt = processor.apply_chat_template(build_messages(views), add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    report = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
    used = ", ".join(v or f"[{os.path.basename(p)}]" for v, p in views)
    return report, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="папка исследования (или родительская папка с --batch)")
    ap.add_argument("--batch", action="store_true", help="обработать все подпапки")
    args = ap.parse_args()

    if args.batch:
        folders = sorted(d for d in glob.glob(os.path.join(args.path, "*")) if os.path.isdir(d))
    else:
        folders = [args.path]
    if not folders:
        print("Папки не найдены."); sys.exit(1)

    print(f"Загружаю модель ({LORA})...")
    model, processor = load_model()

    for folder in folders:
        report, used = infer_folder(model, processor, folder)
        name = os.path.basename(os.path.normpath(folder))
        print("\n" + "=" * 70)
        print(f"ИССЛЕДОВАНИЕ: {name}   (снимки: {used})")
        print("=" * 70)
        if report is None:
            print("  ! " + used)
            continue
        print(report)
        out_path = os.path.join(folder, "report_ru.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n[сохранено -> {out_path}]")


if __name__ == "__main__":
    main()
