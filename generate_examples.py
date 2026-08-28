import os, torch, json
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_ID   = "google/medgemma-4b-it"
MM_LORA    = "./medgemma_multimodal"
DMID_LORA  = "./medgemma_dmid"
IMGS_DIR   = "./dmid/TIFF Images/TIFF Images/"
REPS_DIR   = "./dmid/Reports/Reports/"

# Выбери нужные изображения из тест сета (последние 52)
TEST_IMAGES = ["Img462", "Img461", "Img463", "Img470", "Img475", "Img480", "Img294", "Img295"]

def find_image(img_id):
    img_num = img_id.replace('Img','').replace('IMG','')
    for f in os.listdir(IMGS_DIR):
        if img_num in f:
            return os.path.join(IMGS_DIR, f)
    return None

def gen(model, processor, img_path):
    image = Image.open(img_path).convert("RGB").resize((448,448))
    prompt = "<start_of_turn>user\n<start_of_image>\nGenerate a structured mammography report with breast composition, findings, BI-RADS category and recommendation.<end_of_turn>\n<start_of_turn>model\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()

def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16)

    # Загружаем наш two-stage model
    our_model = PeftModel.from_pretrained(base, DMID_LORA)
    our_model.eval()

    results = []
    for img_id in TEST_IMAGES:
        img_path = find_image(img_id)
        rep_path = os.path.join(REPS_DIR, f"{img_id}.txt")
        
        if not img_path or not os.path.exists(rep_path):
            print(f"Не найдено: {img_id}")
            continue

        with open(rep_path, encoding='utf-8', errors='ignore') as f:
            gt = f.read().strip()

        print(f"\n{'='*60}")
        print(f"Image: {img_id}")
        print(f"GT:   {gt[:200]}")
        
        our = gen(our_model, processor, img_path)
        print(f"Ours: {our[:200]}")

        results.append({
            "img_id": img_id,
            "img_path": img_path,
            "gt": gt,
            "ours": our,
        })

    with open("./qualitative_examples.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nСохранено в qualitative_examples.json")

if __name__ == "__main__":
    main()
