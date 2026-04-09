import os, torch, logging
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN")
MODEL_ID = "google/medgemma-4b-it"
LORA_PATH = "/mnt/c/Users/juman/hard_ml/rag_mammo/new_article/lora_ablation/r64_a64/checkpoint-204"
PROMPT = "Generate a structured mammography radiology report with breast composition (ACR density), findings, BI-RADS category, and recommendation."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    logger.info("Loading MedGemma + LoRA...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    logger.info("Model loaded!")
    return model, processor

model, processor = load_model()

def generate_report(image: Image.Image) -> str:
    image = image.convert("RGB").resize((448, 448))
    prompt = f"<start_of_turn>user\n<start_of_image>\n{PROMPT}<end_of_turn>\n<start_of_turn>model\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Mammography Report Generator\n\n"
        "Send me a mammography image and I'll generate a structured radiology report.\n\n"
        "Disclaimer: Research tool only. Must be reviewed by a radiologist.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing mammogram...")
    try:
        photo = update.message.photo[-1] if update.message.photo else None
        document = update.message.document if not photo else None
        if photo:
            file = await photo.get_file()
        elif document:
            file = await document.get_file()
        else:
            await update.message.reply_text("Please send an image.")
            return
        img_bytes = await file.download_as_bytearray()
        image = Image.open(BytesIO(img_bytes))
        report = generate_report(image)
        await update.message.reply_text(f"Generated Report:\n\n{report}\n\n(AI-generated, review by radiologist required)")
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"Error: {str(e)[:200]}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send a mammography image to get a report.")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot started!")
    app.run_polling()

if __name__ == "__main__":
    main()
