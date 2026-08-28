"""
CLIP+GPT2 and MedCLIP+GPT2 baselines on DMID
Same setup as AMRG for fair comparison
"""
import os, torch, json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, 
                          CLIPModel, CLIPProcessor,
                          TrainingArguments, Trainer)
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

IMGS_DIR  = "./dmid/TIFF Images/TIFF Images/"
REPS_DIR  = "./dmid/Reports/Reports/"
OUT_DIR   = "./clip_gpt2_models"
RESULTS   = "./results/clip_gpt2_results.json"

os.makedirs(OUT_DIR, exist_ok=True)

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

# ═══════════════════════════════════════════
# CLIP + GPT2 Model
# ═══════════════════════════════════════════
class CLIPtoGPT2(nn.Module):
    def __init__(self, clip_model_id, gpt2_model_id="gpt2"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_id)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_id)
        
        clip_dim = self.clip.config.projection_dim
        gpt2_dim = self.gpt2.config.n_embd
        
        # Project CLIP features to GPT2 embedding space
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, gpt2_dim),
            nn.GELU(),
            nn.Linear(gpt2_dim, gpt2_dim),
            nn.LayerNorm(gpt2_dim)
        )
        
        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # Get CLIP image features
        with torch.no_grad():
            clip_out = self.clip.get_image_features(pixel_values=pixel_values)
        if not isinstance(clip_out, torch.Tensor):
            clip_out = clip_out.pooler_output if hasattr(clip_out, "pooler_output") else clip_out[0]
        
        # Project to GPT2 space: [batch, clip_dim] -> [batch, 1, gpt2_dim]
        img_embeds = self.projection(clip_out).unsqueeze(1)
        
        # Get GPT2 text embeddings
        text_embeds = self.gpt2.transformer.wte(input_ids)
        
        # Concatenate: [img_token, text_tokens]
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            img_mask = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        
        # Adjust labels (shift by 1 for image token)
        if labels is not None:
            img_label = torch.full((labels.shape[0], 1), -100, device=labels.device)
            labels = torch.cat([img_label, labels], dim=1)
        
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

class CLIPGPTDataset(Dataset):
    def __init__(self, pairs, clip_processor, gpt2_tokenizer, max_length=256):
        self.pairs = pairs
        self.clip_processor = clip_processor
        self.gpt2_tokenizer = gpt2_tokenizer
        self.max_length = max_length
    
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        item = self.pairs[idx]
        try:
            image = Image.open(item["img_path"]).convert("RGB").resize((224, 224))
        except:
            image = Image.new("RGB", (224, 224), 128)
        
        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        text = item["report"] + self.gpt2_tokenizer.eos_token
        encoding = self.gpt2_tokenizer(text, truncation=True, max_length=self.max_length,
                                        padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}

def train_clip_gpt2(clip_model_id, model_name, train_pairs, val_pairs):
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")
    
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    model = CLIPtoGPT2(clip_model_id, "gpt2")
    model = model.to("cuda")
    
    train_ds = CLIPGPTDataset(train_pairs, clip_processor, gpt2_tokenizer)
    val_ds = CLIPGPTDataset(val_pairs, clip_processor, gpt2_tokenizer)
    
    save_dir = f"{OUT_DIR}/{model_name}"
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=5e-5, weight_decay=0.01
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(10):
        # Train
        model.train()
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_train = total_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)
        
        print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
    
    # Load best
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt"))
    return model, clip_processor, gpt2_tokenizer

def generate_clip_gpt2(model, clip_processor, gpt2_tokenizer, img_path, max_length=200):
    model.eval()
    try:
        image = Image.open(img_path).convert("RGB").resize((224, 224))
    except:
        return ""
    
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
    
    # Get image embedding
    with torch.no_grad():
        clip_out = model.clip.get_image_features(pixel_values=pixel_values)
        if not isinstance(clip_out, torch.Tensor):
            clip_out = clip_out.pooler_output if hasattr(clip_out, "pooler_output") else clip_out[0]
    img_embeds = model.projection(clip_out).unsqueeze(1)
    
    # Start with image token
    generated = []
    past = None
    inputs_embeds = img_embeds
    
    for _ in range(max_length):
        with torch.no_grad():
            if past is None:
                outputs = model.gpt2(inputs_embeds=inputs_embeds, use_cache=True)
            else:
                outputs = model.gpt2(inputs_embeds=inputs_embeds, past_key_values=past, use_cache=True)
        
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == gpt2_tokenizer.eos_token_id:
            break
        
        generated.append(next_token.item())
        inputs_embeds = model.gpt2.transformer.wte(next_token.unsqueeze(0))
    
    return gpt2_tokenizer.decode(generated, skip_special_tokens=True).strip()

def compute_cider(refs, hyps):
    import math
    from collections import Counter
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    N = len(refs)
    df = Counter()
    for ref in refs:
        seen = set()
        tokens = ref.lower().split()
        for n in range(1, 5):
            for ng in ngrams(tokens, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    scores = []
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        ref_tf = Counter()
        hyp_tf = Counter()
        for n in range(1, 5):
            ref_tf.update(ngrams(ref_tokens, n))
            hyp_tf.update(ngrams(hyp_tokens, n))
        ref_tfidf, hyp_tfidf = {}, {}
        for ng, c in ref_tf.items():
            ref_tfidf[ng] = c * math.log(max(1., N) / max(1., df.get(ng, 0)))
        for ng, c in hyp_tf.items():
            hyp_tfidf[ng] = c * math.log(max(1., N) / max(1., df.get(ng, 0)))
        all_ng = set(ref_tfidf) | set(hyp_tfidf)
        dot = sum(ref_tfidf.get(ng, 0) * hyp_tfidf.get(ng, 0) for ng in all_ng)
        nr = math.sqrt(sum(v**2 for v in ref_tfidf.values())) + 1e-8
        nh = math.sqrt(sum(v**2 for v in hyp_tfidf.values())) + 1e-8
        scores.append(dot / (nr * nh))
    return sum(scores) / len(scores) if scores else 0.0

def compute_metrics(refs, hyps, label):
    sf = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s['rouge1'].fmeasure); r2.append(s['rouge2'].fmeasure); rl.append(s['rougeL'].fmeasure)
    meteor = []
    for ref, hyp in zip(refs, hyps):
        try: meteor.append(meteor_score([nltk.word_tokenize(ref.lower())], nltk.word_tokenize(hyp.lower())))
        except: meteor.append(0.0)
    cider = compute_cider(refs, hyps)
    _, _, F = bert_score(hyps, refs, lang="en", verbose=False)
    m = {"bleu1": round(bleu1,4), "bleu4": round(bleu4,4),
         "rouge1": round(sum(r1)/len(r1),4), "rouge2": round(sum(r2)/len(r2),4),
         "rougeL": round(sum(rl)/len(rl),4), "meteor": round(sum(meteor)/len(meteor),4),
         "cider": round(cider,4), "bertscore": round(F.mean().item(),4)}
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k, v in m.items(): print(f"  {k:12s}: {v}")
    return m

def main():
    all_files = sorted(os.listdir(REPS_DIR))
    train_pairs = load_pairs(all_files[:407])
    val_pairs = load_pairs(all_files[407:458])
    test_pairs = load_pairs(all_files[-52:])
    refs = [p["report"] for p in test_pairs]
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    results = {}
    
    # 1. CLIP + GPT2
    model1, proc1, tok1 = train_clip_gpt2(
        "openai/clip-vit-base-patch32", "clip_gpt2", train_pairs, val_pairs)
    hyps1 = []
    for i, p in enumerate(test_pairs):
        hyps1.append(generate_clip_gpt2(model1, proc1, tok1, p["img_path"]))
        print(f"\r  CLIP+GPT2 eval: {i+1}/{len(test_pairs)}", end="", flush=True)
    print()
    results["clip_gpt2"] = compute_metrics(refs, hyps1, "CLIP+GPT2 (fine-tuned)")
    del model1; torch.cuda.empty_cache()
    
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    
    # 2. MedCLIP + GPT2
    # MedCLIP uses same CLIP architecture but medical pretrained
    try:
        medclip_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        model2, proc2, tok2 = train_clip_gpt2(
            medclip_id, "medclip_gpt2", train_pairs, val_pairs)
        hyps2 = []
        for i, p in enumerate(test_pairs):
            hyps2.append(generate_clip_gpt2(model2, proc2, tok2, p["img_path"]))
            print(f"\r  MedCLIP+GPT2 eval: {i+1}/{len(test_pairs)}", end="", flush=True)
        print()
        results["medclip_gpt2"] = compute_metrics(refs, hyps2, "MedCLIP+GPT2 (fine-tuned)")
        del model2; torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nMedCLIP failed: {e}")
        print("Trying BiomedCLIP instead...")
        try:
            biomedclip_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            model2, proc2, tok2 = train_clip_gpt2(
                biomedclip_id, "biomedclip_gpt2", train_pairs, val_pairs)
            hyps2 = []
            for i, p in enumerate(test_pairs):
                hyps2.append(generate_clip_gpt2(model2, proc2, tok2, p["img_path"]))
                print(f"\r  BiomedCLIP+GPT2: {i+1}/{len(test_pairs)}", end="", flush=True)
            print()
            results["biomedclip_gpt2"] = compute_metrics(refs, hyps2, "BiomedCLIP+GPT2 (fine-tuned)")
            del model2
        except Exception as e2:
            print(f"\nBiomedCLIP also failed: {e2}")
    
    torch.cuda.empty_cache()
    
    # Save final
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nВсё сохранено в {RESULTS}")

if __name__ == "__main__":
    main()
