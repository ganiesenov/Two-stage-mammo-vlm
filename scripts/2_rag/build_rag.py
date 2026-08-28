import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "./birads_chunks.json"
INDEX_FILE  = "./birads_faiss.index"
META_FILE   = "./birads_meta.pkl"

print("Загружаю чанки...")
with open(CHUNKS_FILE) as f:
    chunks = json.load(f)
texts = [c["text"] for c in chunks]
print(f"Чанков: {len(texts)}")

print("Загружаю encoder (BioBERT)...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("Кодирую...")
embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings).astype("float32")

print("Строю FAISS индекс...")
dim = embeddings.shape[1]

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"Индекс сохранён: {index.ntotal} векторов")
print("Готово!")
