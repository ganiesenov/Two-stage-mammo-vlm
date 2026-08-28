import requests
from bs4 import BeautifulSoup
import json
import time

URLS = [
    "https://radiologyassistant.nl/breast/bi-rads/bi-rads-for-mammography-and-ultrasound-2013",
    "https://radiologyassistant.nl/breast/bi-rads/bi-rads-mri",
    "https://radiologyassistant.nl/breast/mammography/breast-calcifications-differential-diagnosis",
    "https://radiologyassistant.nl/breast/mammography/masses",
]

chunks = []

for url in URLS:
    print(f"Скрейплю: {url}")
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        # Убираем навигацию и мусор
        for tag in soup(["script","style","nav","header","footer"]):
            tag.decompose()
        # Берём параграфы
        paragraphs = soup.find_all(["p","h1","h2","h3","li"])
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        # Разбиваем на чанки по ~300 слов
        words = text.split()
        for i in range(0, len(words), 250):
            chunk = " ".join(words[i:i+300])
            if len(chunk) > 100:
                chunks.append({"text": chunk, "source": url})
        print(f"  → {len(words)} слов")
        time.sleep(2)
    except Exception as e:
        print(f"  ОШИБКА: {e}")

print(f"\nВсего чанков: {len(chunks)}")
with open("./birads_chunks.json", "w") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print("Сохранено в birads_chunks.json")
