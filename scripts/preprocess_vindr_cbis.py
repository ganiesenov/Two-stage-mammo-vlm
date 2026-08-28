import os, cv2, numpy as np
from PIL import Image
from pathlib import Path

def preprocess_mammogram(img_path, target_size=448):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(str(img_path)).convert('L'))
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 10
        x, y = max(0, x-pad), max(0, y-pad)
        w = min(img.shape[1]-x, w+2*pad)
        h = min(img.shape[0]-y, h+2*pad)
        img = img[y:y+h, x:x+w]
    mid = img.shape[1] // 2
    if img[:, :mid].sum() > img[:, mid:].sum():
        img = cv2.flip(img, 1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

BASE = "."

# 1. VinDr-Mammo
print("="*50)
print("Preprocessing VinDr-Mammo...")
print("="*50)
vindr_out = f"{BASE}/vindr/preprocessed/"
os.makedirs(vindr_out, exist_ok=True)
count = 0
for root, dirs, files in os.walk(f"{BASE}/vindr"):
    if 'preprocessed' in root:
        continue
    for f in files:
        if f.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = preprocess_mammogram(os.path.join(root, f))
                cv2.imwrite(os.path.join(vindr_out, Path(f).stem + '.png'), img)
                count += 1
                if count % 500 == 0:
                    print(f"  {count} done...")
            except:
                pass
print(f"VinDr done: {count} images")

# 2. CBIS-DDSM
print("\n" + "="*50)
print("Preprocessing CBIS-DDSM...")
print("="*50)
cbis_out = f"{BASE}/cbis-ddsm/preprocessed/"
os.makedirs(cbis_out, exist_ok=True)
count = 0
for root, dirs, files in os.walk(f"{BASE}/cbis-ddsm"):
    if 'preprocessed' in root:
        continue
    for f in files:
        if f.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = preprocess_mammogram(os.path.join(root, f))
                cv2.imwrite(os.path.join(cbis_out, Path(f).stem + '.png'), img)
                count += 1
                if count % 500 == 0:
                    print(f"  {count} done...")
            except:
                pass
print(f"CBIS done: {count} images")
print("\nAll preprocessing complete!")
