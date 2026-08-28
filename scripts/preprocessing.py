"""
Mammography Image Preprocessing Pipeline
Like AMRG: Otsu → ROI Crop → Laterality → CLAHE → Resize
"""
import os, cv2, numpy as np
from PIL import Image
from pathlib import Path

def preprocess_mammogram(img_path, target_size=448):
    """Full preprocessing pipeline"""
    # 1. Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try with PIL for TIFF
        pil_img = Image.open(str(img_path)).convert('L')
        img = np.array(pil_img)
    
    original_shape = img.shape
    
    # 2. Otsu thresholding → binary mask
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Find largest contour (breast ROI)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Add small padding
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        img_cropped = img[y:y+h, x:x+w]
    else:
        img_cropped = img
    
    # 4. Laterality matching — ensure breast faces right
    # Check which side has more intensity (breast tissue)
    mid = img_cropped.shape[1] // 2
    left_sum = img_cropped[:, :mid].sum()
    right_sum = img_cropped[:, mid:].sum()
    if left_sum > right_sum:
        img_cropped = cv2.flip(img_cropped, 1)  # Flip horizontally
    
    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_cropped)
    
    # 6. Resize to target
    img_resized = cv2.resize(img_enhanced, (target_size, target_size), 
                              interpolation=cv2.INTER_LANCZOS4)
    
    # 7. Convert to 3-channel (for VLM input)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    return img_rgb

def preprocess_dmid(input_dir, output_dir, target_size=448):
    """Preprocess all DMID images"""
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for f in sorted(os.listdir(input_dir)):
        if f.endswith(('.tif', '.tiff', '.TIF', '.TIFF', '.png', '.jpg')):
            try:
                img = preprocess_mammogram(os.path.join(input_dir, f), target_size)
                out_name = Path(f).stem + '.png'
                cv2.imwrite(os.path.join(output_dir, out_name), img)
                count += 1
            except Exception as e:
                print(f"  Error {f}: {e}")
    
    print(f"Preprocessed {count} images → {output_dir}")

if __name__ == "__main__":
    DMID_IMGS = "./dmid/TIFF Images/TIFF Images/"
    OUT_DIR = "./dmid/preprocessed/"
    
    preprocess_dmid(DMID_IMGS, OUT_DIR, target_size=448)
    
    # Show before/after for first image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    files = sorted(os.listdir(DMID_IMGS))
    orig = Image.open(os.path.join(DMID_IMGS, files[0])).convert('L')
    proc = cv2.imread(os.path.join(OUT_DIR, Path(files[0]).stem + '.png'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(np.array(orig), cmap='gray')
    ax1.set_title(f'Original ({orig.size[0]}x{orig.size[1]})')
    ax1.axis('off')
    ax2.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    ax2.set_title('Preprocessed (448x448)')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig('./paper/figures/fig_preprocessing.png', 
                dpi=300, bbox_inches='tight')
    print("Preprocessing figure saved!")
