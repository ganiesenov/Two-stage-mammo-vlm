import json

# BI-RADS knowledge base из официального ACR Atlas 5th Edition
birads_knowledge = [
    {"text": "BI-RADS Category 0: Incomplete assessment. Additional imaging evaluation and/or prior mammograms are needed. This assessment is used when a finding requires additional imaging workup before a final assessment can be made.", "source": "ACR_BIRADS"},
    {"text": "BI-RADS Category 1: Negative. There is nothing to comment on. The breasts are symmetric and no masses, architectural distortions, or suspicious calcifications are present. Routine screening recommended.", "source": "ACR_BIRADS"},
    {"text": "BI-RADS Category 2: Benign. This is a definitively benign finding. Calcified fibroadenomas, multiple secretory calcifications, fat-containing lesions such as oil cysts, lipomas, galactoceles, and mixed density hamartomas all have characteristically benign appearances. Routine screening recommended.", "source": "ACR_BIRADS"},
    {"text": "BI-RADS Category 3: Probably Benign. A finding placed in this category should have a less than 2% risk of malignancy. Short-interval follow-up (6 months) is suggested to establish stability. This includes non-calcified circumscribed solid mass, focal asymmetry, and solitary group of punctate calcifications.", "source": "ACR_BIRADS"},
    {"text": "BI-RADS Category 4: Suspicious. These findings do not have the classic appearance of malignancy but are sufficiently suspicious to justify a recommendation for biopsy. Category 4A: Low suspicion for malignancy (>2% to <=10%). Category 4B: Moderate suspicion for malignancy (>10% to <=50%). Category 4C: High suspicion for malignancy (>50% to <95%).", "source": "ACR_BIRADS"},
    {"text": "BI-RADS Category 5: Highly Suggestive of Malignancy. These findings have a high probability of being malignant (>=95%). Biopsy is strongly recommended. Findings include spiculated high-density mass, segmental or linear distribution of fine-linear calcifications.", "source": "ACR_BIRADS"},
    {"text": "BI-RADS Category 6: Known Biopsy-Proven Malignancy. This category is reserved for examinations performed after biopsy proof of malignancy, prior to definitive therapy.", "source": "ACR_BIRADS"},
    {"text": "Breast Density Category A: The breasts are almost entirely fatty. Approximately 10% of women have this breast composition. Mammography is highly sensitive in fatty breasts.", "source": "ACR_BIRADS"},
    {"text": "Breast Density Category B: There are scattered areas of fibroglandular density. Approximately 40% of women have this breast composition. Sensitivity of mammography is slightly reduced.", "source": "ACR_BIRADS"},
    {"text": "Breast Density Category C: The breasts are heterogeneously dense, which may obscure small masses. Approximately 40% of women have this breast composition. Supplemental screening may be considered.", "source": "ACR_BIRADS"},
    {"text": "Breast Density Category D: The breasts are extremely dense, which lowers the sensitivity of mammography. Approximately 10% of women have this breast composition. Supplemental screening with ultrasound or MRI is recommended.", "source": "ACR_BIRADS"},
    {"text": "Mass shape descriptors: Round (spherical), Oval (elliptical), Irregular (neither round nor oval). Margin descriptors: Circumscribed (well-defined), Obscured (hidden by adjacent tissue), Microlobulated, Indistinct, Spiculated. Density: High, Equal, Low, Fat-containing.", "source": "ACR_BIRADS"},
    {"text": "Calcification morphology: Typically benign includes skin, vascular, coarse, large rod-like, round, rim, dystrophic, milk of calcium, suture. Suspicious morphology includes amorphous, coarse heterogeneous, fine pleomorphic, fine linear or fine-linear branching calcifications.", "source": "ACR_BIRADS"},
    {"text": "Calcification distribution: Diffuse, Regional, Grouped (cluster), Linear, Segmental. Linear and segmental distributions are more suspicious for malignancy as they suggest calcifications within a duct system.", "source": "ACR_BIRADS"},
    {"text": "Architectural distortion: The normal architecture is distorted with no definite mass visible. This includes spiculations radiating from a point and focal retraction or distortion of the edge of the parenchyma. This finding is suspicious and requires biopsy unless a specific benign diagnosis can be established.", "source": "ACR_BIRADS"},
    {"text": "Asymmetry types: Asymmetry (visible on one projection only), Global asymmetry (occupying at least one quadrant), Focal asymmetry (occupying less than one quadrant, lacking convex borders), Developing asymmetry (new or increasing focal asymmetry, considered suspicious).", "source": "ACR_BIRADS"},
    {"text": "Associated features: Skin retraction, Nipple retraction, Skin thickening, Trabecular thickening, Axillary adenopathy, Architectural distortion, Calcifications associated with mass or asymmetry.", "source": "ACR_BIRADS"},
    {"text": "Management recommendations by BI-RADS: Category 0 - Additional imaging needed. Category 1,2 - Routine annual screening. Category 3 - Short-interval follow-up at 6 months. Category 4,5 - Tissue sampling (biopsy) recommended. Category 6 - Surgical excision when clinically appropriate.", "source": "ACR_BIRADS"},
    {"text": "Fibroadenoma mammography appearance: Oval or round circumscribed mass, may have coarse calcifications (popcorn type). Classified as BI-RADS 2 (benign) when calcified, BI-RADS 3 when non-calcified circumscribed solid mass in young patient.", "source": "ACR_BIRADS"},
    {"text": "Invasive ductal carcinoma mammography appearance: Irregular spiculated high-density mass, often with associated pleomorphic calcifications. Skin thickening and nipple retraction may be present. Classified as BI-RADS 5.", "source": "ACR_BIRADS"},
    {"text": "Ductal carcinoma in situ (DCIS) mammography appearance: Fine linear branching or fine pleomorphic calcifications in linear or segmental distribution. May present without an associated mass. Classified as BI-RADS 4C or 5.", "source": "ACR_BIRADS"},
]

# Загружаем скрейпнутые чанки
with open("./birads_chunks.json") as f:
    scraped = json.load(f)

# Объединяем
all_chunks = birads_knowledge + scraped
print(f"Итого чанков: {len(all_chunks)}")

with open("./birads_chunks.json", "w") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
print("Сохранено!")
