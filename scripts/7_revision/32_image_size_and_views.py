"""
Нативный размер снимков DMID и восстановимость спаривания проекций (R3.3, R3.4; CPU).

R3.3: кратность уменьшения числа пикселей при ресайзе в 448×448 считается по заголовкам
TIFF всех 510 изображений сплита, а не по типичному размеру маммограммы.

R3.4: DMID не публикует соответствие изображение→пациент. Единственный намёк на
спаривание — порядок идентификаторов: 510 изображений дают 127 последовательных блоков
по четыре (IMG001–004, IMG005–008, ...). Блок считается полным, если в нём ровно по
одному виду CCLT, CCRT, MLOLT, MLORT по Metadata.xlsx. Что такой блок — одно
исследование, нигде не подтверждено. Для сравнения: группировка по идентичному
тексту отчёта не даёт ни одного полного четырёхвидового блока.
"""
import os, sys, re, glob, json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

IMG_DIR = f"{ROOT}/dmid/TIFF Images/TIFF Images"
OUT = f"{ROOT}/results/revision/image_size_and_views.json"
VIEWS = ["CCLT", "CCRT", "MLOLT", "MLORT"]


def main():
    man = json.load(open(f"{ROOT}/results/revision/split_manifest.json"))
    ids = sorted(i for v in man["splits"].values() for i in v)

    shapes = {}
    for i in ids:
        with tifffile.TiffFile(f"{IMG_DIR}/{i.upper()}.tif") as t:
            shapes[i] = t.pages[0].shape[:2]
    fold = np.array([h * w / 448 ** 2 for h, w in shapes.values()])
    main_group = [i for i, (h, w) in shapes.items() if (h, w) >= (4784, 3512)]
    hs = [shapes[i] for i in main_group]
    size = {"n_images": len(ids),
            "n_main_group": len(main_group),
            "main_group_min": list(min(hs)), "main_group_max": list(max(hs)),
            "fold_main_group": [round(float(min(h * w for h, w in hs) / 448 ** 2), 1),
                                round(float(max(h * w for h, w in hs) / 448 ** 2), 1)],
            "fold_all": [round(float(fold.min()), 1), round(float(fold.max()), 1)],
            "fold_median": round(float(np.median(fold)), 1)}

    x = pd.read_excel(f"{ROOT}/dmid/Metadata.xlsx", header=None)
    x = x[x[0].astype(str).str.strip().str.upper().str.match(r"^IMG\d+$")]
    key = x[0].astype(str).str.strip().str.upper()
    view = x.groupby(key)[1].first().astype(str).str.strip().to_dict()
    normal = x.groupby(key)[3].apply(lambda s: all(str(v).strip().upper() == "NORM" for v in s)).to_dict()

    up = [i.upper() for i in ids]
    blocks = [up[k:k + 4] for k in range(0, len(up) - 3, 4)]
    complete = [b for b in blocks if sorted(view[i] for i in b) == VIEWS]
    inb = {i for b in complete for i in b}
    rest = [i for i in up if i not in inb]

    groups = defaultdict(list)
    for i in ids:
        txt = open(f"{ROOT}/dmid/Reports/Reports/{i}.txt", errors="ignore").read()
        groups[re.sub(r"\s+", " ", txt.lower()).strip()].append(i.upper())
    text_complete = sum(sorted(view[i] for i in g) == VIEWS for g in groups.values())

    views = {"n_blocks": len(blocks), "n_complete_blocks": len(complete),
             "normal_in_complete": [sum(normal[i] for i in inb), len(inb)],
             "normal_elsewhere": [sum(normal[i] for i in rest), len(rest)],
             "text_groups": len(groups),
             "text_group_sizes": dict(sorted(Counter(len(g) for g in groups.values()).items())),
             "text_groups_complete_4view": text_complete}
    out = {"image_size": size, "view_pairing": views}
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
