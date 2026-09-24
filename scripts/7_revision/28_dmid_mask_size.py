"""
Размер пропущенных и заявленных находок по ROI-маскам DMID (R3.3, CPU).

R3.3: 448-px ресайз снимка 3500–4500 px уменьшает число пикселей в 61–100 раз;
гипотеза — мелкие находки после этого просто не видны энкодеру. Проверяем на лучшей
модели (DMID-only r=64, Table 4): у пропущенных находок площадь маски должна быть
систематически меньше, чем у заявленных.

Покрытие масок: в релизе DMID «ROI Masks» есть для всех изображений с аномалией
в Metadata.xlsx (массы всех типов, кальцинаты, дисторсии, асимметрии), нормам маска
не положена. Маска бинарная, разрешение как у исходного TIFF.

Пропуск/заявление определяются тем же правилом, что и в Table 6
(04_clinical_metrics.asserts_finding, окно отрицания 45 символов).
"""
import os, sys, json, importlib.util
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, load_preds

spec = importlib.util.spec_from_file_location(
    "clin", os.path.join(os.path.dirname(__file__), "04_clinical_metrics.py"))
clin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clin)

MASK_DIR = f"{ROOT}/dmid/ROI Masks/ROI Masks"
IMG_SIZE = 448
OUT = f"{ROOT}/results/revision/dmid_mask_size.json"
MODEL = "dmid_only_r64"


def mask_stats(img_id):
    path = f"{MASK_DIR}/{img_id.upper()}.tif"
    if not os.path.exists(path):
        return None
    m = tifffile.imread(path)
    m = (m[..., 0] if m.ndim == 3 else m) > 0
    h, w = m.shape
    area = int(m.sum())
    return {"h": h, "w": w, "area_px": area,
            "area_pct": 100 * area / (h * w),
            # площадь после ресайза всего снимка в 448×448
            "area_px_448": area * IMG_SIZE * IMG_SIZE / (h * w)}


def mann_whitney_u(a, b):
    """Односторонний U-тест (H1: a < b), нормальное приближение без поправки на связи."""
    from math import erf, sqrt
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks.setdefault(k, (i + j + 1) / 2)
        i = j
    ra = sum(ranks[k] for k, (_, g) in enumerate(allv) if g == 0)
    n1, n2 = len(a), len(b)
    u = ra - n1 * (n1 + 1) / 2
    mu, sd = n1 * n2 / 2, sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd
    return u, 0.5 * (1 + erf(z / sqrt(2)))


def main():
    man = json.load(open(f"{ROOT}/results/revision/split_manifest.json"))
    leaked = set(man["verbatim_duplicate_leakage"]["test_ids"])
    preds = [p for p in load_preds(MODEL, "test") if p["id"] not in leaked]

    rows = []
    for p in preds:
        ref_a, _ = clin.asserts_finding(p["ref"])
        if not ref_a:
            continue
        hyp_a, _ = clin.asserts_finding(p["hyp"])
        st = mask_stats(p["id"])
        rows.append({"id": p["id"], "status": "reported" if hyp_a else "missed",
                     **(st or {"area_px": None})})

    def summ(status):
        v = [r for r in rows if r["status"] == status and r["area_px"] is not None]
        a = np.array([r["area_px"] for r in v], float)
        pct = np.array([r["area_pct"] for r in v], float)
        a448 = np.array([r["area_px_448"] for r in v], float)
        return {"n": len(v),
                "n_without_mask": sum(r["status"] == status and r["area_px"] is None for r in rows),
                "median_area_px": float(np.median(a)), "iqr_area_px": [float(x) for x in np.percentile(a, [25, 75])],
                "median_area_pct": float(np.median(pct)), "iqr_area_pct": [float(x) for x in np.percentile(pct, [25, 75])],
                "median_area_px_448": float(np.median(a448)),
                "iqr_area_px_448": [float(x) for x in np.percentile(a448, [25, 75])]}, list(pct)

    s_miss, pm = summ("missed")
    s_rep, pr = summ("reported")
    u, p = mann_whitney_u(pm, pr)
    out = {"model": MODEL, "n_cases": len(preds), "n_ref_positive": len(rows),
           "missed": s_miss, "reported": s_rep,
           "mann_whitney_one_sided": {"U": u, "p": p, "h1": "missed < reported"},
           "cases": rows}
    json.dump(out, open(OUT, "w"), indent=2)
    for k in ("missed", "reported"):
        s = out[k]
        print(f"{k:9s} n={s['n']:2d} (без маски {s['n_without_mask']})  "
              f"медиана {s['median_area_px']:.0f} px [{s['iqr_area_px'][0]:.0f}–{s['iqr_area_px'][1]:.0f}]  "
              f"{s['median_area_pct']:.3f}% [{s['iqr_area_pct'][0]:.3f}–{s['iqr_area_pct'][1]:.3f}]  "
              f"@448: {s['median_area_px_448']:.0f} px [{s['iqr_area_px_448'][0]:.0f}–{s['iqr_area_px_448'][1]:.0f}]")
    print(f"U={u:.1f}  p(односторонний)={p:.3f}")


if __name__ == "__main__":
    main()
