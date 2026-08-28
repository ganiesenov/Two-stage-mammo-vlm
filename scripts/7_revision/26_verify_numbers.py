"""
Сверка чисел рукописи с сохранёнными результатами (CPU).

Проверяет, что Table 4, 5, 6 и 9 в manuscript/main.tex совпадают с тем, что
лежит в results/revision/*.json на чистом подмножестве теста (n=49). Запускать
после любой правки таблиц: расхождение здесь означает, что в рукописи осталось
число из другого прогона.

Строки таблиц вытаскиваются регулярками из \\begin{tabular} нужного окружения;
макросы разметки (\\rev, \\revf, \\textbf) снимаются перед разбором.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

TEX = f"{ROOT}/manuscript/main.tex"
RES = f"{ROOT}/results/revision"
# Числа в json сохранены с 4 знаками, в таблицах — с 3-4. На границе округления
# (0.6495 → 0.649 или 0.650) допустимы обе записи, поэтому допуск — ровно
# половина последнего разряда таблицы плюс запас на представление float.
TOL = 0.0005 + 1e-9

fails, checks = [], 0


def strip(t):
    for m in ("revf", "revmf", "rev", "revm", "textbf"):
        t = re.sub(r"\\" + m + r"(?![A-Za-z])\s*\{", "{", t)
    return t.replace("{", "").replace("}", "").replace("$", "").replace("\\", "")


def check(label, got, want):
    global checks
    checks += 1
    if want is None or got is None or abs(got - want) > TOL:
        fails.append(f"{label}: в рукописи {got}, в данных {want}")


def table_rows(tex, label):
    """Строки tabular того окружения, где стоит \\label{label}."""
    i = tex.index("\\label{" + label + "}")
    j = tex.index("\\begin{tabular}", i)
    k = tex.index("\\end{tabular}", j)
    body = tex[j:k]
    return [strip(l).strip() for l in body.split("\\\\") if "&" in l]


def nums(row):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+", row)]


def main():
    tex = open(TEX, encoding="utf-8").read()
    clean = json.load(open(f"{RES}/clean_primary.json"))["clean"]
    bert = json.load(open(f"{RES}/bertscore_test_clean.json"))["scores"]
    boot = json.load(open(f"{RES}/bootstrap_ci_test_clean.json"))["comparisons"]
    clin = json.load(open(f"{RES}/clinical_metrics_test_clean.json"))

    # ── Table 4 ──
    key4 = {"MedGemma-4B": "zero_shot", "DMID-only, r=16": "dmid_only_r16",
            "Two-stage, r=16": "two_stage_r16", "DMID-only, r=64": "dmid_only_r64",
            "Two-stage, r=64": "two_stage_r64"}
    cols = ["bleu1", "bleu4", "rouge1", "rougeL", "meteor", "cider"]
    for row in table_rows(tex, "tab:dmid"):
        name = row.split("&")[0].strip()
        if name not in key4:
            continue
        m, v = key4[name], nums(row)
        if len(v) < 7:
            fails.append(f"Table 4 / {name}: разобрано {len(v)} чисел вместо 7")
            continue
        for c, got in zip(cols, v[:6]):
            check(f"Table 4 / {name} / {c}", got, clean[m][c])
        check(f"Table 4 / {name} / BERTScore", v[6], bert[m]["f1"])

    # ── Table 5 ──
    pair = {"r16": "two_stage_r16:dmid_only_r16", "r64": "two_stage_r64:dmid_only_r64"}
    met5 = {"BLEU-1": "bleu1", "BLEU-4": "bleu4", "ROUGE-1": "rouge1",
            "ROUGE-2": "rouge2", "ROUGE-L": "rougeL", "METEOR": "meteor"}
    for row in table_rows(tex, "tab:equalcap"):
        name = row.split("&")[0].strip()
        if name not in met5:
            continue
        v, k = nums(row), met5[name]
        # порядок: Δ, CI_lo, CI_hi, p  (дважды: r=16 и r=64)
        for off, rk in ((0, "r16"), (4, "r64")):
            d = boot[pair[rk]][k]
            check(f"Table 5 / {name} / {rk} Δ", v[off], d["delta"])
            check(f"Table 5 / {name} / {rk} CI_lo", v[off + 1], d["ci_low"])
            check(f"Table 5 / {name} / {rk} CI_hi", v[off + 2], d["ci_high"])

    # ── Table 6 ──
    key6 = {"MedGemma-4B (ZS)": "zero_shot", "DMID-only, r=16": "dmid_only_r16",
            "Two-stage, r=16": "two_stage_r16", "DMID-only, r=64": "dmid_only_r64",
            "Two-stage, r=64": "two_stage_r64"}
    for row in table_rows(tex, "tab:clinical"):
        name = row.split("&")[0].strip().replace("†", "").strip()
        if name not in key6:
            continue
        c = clin[key6[name]]
        v = nums(row)
        # exact, κ, [downgr не число], sens, omis, halluc, dens_acc, dens_κ
        check(f"Table 6 / {name} / exact", v[0], c["birads"]["exact"])
        check(f"Table 6 / {name} / kappa", v[1], c["birads"]["cohens_kappa"])
        check(f"Table 6 / {name} / sens", v[-5], c["findings"]["sensitivity"])
        check(f"Table 6 / {name} / omis", v[-4], c["findings"]["omission_rate"])
        check(f"Table 6 / {name} / dens_acc", v[-2], c["density"]["accuracy"])
        check(f"Table 6 / {name} / dens_kappa", v[-1], c["density"]["cohens_kappa"])

    # ── Table 9 ──
    order = ["r16_a32", "r32_a64", "r64_a64"]
    for row in table_rows(tex, "tab:tost"):
        head = row.split("&")[0].strip()
        if head not in ("369", "800", "1444"):
            continue
        eq = json.load(open(f"{RES}/equivalence_v2d{head}_clean49.json"))["by_config"]
        v = nums(row)
        for i, cfg in enumerate(order):
            e = eq[cfg]
            check(f"Table 9 / {head} / {cfg} Δ", v[3 * i], e["delta_mean"])
            check(f"Table 9 / {head} / {cfg} CI_lo", v[3 * i + 1], e["ci_low"])
            check(f"Table 9 / {head} / {cfg} CI_hi", v[3 * i + 2], e["ci_high"])

    print(f"  проверено значений: {checks}")
    if fails:
        print(f"  РАСХОЖДЕНИЙ: {len(fails)}")
        for f in fails:
            print("    " + f)
        sys.exit(1)
    print("  расхождений нет")


if __name__ == "__main__":
    main()
