"""
Сверка чисел рукописи с сохранёнными результатами (CPU).

Проверяет, что Table 4, 5, 6 и 9 в manuscript/main.tex, Supplementary Tables S3–S5
и числа раунда 3 в прозе (medspaCy, leave-one-out, размер снимков, блоки проекций,
пропуски по типу находки) совпадают с тем, что
лежит в results/revision/*.json на чистом подмножестве теста (n=49). Запускать
после любой правки таблиц: расхождение здесь означает, что в рукописи осталось
число из другого прогона.

Строки таблиц вытаскиваются регулярками из \\begin{tabular} нужного окружения;
макросы разметки (\\rev, \\revf, \\textbf) снимаются перед разбором.
"""
import json, math, os, re, sys

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


SUPP = f"{ROOT}/manuscript/supplementary.tex"
NAMES = {"MedGemma-4B (ZS)": "zero_shot", "DMID-only, r=16": "dmid_only_r16",
         "Two-stage, r=16": "two_stage_r16", "DMID-only, r=64": "dmid_only_r64",
         "Two-stage, r=64": "two_stage_r64"}


def supp_rows(sup, label, k=0):
    """Строки k-го tabular после \\label{label} в supplementary.tex."""
    i = sup.index("\\label{" + label + "}")
    for _ in range(k + 1):
        j = sup.index("\\begin{tabular}", i)
        i = sup.index("\\end{tabular}", j)
    return [strip(l).strip() for l in sup[j:i].split("\\\\") if "&" in l]


def verify_supplementary(clin):
    """S3–S5: таблицы генерируются 33_supp_tables.py, здесь ловится ручная правка
    или устаревший прогон генератора."""
    sup = open(SUPP, encoding="utf-8").read()
    # S3 — плотность
    for row in supp_rows(sup, "stab:density"):
        name = row.split("&")[0].strip()
        if name not in NAMES:
            continue
        d, v = clin[NAMES[name]]["density"], nums(row.split("&", 2)[2])
        check(f"S3 / {name} / acc", v[0], d["accuracy"])
        check(f"S3 / {name} / kappa", v[1], d["cohens_kappa"])
    # S4 — окно, согласие с medspaCy, категории
    w = json.load(open(f"{RES}/negation_window_sweep.json"))
    ms = json.load(open(f"{RES}/negation_medspacy.json"))
    for row in supp_rows(sup, "stab:negation", 0):
        name = row.split("&")[0].strip()
        if name not in NAMES:
            continue
        v = nums(row)
        for i, win in enumerate(["30", "45", "60", "90"]):
            r = w["clean49"][NAMES[name]][win]
            check(f"S4a / {name} / {win} sens", v[2 * i], r["sensitivity"])
            check(f"S4a / {name} / {win} halluc", v[2 * i + 1], r["hallucination_rate"])
    rows_b = supp_rows(sup, "stab:negation", 1)
    srcs = [ms["clean49"]["mention_level"], ms["clean49"]["report_level"],
            ms["mention_level"], ms["report_level"]]
    for row, src in zip([r for r in rows_b if "%" in r], srcs):
        v = [float(x) for x in re.findall(r"\d+\.?\d*", row.split("&", 2)[2])]
        check(f"S4b / {row[:30]} / n", v[0], src["n"])
        check(f"S4b / {row[:30]} / agreement%", v[1] / 100, src["agreement"])
        check(f"S4b / {row[:30]} / kappa", round(v[2], 2), round(src["kappa"], 2))
    for row in supp_rows(sup, "stab:negation", 2):
        name = row.split("&")[0].strip()
        if name not in NAMES:
            continue
        c = w["category_level"]["clean49"][NAMES[name]]
        v = [int(x) for x in re.findall(r"\d+", row.split("&", 1)[1])]
        want = [c["extra_categories"], c["hyp_categories"], c["cases_with_extra_category"],
                c["cases_with_finding_on_normal_reference"], c["missing_categories"], c["ref_categories"]]
        for lab, g, t in zip(["extra", "hyp", "cases", "normal", "missing", "ref"], v, want):
            check(f"S4c / {name} / {lab}", g, t)
    # S5 — leave-one-out
    loo = json.load(open(f"{RES}/leave_one_out.json"))
    keys = {"BLEU-1": "bleu1", "BLEU-4": "bleu4", "ROUGE-1": "rouge1", "ROUGE-L": "rougeL",
            "METEOR": "meteor", "CIDEr": "cider", "BERTScore": "bertscore",
            "Sensitivity": "sensitivity", "BI-RADS exact": "birads_exact"}
    models = ["dmid_only_r16", "two_stage_r16", "dmid_only_r64", "two_stage_r64"]
    for row in supp_rows(sup, "stab:loo"):
        name = row.split("&")[0].strip()
        if name not in keys:
            continue
        v = nums(row)
        for i, m in enumerate(models):
            e = loo["models"][m][keys[name]]
            check(f"S5 / {name} / {m} value", v[2 * i], e["full"])
            check(f"S5 / {name} / {m} shift", v[2 * i + 1], e["max_abs_shift"])


def verify_prose(tex):
    """Числа раунда 3 в прозе: берутся из данных и ищутся в тексте дословно."""
    def need(label, text):
        global checks
        checks += 1
        if text not in tex:
            fails.append(f"проза / {label}: в рукописи нет «{text}»")

    ms = json.load(open(f"{RES}/negation_medspacy.json"))["clean49"]
    need("medspaCy mentions %", f"${100 * ms['mention_level']['agreement']:.1f}\\%$ of the ${ms['mention_level']['n']}$")
    need("medspaCy mentions κ", f"$\\kappa = {ms['mention_level']['kappa']:.2f}$")
    need("medspaCy reports %", f"${100 * ms['report_level']['agreement']:.1f}\\%$ of the ${ms['report_level']['n']}$")
    need("medspaCy reports κ", f"$\\kappa = {ms['report_level']['kappa']:.2f}$")

    loo = json.load(open(f"{RES}/leave_one_out.json"))
    text_max = max(loo["models"][m][k]["max_abs_shift"] for m in loo["models"]
                   for k in ("bleu1", "bleu4", "rouge1", "rougeL", "meteor"))
    cider_max = max(v["cider"]["max_abs_shift"] for v in loo["models"].values())
    bert_max = max(v["bertscore"]["max_abs_shift"] for v in loo["models"].values())
    delta_max = max(v["max_abs_shift"] for v in loo["paired_rougeL_delta"].values())
    # «не больше X»: X — верхняя граница, поэтому округление вверх до 3 знаков
    up = lambda x: math.ceil(round(x * 1000, 6)) / 1000
    need("LOO text", f"other than CIDEr by more than ${up(text_max):.3f}$")
    need("LOO CIDEr", f"(CIDEr: ${up(cider_max):.3f}$)")
    need("LOO BERTScore", f"BERTScore by more than ${up(bert_max):.3f}$")
    need("LOO Δ", f"difference by more than ${up(delta_max):.3f}$")

    iv = json.load(open(f"{RES}/image_size_and_views.json"))
    sz, vw = iv["image_size"], iv["view_pairing"]
    need("размер снимков", f"$4784\\times3512$ to $6000\\times4754$ px ({sz['n_main_group']} of {sz['n_images']} images)")
    need("кратность", f"{sz['fold_main_group'][0]:.0f}- to {sz['fold_main_group'][1]:.0f}-fold")
    need("медиана кратности", f"median of {sz['fold_median']:.0f}-fold")
    need("блоки видов", f"{vw['n_complete_blocks']} of {vw['n_blocks']} consecutive four-image blocks")
    nc, na = vw["normal_in_complete"]; ec, ea = vw["normal_elsewhere"]
    need("норма в блоках", f"${100 * nc / na:.0f}\\%$ of the images in complete blocks are normal, against ${100 * ec / ea:.0f}\\%$")

    mk = json.load(open(f"{RES}/dmid_mask_size.json"))["by_calcification"]
    miss, rep = mk["missed"], mk["reported"]
    need("пропуски без кальцинатов",
         f"{miss['without_calcification']} of {miss['without_calcification'] + miss['with_calcification']} omitted finding cases are masses without calcifications")
    need("кальцинаты названы",
         f"{rep['with_calcification']} of {rep['with_calcification'] + miss['with_calcification']} calcification-bearing cases are reported")


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
        # exact, κ, [downgr не число], sens, omis, halluc  (плотность вынесена в S3)
        check(f"Table 6 / {name} / exact", v[0], c["birads"]["exact"])
        check(f"Table 6 / {name} / kappa", v[1], c["birads"]["cohens_kappa"])
        check(f"Table 6 / {name} / sens", v[-3], c["findings"]["sensitivity"])
        check(f"Table 6 / {name} / omis", v[-2], c["findings"]["omission_rate"])
        check(f"Table 6 / {name} / halluc", v[-1], c["findings"]["hallucination_rate"])

    verify_supplementary(clin)
    verify_prose(tex)

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
