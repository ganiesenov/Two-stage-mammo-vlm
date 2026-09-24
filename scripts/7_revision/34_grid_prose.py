"""
Числа прозы, зависящие от сетки эквивалентности (round 3, CPU, после 06_equivalence.py).

Проза Results/Discussion/Limitations опирается на величины, которых нет в Table 9 в
явном виде: SD между сидами по рангу, размах между сидами, среднее и максимальное |Δ|,
знаки по колонкам, «худший» сид, средние DMID-only по рангу (Table 8 text), сужение
90% CI при переходе с трёх сидов на пять. Всё считается здесь, на обоих сплитах,
чтобы текст переписывался из данных, а не патчился.

Вход: results/revision/capacity_sweep.json, сохранённые генерации sweep_*,
equivalence_v2d*{,_clean49}.json (текущие) и *.3seed.json (снимок до раунда 3).
Выход: results/revision/grid_prose.json
"""
import os, sys, json, importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT

spec = importlib.util.spec_from_file_location(
    "eqv", os.path.join(os.path.dirname(__file__), "06_equivalence.py"))
eqv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eqv)

RES = f"{ROOT}/results/revision"
DOSES = ["369", "800", "1444"]
CONFIGS = [(16, 32), (32, 64), (64, 64)]
OUT = f"{RES}/grid_prose.json"


def cell_values(res, arm, r, a, tag, seeds):
    return {s: eqv.cell_metric(res, arm, r, a, s, tag) for s in seeds}


def split_summary(res, split):
    eqv.KEEP_IDS = None
    if split == "clean49":
        man = json.load(open(f"{RES}/split_manifest.json"))
        eqv.KEEP_IDS = set(man["clean_test_ids"])
    seeds = sorted({v["seed"] for v in res.values()})
    suffix = "_clean49" if split == "clean49" else ""

    out = {"seeds": seeds, "cells": {}, "dmid_only": {}, "sd_by_rank": {}, "spread_by_rank": {}}
    do = {}
    for r, a in CONFIGS:
        do[(r, a)] = {s: v for s, v in cell_values(res, "dmid_only", r, a, "", seeds).items() if v is not None}
        vals = list(do[(r, a)].values())
        out["dmid_only"][f"r{r}_a{a}"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                          "n": len(vals), "by_seed": do[(r, a)]}
    deltas, worst = [], {}
    for dose in DOSES:
        tag = f"_v2d{dose}"
        eq = json.load(open(f"{RES}/equivalence_v2d{dose}{suffix}.json"))["by_config"]
        old = json.load(open(f"{RES}/equivalence_v2d{dose}{suffix}.3seed.json"))["by_config"]
        for r, a in CONFIGS:
            c = f"r{r}_a{a}"
            ts = {s: v for s, v in cell_values(res, "two_stage", r, a, tag, seeds).items() if v is not None}
            e, o = eq[c], old[c]
            w_new, w_old = e["ci_high"] - e["ci_low"], o["ci_high"] - o["ci_low"]
            out["cells"][f"{dose}/{c}"] = {
                "delta": e["delta_mean"], "ci": [e["ci_low"], e["ci_high"]], "verdict": e["verdict"],
                "n_runs": e["n_runs"], "two_stage_sd": e["two_stage_sd"], "dmid_only_sd": e["dmid_only_sd"],
                "ci_width": w_new, "ci_width_3seed": w_old, "ci_narrowing_pct": 100 * (1 - w_new / w_old),
                "verdict_3seed": o["verdict"], "delta_3seed": o["delta_mean"],
                "worst_seed_two_stage": min(ts, key=ts.get),
                "worst_seed_dmid_only": min(do[(r, a)], key=do[(r, a)].get)}
            deltas.append(e["delta_mean"])
    for r, a in CONFIGS:
        c = f"r{r}_a{a}"
        sds, vals = [], list(do[(r, a)].values())
        for dose in DOSES:
            cell = out["cells"][f"{dose}/{c}"]
            sds += [cell["two_stage_sd"], cell["dmid_only_sd"]]
            vals += [v for v in cell_values(res, "two_stage", r, a, f"_v2d{dose}", seeds).values() if v is not None]
        out["sd_by_rank"][c] = [min(sds), max(sds)]
        out["spread_by_rank"][c] = max(vals) - min(vals)
        out[f"signs_{c}"] = [np.sign(out["cells"][f"{d}/{c}"]["delta"]).item() for d in DOSES]
    all_sd = [x for v in out["sd_by_rank"].values() for x in v]
    out["sd_all"] = [min(all_sd), max(all_sd)]
    out["mean_abs_delta"] = float(np.mean(np.abs(deltas)))
    out["max_abs_delta"] = float(np.max(np.abs(deltas)))
    out["verdicts"] = {v: sum(c["verdict"] == v for c in out["cells"].values())
                       for v in ("ЭКВИВАЛЕНТНОСТЬ", "НЕОПРЕДЕЛЁННОСТЬ", "ПРЕВОСХОДСТВО", "УСТУПАЕТ")}
    out["mean_ci_narrowing_pct"] = float(np.mean([c["ci_narrowing_pct"] for c in out["cells"].values()]))
    worst_ts = [c["worst_seed_two_stage"] for c in out["cells"].values()]
    worst_do = [c["worst_seed_dmid_only"] for c in out["cells"].values()]
    out["worst_seed_counts"] = {"two_stage": {str(s): worst_ts.count(s) for s in set(worst_ts)},
                                "dmid_only_by_config": {str(s): worst_do[:3].count(s) for s in set(worst_do)}}
    pre = json.load(open(f"{RES}/equivalence{suffix}.json"))["by_config"]["r16_a32"]
    out["preregistered"] = {"delta": pre["delta_mean"], "ci": [pre["ci_low"], pre["ci_high"]],
                            "n_runs": pre["n_runs"], "verdict": pre["verdict"]}
    return out


def main():
    res = json.load(open(eqv.SWEEP))
    out = {s: split_summary(res, s) for s in ("clean49", "full52")}
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False, default=float)
    for s, o in out.items():
        print(f"\n== {s}: сиды {o['seeds']}")
        for k, c in o["cells"].items():
            print(f"  {k:14s} Δ {c['delta']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] {c['verdict'][:5]} "
                  f"(3 сида: {c['delta_3seed']:+.4f} {c['verdict_3seed'][:5]}; CI уже на {c['ci_narrowing_pct']:.0f}%)  "
                  f"худший сид ts {c['worst_seed_two_stage']} / do {c['worst_seed_dmid_only']}")
        print(f"  вердикты {o['verdicts']}  среднее |Δ| {o['mean_abs_delta']:.4f}  max |Δ| {o['max_abs_delta']:.4f}")
        print(f"  SD по рангу {o['sd_by_rank']}  все {o['sd_all']}  размах {o['spread_by_rank']}")
        print(f"  DMID-only средние " + ", ".join(f"{c} {v['mean']:.4f}" for c, v in o["dmid_only"].items()))
        print(f"  знаки: r16 {o['signs_r16_a32']} r32 {o['signs_r32_a64']} r64 {o['signs_r64_a64']}")
        print(f"  сужение CI в среднем {o['mean_ci_narrowing_pct']:.0f}%  худший сид {o['worst_seed_counts']}")
        print(f"  пререгистрированный: {o['preregistered']}")


if __name__ == "__main__":
    main()
