#!/usr/bin/env bash
# Round 3 (R1.2, R3.5): сетка расширяется с трёх до пяти инициализаций.
# Добавляются сиды 45, 46 во все 27 ячеек two_stage и 9 ячеек dmid_only.
#   Stage 1: 3 дозы × 2 сида = 6;  Stage 2: 18 two_stage + 6 dmid_only = 24.
# dmid_only от дозы не зависит, поэтому считается один раз, в фазе 1444.
# Прогон возобновляемый: готовые ячейки пропускаются по ключу в capacity_sweep.json.
set -u
cd .
PY=/home/gani/miniconda3/envs/rag_mammo/bin/python
S=scripts/7_revision/05_capacity_sweep.py

run() { echo -e "\n\n##### $* #####\n"; $PY $S "$@" || echo "!!! фаза упала: $* (продолжаю)"; }

run --synth-v2 --dose 1444 --configs 16:32 64:64 32:64 --seeds 45 46 --arms two_stage dmid_only
run --synth-v2 --dose 800  --configs 16:32 64:64 32:64 --seeds 45 46 --arms two_stage
run --synth-v2 --dose 369  --configs 16:32 64:64 32:64 --seeds 45 46 --arms two_stage

echo -e "\n\n===== СИДЫ 45/46 ЗАВЕРШЕНЫ ====="
