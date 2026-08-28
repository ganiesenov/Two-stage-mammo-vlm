#!/usr/bin/env bash
# Драйвер дозозависимого свипа. Порядок фаз выбран так, чтобы решающие ячейки
# закрылись первыми: если максимальная доза на обоих якорях не даёт эффекта,
# отрицательный результат по сути установлен, а остальные фазы лишь достраивают кривую.
#
# Сетка: ранги (16,32) (32,64) (64,64) × дозы 369/800/1444 × сиды 42/43/44.
#   two_stage  — 27 ячеек (доза применима)
#   dmid_only  —  9 ячеек (Stage 1 отсутствует, от дозы не зависит; r=16 уже посчитан ранее)
#
# Прогон возобновляемый: готовые ячейки пропускаются по ключу в capacity_sweep.json.
set -u
cd .
# Интерпретатор окружения rag_mammo: в базовом python3 нет torch, и весь свип
# падал на import (см. logs/dose_sweep.log от 2026-08-03).
PY=/home/gani/miniconda3/envs/rag_mammo/bin/python
S=scripts/7_revision/05_capacity_sweep.py

run() { echo -e "\n\n##### $* #####\n"; $PY $S "$@" || echo "!!! фаза упала: $* (продолжаю)"; }

# ── Фаза A: максимальная доза на обоих якорях + недостающий компаратор r=64 ──
run --synth-v2 --dose 1444 --configs 64:64 16:32 --arms two_stage dmid_only

# ── Фаза B и C: остальные дозы на тех же якорях (компаратор уже посчитан) ──
run --synth-v2 --dose 800  --configs 64:64 16:32 --arms two_stage
run --synth-v2 --dose 369  --configs 64:64 16:32 --arms two_stage

# ── Фаза D: средняя точка ранга, достраивает форму кривой ──
run --synth-v2 --dose 1444 --configs 32:64 --arms two_stage dmid_only
run --synth-v2 --dose 800  --configs 32:64 --arms two_stage
run --synth-v2 --dose 369  --configs 32:64 --arms two_stage

echo -e "\n\n===== СВИП ЗАВЕРШЁН ====="
$PY - <<'EOF'
import json
d = json.load(open("results/revision/capacity_sweep.json"))
print(f"всего ячеек: {len(d)}")
for k in sorted(d):
    v = d[k]
    print(f"  {k:46s} rougeL={v.get('rougeL', float('nan')):.4f}")
EOF
