#!/usr/bin/env bash
# Дообучение r=16 α=32 единым механизмом слияния + оценка на валидации и тесте.
# Закрывает последнюю неравномерность условий в Table 8 (см. R1.13, R3.2, R3.3).
set -u
cd .
PY=/home/gani/miniconda3/envs/rag_mammo/bin/python

echo -e "\n##### обучение r16_a32 (механизм слияния) #####\n"
$PY scripts/7_revision/15_train_r16_ablation.py || { echo "!!! обучение упало"; exit 1; }

echo -e "\n##### оценка на валидации #####\n"
$PY scripts/7_revision/01_infer.py --split val  --models two_stage_r16_a32 || echo "!!! val упал"

echo -e "\n##### оценка на тесте #####\n"
$PY scripts/7_revision/01_infer.py --split test --models two_stage_r16_a32 || echo "!!! test упал"

echo -e "\n===== r16_a32 ЗАКРЫТА ====="
