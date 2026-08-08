# Прогоны ревизии Scientific Reports

Интерпретатор: `/home/gani/miniconda3/envs/rag_mammo/bin/python` (torch 2.11+cu130,
transformers 5.5.0, peft 0.18.1). Все зависимости проверены.

Общий принцип: **все модели гоняются в одинаковых условиях** (4-bit NF4, 448 px, greedy,
200 новых токенов) и **все генерации сохраняются по-сэмплово** в
`results/revision/predictions/`. Именно отсутствие сохранённых генераций блокировало
доверительные интервалы и пересчёт клинических метрик в исходной работе.

## Порядок запуска

```bash
cd /mnt/c/Users/juman/hard_ml/rag_mammo/new_article
PY=/home/gani/miniconda3/envs/rag_mammo/bin/python

# 0. Манифест сплита и утечки — CPU, уже выполнено
python3 scripts/7_revision/00_split_manifest.py

# 1. Бейзлайны на тесте с сохранением генераций (~30 мин)
$PY scripts/7_revision/01_infer.py --split test \
    --models zero_shot dmid_only_r16 two_stage_r16 two_stage_r64

# 2. Недостающий бейзлайн равной ёмкости: DMID-only при r=64 (~40 мин)
#    Это главная методологическая правка — см. REVISION_PLAN.md п. 1.1
$PY scripts/7_revision/02_train_dmid_only_r64.py
$PY scripts/7_revision/01_infer.py --split test --models dmid_only_r64

# 3. Выбор конфигурации LoRA по ВАЛИДАЦИИ, а не по тесту (R1.13, R3.2) (~35 мин)
$PY scripts/7_revision/01_infer.py --split val --models ablation_all two_stage_r16

# 4. Абляция RAG на Stage 2 (R1.7, R3.5) (~15 мин)
$PY scripts/7_revision/01_infer.py --split test --models two_stage_r64 --rag

# 5. Доверительные интервалы и парные тесты — CPU (~10 мин)
$PY scripts/7_revision/03_bootstrap_ci.py --split test \
    --models zero_shot dmid_only_r16 two_stage_r16 dmid_only_r64 two_stage_r64 \
    --compare two_stage_r64:dmid_only_r64 two_stage_r16:dmid_only_r16
# то же на чистом подмножестве (без 3 дословных дубликатов) — sensitivity analysis
$PY scripts/7_revision/03_bootstrap_ci.py --split test --clean-only \
    --models dmid_only_r64 two_stage_r64 --compare two_stage_r64:dmid_only_r64

# 6. Клинические метрики на всех 52 (R3.8, R1.3, R1.6) — CPU (~2 мин)
$PY scripts/7_revision/04_clinical_metrics.py --split test \
    --models zero_shot dmid_only_r16 two_stage_r16 dmid_only_r64 two_stage_r64
```

## Что чем закрывается

| Скрипт | Замечания |
|---|---|
| `00_split_manifest.py` | R1.1, R1.5, R3.1 |
| `01_infer.py` (test) | разблокирует всё остальное; R3.7 (CIDEr) |
| `02_train_dmid_only_r64.py` | R1.11, R3.2, R3.7 — и ядро проблемы из п. 1.1 плана |
| `01_infer.py --split val` | R1.13, R3.2 |
| `01_infer.py --rag` | R1.7, R3.5 |
| `03_bootstrap_ci.py` | R1.9, R2.2, R3.6 |
| `04_clinical_metrics.py` | R1.3, R1.6, R3.4, R3.8 + лист для reader study |

## Чего эти прогоны НЕ закрывают

Требует людей, не вычислений — критический путь по срокам:
- **R1.10, R2.4, R3.4** — оценка сертифицированными радиологами.
  `04_clinical_metrics.py` генерирует пустой `reader_study_sheet.csv`; интерфейс —
  в `validation_app/`.
- **R1.4, R2.3** — экспертная проверка выборки синтетических отчётов.
- **R1.3** — разметка галлюцинаций двумя аннотаторами и κ между ними
  (автоматический прокси в `04_` даёт только воспроизводимую нижнюю границу).

## Замечания по коду

- У конфигураций абляции нет верхнеуровневых адаптеров: `lora_ablation.py` не вызывал
  `trainer.save_model()`, поэтому в `01_infer.py` прописаны конкретные лучшие чекпойнты
  по `eval_loss` из `trainer_state.json`.
- `two_stage_r16` (`medgemma_dmid/`) обучалась продолжением адаптера Stage 1, а конфигурации
  абляции — через `merge_and_unload()` + новый адаптер. Это два разных механизма; в Methods
  нужно описать оба (R3.3).
- `r=16` для Table 8 стоит переобучить механизмом `merge_and_unload`, иначе эта строка
  не сопоставима с остальными.
