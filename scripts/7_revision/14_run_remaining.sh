#!/usr/bin/env bash
# Три прогона, оставшиеся от плана ревизии после закрытия дозозависимого свипа.
# Порядок: сначала короткая абляция RAG (падает быстро, если индекс или faiss не на месте),
# затем длинный проход по валидации.
set -u
cd /mnt/c/Users/juman/hard_ml/rag_mammo/new_article
PY=/home/gani/miniconda3/envs/rag_mammo/bin/python
S=scripts/7_revision/01_infer.py

run() { echo -e "\n\n##### $* #####\n"; $PY $S "$@" || echo "!!! прогон упал: $* (продолжаю)"; }

# ── Абляция RAG на Stage 2 (R1.7, R3.5) ──
# Ветвь «без RAG» уже сохранена как two_stage_r64__test.json в тех же условиях
# (промпт B, 4-bit NF4, 448 px, greedy), поэтому доcчитывается только ветвь с RAG.
run --split test --models two_stage_r64 --rag --rag-k 1

# ── Выбор конфигурации LoRA по валидации (R1.13, R3.2) ──
# Исходная абляция выбирала r/α по тесту. Здесь все конфигурации, чьи чекпойнты живы,
# оцениваются на валидации; тест остаётся для одной финальной модели.
run --split val --models ablation_all

echo -e "\n\n===== ОСТАВШИЕСЯ ПРОГОНЫ ЗАВЕРШЕНЫ ====="
