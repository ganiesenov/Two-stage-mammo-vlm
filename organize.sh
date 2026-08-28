#!/bin/bash
BASE="."
cd $BASE

# Создаём папки
mkdir -p scripts/1_generate scripts/2_rag scripts/3_finetune scripts/4_evaluate scripts/5_inference
mkdir -p results
mkdir -p data/rag

# Перемещаем RAG данные
mv birads_chunks.json birads_faiss.index birads_meta.pkl data/rag/ 2>/dev/null

# Скрипты генерации
mv generate_ollama.py generate_cbis.py scripts/1_generate/ 2>/dev/null

# RAG скрипты
mv build_rag.py birads_manual.py scrape_birads.py scripts/2_rag/ 2>/dev/null

# Fine-tune
mv finetune_multimodal.py finetune_medgemma.py scripts/3_finetune/ 2>/dev/null

# Evaluation
mv evaluate_multimodal.py evaluate_sota.py evaluate_final.py evaluate.py \
   significance.py birads_consistency.py clinical_metrics.py eval_radgraph.py scripts/4_evaluate/ 2>/dev/null

# Inference
mv rag_inference.py rag_postprocess.py scripts/5_inference/ 2>/dev/null

# Results
mv eval_results*.json eval_rag_posthoc.json birads_consistency.json \
   clinical_metrics.json significance_results.json results/ 2>/dev/null

# Удаляем устаревшие файлы
rm -f generate_reports.py

echo "Готово!"
ls -la
