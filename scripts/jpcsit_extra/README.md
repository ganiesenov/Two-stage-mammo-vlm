# JPCSIT Extra Experiments

Дополнительные эксперименты для статьи в JPCSIT (RAG-аугментация для маммографии).

## Структура

```
new_article/scripts/jpcsit_extra/
├── _common.py              # переиспользуемые функции (модель, метрики, retrieval)
├── exp1_embeddings.py      # Эксперимент 1: 4 эмбеддинг-модели
├── exp2_topk.py            # Эксперимент 2: top-k ∈ {1, 2, 3, 5, 10}
├── exp3_faiss.py           # Эксперимент 3: Flat / IVF / HNSW
└── make_jpcsit_figures.py  # генерация 5 PNG-графиков из JSON
```

Результаты сохраняются в `new_article/jpcsit_results/`:
- `exp1_embeddings.json`, `exp2_topk.json`, `exp3_faiss.json`
- `figures/fig1_embeddings.png` … `fig5_context_halluc.png`

## Зависимости (всё уже установлено для основного проекта)

```bash
pip install sentence-transformers faiss-gpu transformers peft \
            rouge-score nltk bert-score matplotlib
```

Дополнительно для exp1 нужно скачать BiomedBERT-encoder, что произойдёт автоматически
при первом запуске SentenceTransformer (~400 МБ).

## Перед запуском

1. Убедись что Ollama запущен: `ollama serve` в отдельном терминале.
2. Убедись что модель `llama3.1:8b` подтянута: `ollama list` → если нет: `ollama pull llama3.1:8b`.
3. Активируй conda-окружение проекта.

## Запуск (последовательно)

```bash
cd /mnt/c/Users/juman/hard_ml/rag_mammo/new_article/scripts/jpcsit_extra

# Эксперимент 1: ~40-80 минут
python exp1_embeddings.py

# После завершения посмотри какой энкодер выиграл и обнови BEST_ENCODER в exp2_topk.py
# (если выиграл не all-MiniLM-L6-v2)

# Эксперимент 2: ~30-60 минут
python exp2_topk.py

# Эксперимент 3: ~10-15 минут (3 индекса × 30 кейсов)
python exp3_faiss.py

# Генерация всех графиков (~10 секунд)
python make_jpcsit_figures.py
```

## Что получишь

### Числовые результаты для новых таблиц в статье

**Table 3** (Exp 1):
| Encoder | Dim | BLEU-4 | ROUGE-L | BERTScore | Halluc | Latency (ms) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | … | … | … | … | … |
| all-mpnet-base-v2 | 768 | … | … | … | … | … |
| S-PubMedBert | 768 | … | … | … | … | … |
| PubMedBert-base | 768 | … | … | … | … | … |

**Table 4** (Exp 2):
| k | Avg ctx (words) | ROUGE-L | BERTScore | Hallucination |
|---|---|---|---|---|
| 1 | … | … | … | … |
| 2 | … | … | … | … |
| 3 | … | … | … | … |
| 5 | … | … | … | … |
| 10 | … | … | … | … |

**Table 5** (Exp 3):
| Index | Build (s) | Latency (ms) | Recall@2 | ROUGE-L | BERTScore |
|---|---|---|---|---|---|
| FlatIP | … | … | 1.000 | … | … |
| IVFFlat | … | … | … | … | … |
| HNSWFlat | … | … | … | … | … |

### 5 PNG-графиков (300 DPI, готовы к вставке)

1. `fig1_embeddings.png` — bar chart: NLG метрики по 4 эмбеддингам
2. `fig2_topk.png` — line plot: ROUGE-L и hallucination vs k (с аннотациями длины контекста)
3. `fig3_faiss.png` — две панели: scatter latency-vs-recall + bar chart end-to-end
4. `fig4_heatmap.png` — heatmap: 6 метрик × 4 энкодера
5. `fig5_context_halluc.png` — scatter: длина контекста vs hallucination, цвет = ROUGE-L

## Что делать с результатами

Когда все 3 эксперимента отработают:

1. Перешли мне три JSON-файла (`exp1_embeddings.json`, `exp2_topk.json`, `exp3_faiss.json`)
   и я обновлю текст JPCSIT-статьи: вставлю реальные таблицы, реальные цифры в Discussion,
   и заменю плейсхолдеры рисунков на готовые PNG.

2. Также пришли мне `eval_dmid_full.json` ещё раз — я заметил, что цифры в текущей JPCSIT
   версии немного отличаются от твоих фактических (ROUGE-L 0.671 vs 0.662, и т.д.) —
   обновлю и их.

## Если что-то падает

- Ollama не отвечает → проверь `curl http://localhost:11434/api/tags`
- Out-of-memory на GPU → уменьши `batch_size` в `encoder.encode(...)` в exp1
- BiomedBERT не скачивается → попробуй `huggingface-cli login` или скачай вручную
- Скрипт упал на 3-м из 4-х энкодеров в exp1 → не страшно, JSON сохраняется после
  каждого энкодера, можешь руками удалить упавший и продолжить

## Время выполнения (RTX 5090, оценка)

| Эксперимент | Время |
|---|---|
| Exp 1 (4 энкодера × 30 кейсов × Ollama refine) | 40-80 мин |
| Exp 2 (5 значений k × 30 кейсов × Ollama refine) | 30-60 мин |
| Exp 3 (3 индекса × 30 кейсов + латентность) | 10-15 мин |
| **Итого** | **~1.5-2.5 часа** |

Можно запустить все 3 последовательно одной командой:
```bash
python exp1_embeddings.py && python exp2_topk.py && python exp3_faiss.py && python make_jpcsit_figures.py
```
