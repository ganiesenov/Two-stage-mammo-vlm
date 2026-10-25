# Two-Stage Synthetic-to-Real Transfer Learning for Automated Mammography Report Generation

Code and evaluation materials for:

**Two-stage synthetic-to-real transfer learning for automated mammography report generation using vision-language models**
Gani Esen, Marat Nurtas, Jandos Amankulov, Laura La Paglia.
*Scientific Reports* (2026). https://doi.org/10.1038/s41598-026-76357-4

Archived at Zenodo: concept DOI [10.5281/zenodo.21850474](https://doi.org/10.5281/zenodo.21850474) (resolves to the latest version).

## Summary

The repository contains a two-stage synthetic-to-real training pipeline for mammography report
generation and a controlled evaluation of whether its synthetic pretraining stage contributes
to final report quality.

- **Stage 1.** BI-RADS annotations from VinDr-Mammo and CBIS-DDSM are converted into synthetic
  reports (400 generated, 390 retained) with Llama-3.1-8B, and MedGemma-4B is pretrained on them
  with multimodal LoRA.
- **Stage 2.** The model is fine-tuned on 407 real radiologist reports from DMID.

Main results:

- **Primary test set.** All results are reported on the leak-free DMID test subset (n = 49): three of the 52
  test images (Img462, Img464, Img506) have a report that appears verbatim in the training split
  and are excluded. The full 52-image split is used only as a sensitivity analysis.
- **Text metrics.** The two-stage model at r = 64 reaches ROUGE-L 0.656, METEOR 0.666, CIDEr 0.649 and BERTScore
  0.917.
- **At matched adapter capacity the two training schemes are equivalent.** The earlier
  apparent advantage of two-stage training was measured between models of unequal adapter
  capacity; capacity alone reproduces the gap. Across 60 training runs (three Stage 1 corpus
  sizes, three capacities, five initialisations) and an equivalence margin fixed in advance
  (Δ = 0.02 ROUGE-L), no configuration favours two-stage training: eight cells are equivalent and
  one is inconclusive.
- **Clinical findings** (surrogate measures against the reference reports, not expert adjudication):
  - under an assertion-level criterion no model fabricates findings on normal studies; the failure mode is omission (the two-stage r = 64 model omits a reference finding in 14 of 40 cases);
  - of 22 reference-suspicious cases (BI-RADS 4–5), the r = 64 models assign 7 to BI-RADS 1–2;
  - BI-RADS agreement never exceeds κ = 0.14;
  - both r = 64 models assign ACR density B to every test case, so their density accuracy equals the prevalence of B (κ = 0).
- **Textual and clinical quality diverge.** Where two-stage training gains 0.058 ROUGE-L (r = 16), BI-RADS agreement and sensitivity both fall.

The models are research artefacts. They are not medical devices and must not be used for
diagnosis or treatment of patients.

## Results on the leak-free DMID test subset (n = 49)

Table 4 of the paper. Rows marked ‡ are quoted on the full 52-image split: per-sample generations
were not retained for those models, so they cannot be recomputed on the leak-free subset and are
shown for context only. The AMRG row is reported on the full 52-image split as published. It used
a different split assignment and a different category-extraction rule, so it is not directly
comparable; its BLEU-4 entry is omitted because the source reports BLEU-1 only. Best value per
column in bold.

| Method | Type | BLEU-1 | BLEU-4 | ROUGE-1 | ROUGE-L | METEOR | CIDEr | BERTScore |
|---|---|---|---|---|---|---|---|---|
| LLaVA-1.6-7B‡ | VLM, ZS | 0.059 | 0.001 | 0.157 | 0.109 | 0.188 | 0.005 | 0.810 |
| MedGemma-4B | VLM, ZS | 0.062 | 0.001 | 0.152 | 0.098 | 0.191 | 0.000 | 0.805 |
| Qwen2.5-VL-7B‡ | VLM, ZS | 0.072 | 0.002 | 0.202 | 0.145 | 0.207 | 0.009 | 0.823 |
| Qwen3-VL-8B‡ | VLM, ZS | 0.108 | 0.002 | 0.236 | 0.147 | 0.228 | 0.007 | — |
| MedCLIP+GPT2‡ | Pipeline, FT | 0.378 | 0.217 | 0.648 | 0.571 | 0.555 | 0.054 | 0.905 |
| CLIP+GPT2‡ | Pipeline, FT | 0.384 | 0.238 | 0.673 | 0.613 | 0.591 | 0.078 | 0.909 |
| AMRG (Sung et al.)‡ | VLM, FT | 0.308 | — | 0.575 | 0.569 | 0.615 | 0.582 | — |
| DMID-only, r = 16 | VLM, FT | 0.432 | 0.193 | 0.660 | 0.586 | 0.578 | 0.362 | 0.907 |
| Two-stage, r = 16 | VLM, 2S-FT | 0.488 | 0.268 | 0.703 | 0.644 | 0.638 | 0.556 | 0.914 |
| DMID-only, r = 64 | VLM, FT | **0.532** | **0.305** | **0.723** | 0.655 | **0.673** | 0.612 | **0.918** |
| Two-stage, r = 64 | VLM, 2S-FT | 0.520 | 0.302 | 0.718 | **0.656** | 0.666 | **0.649** | 0.917 |

ZS: zero-shot; FT: fine-tuned; 2S-FT: two-stage fine-tuning. At r = 64 the paired difference
two-stage minus DMID-only is +0.0010 ROUGE-L (95% CI [−0.0187, +0.0164], p = 0.91).

Clinical metrics of the fine-tuned models (Table 6 of the paper, matched adapter rank):

| Method | BI-RADS exact | κ | Downgraded (of 22) | Sensitivity | Omission | Hallucination |
|---|---|---|---|---|---|---|
| DMID-only, r = 16 | 0.327 | 0.126 | 8 | 0.575 | 0.425 | 0.000 |
| Two-stage, r = 16 | 0.286 | 0.080 | 10 | 0.525 | 0.475 | 0.000 |
| DMID-only, r = 64 | 0.327 | 0.131 | 7 | 0.675 | 0.325 | 0.000 |
| Two-stage, r = 64 | 0.327 | 0.133 | 7 | 0.650 | 0.350 | 0.000 |

## Model weights

The two-stage r = 64 adapters (Stage 1 and Stage 2) are available on the Hugging Face Hub:
[ganiesen/medgemma-4b-mammo-report-two-stage-r64](https://huggingface.co/ganiesen/medgemma-4b-mammo-report-two-stage-r64).
They are a Model Derivative of MedGemma, are distributed under the Health AI Developer
Foundations Terms of Use and require accepting those terms; loading also requires your own
access to `google/medgemma-4b-it`.

## Repository layout

```
scripts/
  1_generate/     Stage 1 synthetic report generation (Ollama, Llama-3.1-8B)
  3_finetune/     Stage 1 and Stage 2 training, LoRA ablation, CLIP/MedCLIP+GPT2 baselines
  4_evaluate/     evaluation scripts of the original submission
  7_revision/     evaluation of the published version: split manifest and leakage check,
                  inference with saved per-sample generations, DMID-only r = 64 baseline,
                  bootstrap CIs, clinical metrics, capacity/corpus-size sweep, equivalence
                  tests, sensitivity analyses, figures (shared loading code: _common.py)
data/             synthetic Stage 1 corpora (original and regenerated)
results/          metrics; results/revision/ holds the per-sample generations and analyses
manuscript/       LaTeX source of the paper and supplementary material
```

The BI-RADS retrieval scripts (`scripts/2_rag/`, `scripts/5_inference/rag_*.py`, `data/rag/`) are
not part of the evaluated system: retrieval was ablated and is reported in the paper as a
negative result.

Evaluation protocol for all trained models (`scripts/7_revision/_common.py`): MedGemma-4B
(`google/medgemma-4b-it`) with 4-bit NF4 quantisation, 448 × 448 input, a single prompt, greedy
decoding, 200 new tokens.

## Datasets

The datasets are not redistributed. Obtain them from the official sources under their own terms.

| Dataset | Use | Source |
|---|---|---|
| VinDr-Mammo (Nguyen et al., 2023, [10.1038/s41597-023-02100-7](https://doi.org/10.1038/s41597-023-02100-7)) | Stage 1 annotations | [PhysioNet](https://physionet.org/content/vindr-mammo/) |
| CBIS-DDSM (Lee et al., 2017, [10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177)) | Stage 1 annotations | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) |
| DMID (Oza et al., 2024, [10.1007/s13534-023-00339-y](https://doi.org/10.1007/s13534-023-00339-y)) | Stage 2 and test, paired reports | [figshare, 10.6084/m9.figshare.24522883](https://doi.org/10.6084/m9.figshare.24522883) |

DMID is released for research and education only. The split follows AMRG (407 train,
51 validation, 52 test image-report pairs). The per-image split assignment and the leakage
check are in `results/revision/split_manifest.json`.

## Citation

```bibtex
@article{esen2026twostage,
  title   = {Two-stage synthetic-to-real transfer learning for automated mammography report generation using vision-language models},
  author  = {Esen, Gani and Nurtas, Marat and Amankulov, Jandos and La Paglia, Laura},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-76357-4}
}
```

## License

The code in this repository is released under the Apache License 2.0 (see `LICENSE`).

The base model MedGemma and any weights derived from it are not covered by this license.
They are subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
The datasets remain under the terms set by their providers.
