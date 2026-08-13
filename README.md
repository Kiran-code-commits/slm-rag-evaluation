# SLM RAG Evaluation: Low-Resource Urdu Tourism Question Answering

This repository accompanies a manuscript currently under review at **Information Processing &
Management**. A full citation will be added upon acceptance.

It contains the code, evaluation datasets, and results for a retrieval-augmented generation (RAG)
benchmark of five small language models (SLMs) on Urdu-language tourism question answering for
Gilgit-Baltistan, Pakistan.

## Overview

Five instruction-tuned SLMs (Qwen2.5-1.5B, Qwen3-1.7B, Gemma-2-2B, Gemma-4-E2B, TinyAya-Fire) are
evaluated under both baseline (no retrieval) and RAG (retrieval-augmented) conditions on a locked
150-question stratified evaluation set, using BERTScore F1 (mBERT and XLM-RoBERTa-large) as the
automatic metric.

## Repository structure

```
.
├── slm-rag-evaluation-pipeline.ipynb   # Main notebook: retrieval, generation, and scoring pipeline
├── requirements.txt                     # Pinned dependencies
├── LICENSE                              # MIT License
├── data/
│   ├── qa_pairs_900_full.csv            # Full curated Urdu tourism QA dataset (900 pairs)
│   └── eval_150_locked.csv              # Locked stratified evaluation subset (150 pairs; 50 easy/medium/hard)
└── results/
    ├── rag_evaluation_results_{model}_150.csv   # Per-question generations + BERTScore F1, one file per model
    ├── rag_chart_{model}_150.png                # Baseline-vs-RAG F1 chart by difficulty tier, one per model
    └── summary_all_models.csv                   # Aggregate F1 summary across all five models
```

Model key used in filenames: `qwen` = Qwen2.5-1.5B, `qwen3` = Qwen3-1.7B, `gemma` = Gemma-2-2B,
`gemma_e2b` = Gemma-4-E2B, `aya` = TinyAya-Fire.

## Dataset

The full 900-pair dataset and the locked 150-item evaluation subset are included in `data/` and
are also permanently archived on Zenodo with a persistent DOI (link and citation to be added upon
acceptance, per double-blind review requirements).

**Note on the source corpus:** the underlying 231-chunk tourism knowledge base (used for retrieval)
is *not* included in this repository or on Zenodo, as portions derive from third-party travel
content with unresolved copyright status for redistribution. The QA pairs and evaluation set —
which are original curated content — are freely available.

## Setup

This notebook was developed and tested on Kaggle's GPU notebook environment
(`torch==2.10.0+cu128`). To reproduce:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If running outside Kaggle, install a `torch` build matching your own CUDA version first — see
   [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

2. Gemma and TinyAya-Fire are gated models on Hugging Face. Accept the license terms on their
   model pages, generate an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), and provide it to the
   notebook's authentication cell (adapt from Kaggle Secrets if running elsewhere).

3. Update the `LOCKED_PATH` variable in the notebook if you move `data/eval_150_locked.csv`
   elsewhere.

4. Set `ACTIVE_MODEL` in the model-loading cell to one of `"qwen"`, `"qwen3"`, `"gemma"`,
   `"gemma_e2b"`, `"aya"`, and run all cells top to bottom. Repeat once per model to reproduce all
   five sets of results.

## Reproducibility notes

- All decoding is greedy (`do_sample=False`) with `repetition_penalty=1.1`,
  `no_repeat_ngram_size=0`, `max_new_tokens=100`, applied identically across all five models and
  both conditions.
- Dependency versions are pinned in `requirements.txt`. `transformers` in particular releases new
  versions roughly every 1–2 weeks; running with unpinned versions is not guaranteed to reproduce
  the reported results.
- The generation loop checkpoints progress and automatically invalidates its checkpoint if the
  evaluation set content changes, to avoid silently reusing stale answers after a data correction.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The dataset is released
for research use; see the Zenodo record (link forthcoming upon acceptance) for dataset-specific
terms.
