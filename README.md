# Transferable Evaluation of Evidence Retrieval and Claim Verification across Fact-Checking Benchmarks

FCBench is a modular benchmark framework for evaluating fact-checking pipelines across four datasets. It separates evidence retrieval from veracity prediction to pinpoint where systems fail — in finding evidence or in reasoning over it.

---

## Overview

Fact-checking as a two-stage pipeline:

1. **Evidence Retrieval** — Given a claim, retrieve relevant evidence documents from a knowledge store (TF-IDF, BM25, or neural methods).
2. **Veracity Prediction** — Classify the claim as *Supported*, *Refuted*, *Not Enough Evidence*, or *Conflicting Evidence*, either from the claim alone or conditioned on retrieved evidence.

FCBench runs both stages across four datasets and multiple model families, reporting per-class metrics and retrieval quality at various cutoffs.

---

## Datasets

| Dataset | Labels | Domain | Evidence |
|---|---|---|---|
| **AVeriTeC** | Supported / Refuted / Not Enough Evidence / Conflicting Evidence | General web | Web-scraped documents, multi-hop reasoning |
| **SciFact** | Supported / Refuted / Not Enough Evidence | Science | Scientific abstracts |
| **ClimateCheck** | Supported / Refuted / Not Enough Evidence | Climate science | Climate publications |
| **ClimateFEVER** | Supported / Refuted / Not Enough Evidence / Conflicting Evidence | Climate science | FEVER-style, imbalanced |

Processed datasets are stored in `cleaned_datasets/{dataset}/`.

---

## Project Structure

```
FCBench/
├── src/
│   ├── experiment_baseline.py      # Baselines and neural models (claim-only)
│   ├── experiment_retrieval.py     # Evidence-augmented transformer training
│   ├── experiment_llm.py           # LLM-based fact-checking (Llama, Phi)
│   ├── data_augmentation.py        # Language/gibberish detection enrichment
│   ├── knowledge_augmentation.py   # Evidence chunking and embedding
│   ├── model/
│   │   ├── baseline.py             # TF-IDF / BM25 + Logistic Regression
│   │   ├── distilbert.py           # DistilRoBERTa fine-tuning
│   │   ├── longformer.py           # Longformer (4096-token) fine-tuning
│   │   └── sparse.py               # Sparse retrieval from knowledge store
│   └── utils/
│       ├── builder.py              # Dataset loading and preprocessing
│       ├── eval.py                 # Recall@K, Precision@K, F1@K metrics
│       ├── logger.py               # Training logging and bootstrap CI
│       ├── predict.py              # Inference pipeline
│       ├── report_veracity.py      # Result aggregation
│       ├── plot_results.py         # Publication-quality figure generation
│       └── make_splits.py          # Stratified train/test/dev splitting
├── cleaned_datasets/               # Preprocessed splits per dataset
├── experiment_results/             # All outputs (metrics, predictions, figures)
├── requirements.txt
├── run.sh                          # Example SLURM job script
└── download_data.sh                # Data download helper
```

---

## Installation

```bash
git clone <repo-url>
cd FCBench
pip install -r requirements.txt
```

For distributed training (Longformer) ensure PyTorch is installed with CUDA support. GPU memory requirements:

- DistilRoBERTa: ≥16 GB
- Longformer: ≥48 GB (supports multi-GPU via `torchrun`)

---

## Data Setup

```bash
bash download_data.sh
```

This downloads and places datasets into `data/cleaned_datasets/`. If using external LLM APIs, create a `.env` file with the required keys (see `experiment_llm.py` for expected variable names).

---

## Running Experiments

All experiment scripts are Python modules run from the project root.

### 1. Sparse and Neural Baselines (Claim-Only)

Trains TF-IDF+LR, BM25+LR, DistilRoBERTa, and Longformer on claim text without retrieved evidence.

```bash
# Sparse baselines
python3 -m src.experiment_baseline \
    --log logs/baseline.out \
    --dataset_list averitec scifact climatecheck climatefever \
    --seed_list 42 26 123 \
    --baseline \
    --reset

# DistilRoBERTa
python3 -m src.experiment_baseline \
    --log logs/distilroberta.out \
    --reset \
    --dataset_list averitec scifact climatecheck climatefever \
    --seed_list 42 26 123 \
    --distilRoBERTa \
    --batch_size 16 \
    --accumulation_steps 2

# Longformer (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m src.experiment_baseline \
    --log logs/longformer.out \
    --reset \
    --dataset_list averitec scifact climatecheck climatefever \
    --seed_list 42 26 123 \
    --longformer \
    --batch_size 2 \
    --accumulation_steps 16
```

### 2. Evidence-Augmented Models

Fine-tunes transformers on claim + retrieved evidence. Requires retrieval output in `experiment_results/{dataset}/{method}_retrieval.json`.

```bash
# DistilRoBERTa with evidence
python3 -m src.experiment_retrieval \
    --log logs/retrieval/distilroberta \
    --reset -d \
    --batch_size 16 \
    --accumulation_steps 2 \
    --dataset_list averitec scifact climatecheck climatefever \
    --seed_list 42 43 27

# Longformer with evidence (multi-GPU)
CUDA_VISIBLE_DEVICES=0 python3 -m src.experiment_retrieval \
    --log logs/retrieval/longformer \
    --reset -l \
    --batch_size 2 \
    --accumulation_steps 16 \
    --dataset_list averitec scifact climatecheck climatefever \
    --seed_list 42 43 27
```

### 3. LLM-Based Fact-Checking

Runs instruction-tuned LLMs (Llama-3.1-8B/70B) with or without retrieved evidence.

```bash
# With BM25 retrieval (top-5 passages)
python3 -m src.experiment_llm \
    --log logs/llm_bm25 \
    --dataset_list averitec scifact climatecheck climatefever \
    --retrieval bm25 \
    --top_k 5 \
    --reset

# Claim-only (no retrieval)
python3 -m src.experiment_llm \
    --log logs/llm_claim_only \
    --dataset_list averitec scifact climatecheck climatefever \
    --retrieval none \
    --reset



```
