#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --constraint=48GB
#SBATCH --job-name=fcbench
#SBATCH --mem=64G
#SBATCH -o FCBench/logs/run_%j.out # STDOUT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

GEMINI_KEY=""
export HUGGING_FACE_HUB_TOKEN=""

SYSTEM_NAME="fcbench"
SPLIT="dev"
BASE_DIR="." # Current directory

# Dataset configuration
# Supported values: averitec | climatecheck | climatefever | scifact
DATASET_TYPE="averitec"
SPARSE_METHOD="random" # bm25 | tfidf | random

# Paths — adjust per dataset
DATA_STORE="FCBench/data/cleaned_datasets"
KNOWLEDGE_STORE="FCBench/knowledge_store/${DATASET_TYPE}"


# python -m src.data_augmentation -b 8 -t 'evidences'
# python -m src.knowledge_augmentation -d 'feverous'

# python -m src.model.sparse --CLAIMS_PATH "${DATA_STORE}/${DATASET_TYPE}/${SPLIT}.pkl" \
#                            --KNOWLEDGE_STORE_PATH "${KNOWLEDGE_STORE}" \
#                            --dataset_type "${DATASET_TYPE}" \
#                            --method "${SPARSE_METHOD}"

# TFIDF/BM25 baseline -b| Longformer -l | distilRoBERTa -d 
# torchrun --nproc_per_node=2 -m src.experiment_baseline -- \
#         --log logs/auto/log.out \
#         --reset \
#         --dataset_list averitec scifact climatecheck climatefever \
#         --seed_list 42 26 123 \
#         -l \
#         --batch_size 4 \
#         --accumulation_steps 8

# Claim-only LLM: retrieval none | LLM + BM25 retrieval bm25
python3 -m src.experiment_llm \
        --log logs/auto/llm_bm25 \
        --dataset_list climatecheck climatefever \
        --retrieval bm25 \ 
        --top_k 5 \
        --reset
