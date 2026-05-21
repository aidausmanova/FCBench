#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --constraint=48GB
#SBATCH --job-name=fcbench
#SBATCH --mem=64G
#SBATCH -o /storage/usmanova/FCBench/logs/exp/run_%j.out # STDOUT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

GEMINI_KEY="AIzaSyBZ8tgGpKN_UBuw2pbqYhjrG6WihnjOx1A"

SYSTEM_NAME="blue_flash"
SPLIT="dev"  # Change this to "dev", or "test"
BASE_DIR="." # Current directory

# Dataset configuration
# Supported values: averitec | climatecheck | climatefever | scifact | fever | feverous
DATASET_TYPE="averitec"
SPARSE_METHOD="random" # bm25 | tfidf | random

# Paths — adjust per dataset
DATA_STORE="/storage/usmanova/FCBench/data/cleaned_datasets"
KNOWLEDGE_STORE="/storage/usmanova/FCBench/knowledge_store/${DATASET_TYPE}"

export HUGGING_FACE_HUB_TOKEN="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"

# python -m src.data_augmentation -b 8 -t 'evidences'
# python -m src.knowledge_augmentation -d 'feverous'

# python -m src.preprocess_fever \
#     -c data/cleaned_datasets/fever/dev.pkl \
#     -k knowledge_store/fever/ \
#     -o knowledge_store/fever_per_claim/dev/ \
#     -n 10 \
#     --include_gold 

# python -m src.model.sparse --CLAIMS_PATH "${DATA_STORE}/${DATASET_TYPE}/${SPLIT}.pkl" \
#                            --KNOWLEDGE_STORE_PATH "${KNOWLEDGE_STORE}" \
#                            --dataset_type "${DATASET_TYPE}" \
#                            --method "${SPARSE_METHOD}"

# python -m src.utils.eval --gemini_key "${GEMINI_KEY}" \
#                          --gemini_model gemini-2.0-flash \
#                          --proxy_model rausch/deberta-climatecheck-2463191-step26000 \
#                          --ev2r_n 5 \
#                          --cache_db experiment_results/cache/ev2r_cache.db

# python -m src.utils.eval

# python3 -m src.utils.predict \
#         --distilbert \
#         --hub_token "${HUGGING_FACE_HUB_TOKEN}" \
#         --batch_size 1 \
#         --accumulation_steps 8

        #    --longformer \

# torchrun --nproc_per_node=2 -m src.experiment -- \
#         --log logs/auto/log.out \
#         --reset \
#         --dataset_list averitec scifact climatecheck climatefever \
#         --seed_list 42 26 123 \
#         -l \
#         --batch_size 4 \
#         --accumulation_steps 8

# Claim-only LLM (no retrieval)
# python3 -m src.experiment_llm \
#         --log logs/auto/llm_claim_only \
#         --dataset_list averitec scifact climatecheck climatefever \
#         --retrieval none \
#         --reset

# LLM + BM25 retrieved evidence (top-5 passages)
python3 -m src.experiment_llm \
        --log logs/auto/llm_bm25 \
        --dataset_list climatecheck climatefever \
        --retrieval bm25 \
        --top_k 5 \
        --reset

