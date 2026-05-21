#!/bin/bash

BASE_PATH="."

# Create required directories if they don't exist
# mkdir -p $BASE_PATH/data
# mkdir -p $BASE_PATH/data/averitec
mkdir -p $BASE_PATH/knowledge_store

# For downloading json files
# if [ ! -f "$BASE_PATH/data_store/averitec/train.json" ]; then
#     wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json -O $BASE_PATH/data_store/averitec/train.json
# fi

# if [ ! -f "$BASE_PATH/data_store/averitec/dev.json" ]; then
#     wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json -O $BASE_PATH/data_store/averitec/dev.json
# fi

# if [ ! -f "$BASE_PATH/data_store/averitec/test_2025.json" ]; then
#     wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/test_2025.json -O $BASE_PATH/data_store/averitec/test_2025.json
# fi

# For knowledge store - Averitec
# if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
#     wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip -O $BASE_PATH/knowledge_store/dev_knowledge_store.zip
#     unzip $BASE_PATH/knowledge_store/dev_knowledge_store.zip -d $BASE_PATH/knowledge_store/
#     mv $BASE_PATH/knowledge_store/output_dev $BASE_PATH/knowledge_store/dev
#     rm $BASE_PATH/knowledge_store/dev_knowledge_store.zip
# fi

if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
    wget https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/urls/dev_urls.zip -O $BASE_PATH/knowledge_store/dev_urls.zip
    unzip $BASE_PATH/knowledge_store/dev_urls.zip -d $BASE_PATH/knowledge_store/
    # mv $BASE_PATH/knowledge_store/output_dev $BASE_PATH/knowledge_store/dev
    # rm $BASE_PATH/knowledge_store/averitec_dev_urls.zip
fi

# # For knowledge store - FEVER
# if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
#     wget https://fever.ai/download/fever/wiki-pages.zip -O $BASE_PATH/knowledge_store/wiki-pages.zip
#     unzip $BASE_PATH/knowledge_store/wiki-pages.zip -d $BASE_PATH/knowledge_store/
#     mv $BASE_PATH/knowledge_store/wiki-pages $BASE_PATH/knowledge_store/fever
#     rm $BASE_PATH/knowledge_store/wiki-pages.zip
# fi

# # For knowledge store - FEVERous
# if [ ! -d "$BASE_PATH/knowledge_store/dev" ]; then
#     wget https://fever.ai/download/feverous/feverous-wiki-pages.zip -O $BASE_PATH/knowledge_store/feverous-wiki-pagess.zip
#     unzip $BASE_PATH/knowledge_store/feverous-wiki-pages.zip -d $BASE_PATH/knowledge_store/
#     mv $BASE_PATH/knowledge_store/FeverousWikiv1 $BASE_PATH/knowledge_store/feverous
#     rm $BASE_PATH/knowledge_store/feverous-wiki-pages.zip
# fi

# # For knowledge store - ClimateCheck
# if [ ! -d "$BASE_PATH/knowledge_store/" ]; then
#     wget https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus/resolve/main/climatecheck_publications_corpus.parquet -O $BASE_PATH/knowledge_store/climatecheck_publications_corpus.parquet
#     unzip $BASE_PATH/knowledge_store/climatecheck_publications_corpus.parquet -d $BASE_PATH/knowledge_store/
#     mv $BASE_PATH/knowledge_store/climatecheck_publications_corpus $BASE_PATH/knowledge_store/climatecheck
#     rm $BASE_PATH/knowledge_store/climatecheck_publications_corpus.parquet
# fi