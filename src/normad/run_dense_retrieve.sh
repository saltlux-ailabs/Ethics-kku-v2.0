#!/bin/bash

models=("e5_base" "e5_large" "miniLM_L6" "miniLM_L12" "bge_m3" "llm_embedder")

for model in "${models[@]}"
do
    echo "Running retriever: $model"
    python3 -m src.normad_region.dense_retrieve --retriever "$model"
done