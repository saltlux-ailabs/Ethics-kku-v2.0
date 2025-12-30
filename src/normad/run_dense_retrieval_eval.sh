#!/bin/bash

models=("e5_base" "e5_large" "miniLM_L6" "miniLM_L12" "bge_m3" "llm_embedder")

for model in "${models[@]}"
do
    echo "----------------------------------------"
    echo "Evaluating Retrieval Performance of $model"
    python3 -m src.normad_region.dense_retrieval_eval --retriever "$model"
done