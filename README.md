# Normative RAG

This repository contains the code and resources for **Normative RAG**, a framework for evaluating and improving normative reasoning in Large Language Models (LLMs) using Retrieval-Augmented Generation (RAG).

The project currently supports experimentation with:
- **Datasets:** NormAD, Scruples
- **Methods:** Dense Retrieval, HyDE, Naive Generation, RAG, Rewrite-Retrieve-Read
- **Models:** GPT-4, vLLM-supported open-source models

## 📂 Project Structure

```bash
.
├── data/                   # Dataset files (NormAD, Scruples)
├── index/                  # Pre-computed dense vector indices
├── scripts/                # Shell scripts for server
└── src/
    ├── normad/             # NormAD-specific experiments (retrieve, generate, eval)
    ├── scruples/           # Scruples-specific experiments (retrieve, generate, eval)
    ├── search/             # Retriever implementations (E5, etc.)
    ├── ethic_agent.py      # Core agent logic
    ├── gpt4.py             # OpenAI API wrapper
    ├── vllm_client.py      # vLLM client implementation
    └── ...
```

## 1. Start VLLM Server (Optional)
If you plan to use open-source LLMs hosted locally, start the vLLM server first:
bash scripts/start_vllm_server.sh

## 2. Dense Retrieval
Perform retrieval to find relevant norms or situations from the dataset.
Example (NormAD):

```
# Run dense retrieval using E5-Large
python3 -m src.normad.dense_retrieve \
    --retriever e5_large
```

## 3. RAG Generation
Example (NormAD):
Generate responses based on the retrieved documents.
```
python3 -m src.normad.rag \
    --dataset_name normad\
    --retriever e5_large \
    --model gpt4 \
    --batch_size 16
```

## Evaluation
Example (NormAD):
Evaluate the generated responses against gold labels or using model-based evaluation.
```
# Evaluate retrieval performance
python3 -m src.normad.dense_retrieval_eval \
    --retriever e5_large
```

# Evaluate RAG generation quality
```
python3 -m src.normad.rag_eval \
    --input_dir your_input_dir \
    --output_dir your_output_dir
```
