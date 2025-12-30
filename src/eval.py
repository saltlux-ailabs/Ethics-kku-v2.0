from metrics import eval_accuracy

file_paths = [
    "/home/stv10121/ethics/src/search/data/inference_naive_rag.jsonl",
    "/home/stv10121/ethics/src/search/data/inference_hyde_rag.jsonl"
]

for file_path in file_paths:
    print(f"accuracy of {file_path}")
    eval_accuracy(file_path)