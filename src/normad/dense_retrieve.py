import os
import argparse

from src.data_utils import load_json_file, save_json_file
from src.search.searcher import E5BaseSearcher, E5LargeSearcher, MiniLML6Searcher, MiniLML12Searcher, BGEM3Searcher, LLMEmbedderSearcher, Contriever


RETRIEVERS = {
    "e5_base": E5BaseSearcher,
    "e5_large": E5LargeSearcher,
    "miniLM_L6": MiniLML6Searcher,
    "miniLM_L12": MiniLML12Searcher,
    "bge_m3": BGEM3Searcher,
    "llm_embedder": LLMEmbedderSearcher,
    "contriever": Contriever,
}

def main(args):
    # load dataset
    input_file_path = f"data/normad/{args.dataset_name}.json"
    dataset = load_json_file(input_file_path)

    # load searcher
    searcher = RETRIEVERS[args.retriever](dataset=args.dataset_name)

    # prepare batch queries
    batch_queries = [f"{data['region']}\n{data['situation']}" for data in dataset]
    
    for topk in [1, 3, 5, 10]:
        # retrieve documents
        batch_docs = searcher.batch_search_with_docs(batch_queries, k=topk)

        # save results
        output_path = f"outputs/normad_region/{args.retriever}/retrieval/base_top_{topk}.json"
        
        results = []
        for data, docs in zip(dataset, batch_docs):
            results.append(
                {
                    'ex_id': data['ex_id'],
                    'country': data['country'],
                    'region': data['region'],
                    'statement': data['situation'],
                    'answer': data['answer'],
                    'answer_rot': data['answer_rot'],
                    'retrieved_rot': docs,
                }
            )
        save_json_file(results, output_path)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, choices=["e5_base", "e5_large", "miniLM_L6", "miniLM_L12", "bge_m3", "llm_embedder"])
    args = parser.parse_args()
    main(args)