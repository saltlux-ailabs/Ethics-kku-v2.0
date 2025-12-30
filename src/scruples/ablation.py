import argparse

from src.data_utils import load_json_file, save_json_file
from src.search.contriever import Contriever
# from src.search.e5_searcher import E5Searcher
# from src.search.miniLM_searcher import MiniLMSearcher


def main(args):
    # load dataset
    dataset = load_json_file(f"data/scruples/{args.shot}/scruples_sim.json")
        
    if args.output_type == "story_ir_rot":
        index_version = "v0"
        batch_queries = [f"{data['story_ir']}" for data in dataset]
    elif args.output_type == "story_rot_ir":
        index_version = "v1"
        batch_queries = [f"{data['story']}" for data in dataset]
    else:
        raise ValueError(f"Unsupported output type: {args.output_type}")

    # load searcher
    searcher = Contriever(index_version=index_version, dataset="scruples", shot=args.shot)
    # searcher = E5Searcher(index_version=index_version, dataset="scruples", shot=args.shot)
    
    for topk in [1, 3, 5, 10]:
        batch_docs = searcher.batch_search_with_docs(
            batch_queries,
            k=topk,
        )

        output_path = f"outputs/scruples/contriever/retrieval/{args.shot}/{args.output_type}_top_{topk}.json"

        results = []
        for data, docs in zip(dataset, batch_docs):
            results.append(
                {
                    'ex_id': data['ex_id'],
                    'story': data['story'],
                    'answer': data['answer'],
                    'rot': data['rot'],
                    'retrieved_rot': docs,
                }
            )

        save_json_file(results, output_path)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=str, default="3shot", choices=["1shot", "2shot", "3shot"])
    parser.add_argument("--output_type", type=str, default="story_ir_rot", choices=["story_ir_rot", "story_rot_ir"])
    args = parser.parse_args()
    main(args)