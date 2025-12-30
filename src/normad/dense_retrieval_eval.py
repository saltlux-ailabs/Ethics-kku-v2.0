import math
import argparse

from src.data_utils import load_json_file, save_json_file


def calculate_recall(ranked_lists_and_ground_truths):
    """
    Calculates the Recall.

    Args:
        ranked_lists_and_ground_truths (list of tuples):
            Each tuple = (ranked_list, ground_truth_item)

    Returns:
        float: Mean Recall score.
    """
    recalls = []
    for ranked_list, ground_truth_item in ranked_lists_and_ground_truths:
        if ground_truth_item in ranked_list:
            recalls.append(1)
        else:
            recalls.append(0)
    return round(sum(recalls) / len(recalls), 4)


def calculate_mrr(ranked_lists_and_ground_truths):
    """
    Calculates the Mean Reciprocal Rank (MRR).

    Args:
        ranked_lists_and_ground_truths (list of tuples): A list where each tuple
            represents a query and contains:
            - ranked_list (list): A list of items ranked by the system.
            - ground_truth_item (any): The first relevant item for that query.

    Returns:
        float: The Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    for ranked_list, ground_truth_item in ranked_lists_and_ground_truths:
        found_relevant = False

        for i, item in enumerate(ranked_list):
            if item == ground_truth_item:
                reciprocal_rank = 1 / (i + 1)  # i+1 because rank is 1-indexed
                reciprocal_ranks.append(reciprocal_rank)
                found_relevant = True
                break
        if not found_relevant:
            # If the relevant item is not found in the ranked list, its reciprocal rank is 0.
            reciprocal_ranks.append(0)

    if not reciprocal_ranks:
        return 0.0  # Handle case with no queries

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def calculate_ndcg(ranked_lists_and_ground_truths, k=None):
    """
    Calculates the Mean nDCG for single relevant document per query.
    """
    ndcg_scores = []
    
    for ranked_list, ground_truth_item in ranked_lists_and_ground_truths:
        if k is not None:
            ranked_list = ranked_list[:k]

        try:
            rank_index = ranked_list.index(ground_truth_item) + 1
            # DCG = 1 / log2(rank + 1)  (rank_index is 0-based)
            dcg = 1 / math.log2(rank_index + 1)
        except ValueError:
            dcg = 0.0  # not found

        # IDCG: DCG when the relevant document is at the top rank
        idcg = 1 / math.log2(1 + 1)  # = 1.0
        
        # calculate ndcg for each query
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    if not ndcg_scores:
        return 0.0

    return round(sum(ndcg_scores) / len(ndcg_scores), 4)


def main(args):
    metrics = {}
    for topk in [1, 3, 5, 10]:
        input_file_path = f"outputs/normad_region/{args.retriever}/retrieval/base_top_{topk}.json"
        dataset = load_json_file(input_file_path)
        pred_gold_pairs = [(data['retrieved_rot'], data['answer_rot']) for data in dataset]
        
        recall = calculate_recall(pred_gold_pairs)
        metrics[f"recall@{topk}"] = recall

        if topk == 10:
            metrics["mrr@10"] = calculate_mrr(pred_gold_pairs)
            metrics["ndcg@10"] = calculate_ndcg(pred_gold_pairs)

    output_file_path = f"outputs/normad_region/{args.retriever}/retrieval/base_metrics.json"
    save_json_file(metrics, output_file_path)
    print(f"Saved metrics to {output_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, choices=["e5_base", "e5_large", "miniLM_L6", "miniLM_L12", "bge_m3", "llm_embedder"])
    args = parser.parse_args()
    main(args)