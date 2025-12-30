import os
import argparse

from src.data_utils import load_json_file, save_json_file
from src.metrics import compute_acc_and_f1


def main(args):
    choices = ["base", "query2doc", "hyde", "rewrite", "ours"]
    for choice in choices:
        for top_k in [1, 3, 5, 10]:
            dataset = load_json_file(os.path.join(args.input_dir, f"{choice}_top_{top_k}.json"))
            answers = [data['answer'] for data in dataset]
            preds = [data['pred'] for data in dataset]
            
            # calculate metrics
            metrics = compute_acc_and_f1(labels=['right', 'wrong'], golds=answers, preds=preds)
            
            # save results
            result = {
                'acc': metrics['acc'],
                'f1': metrics['f1'],
                }
            save_json_file(result, os.path.join(args.output_dir, f"{choice}_top_{top_k}.json"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str, default="outputs/scruples/e5/generation/gpt4/preds")
    args.add_argument("--output_dir", type=str, default="outputs/scruples/e5/generation/gpt4/metrics")  

    args = args.parse_args()
    main(args)