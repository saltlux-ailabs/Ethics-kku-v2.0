import os
import argparse

from tqdm import tqdm

from src.data_utils import load_json_file, save_json_file
from src.prompts import scruples_rag_prompt
# from src.vllm_client import VllmClient
# from src.ethic_agent import EthicAgent
from src.gpt4 import GPT4


# setup
# vllm_client: VllmClient = VllmClient(model="Qwen/Qwen2.5-7B-Instruct")
# agent: EthicAgent = EthicAgent(vllm_client)
agent: GPT4 = GPT4()


def main(args):
    choices = ["base", "query2doc", "rewrite", "hyde", "ours"]
    for choice in choices:
        
        for topk in [1, 3, 5, 10]:
            # load dataset
            dataset = load_json_file(f"outputs/{args.dataset_name}/{args.retriever}/retrieval/{choice}_top_{topk}.json")

            # build prompts
            prompts = [scruples_rag_prompt(data) for data in dataset]

            results = []
            batch_size = args.batch_size
            num_items = len(prompts)
            num_batches = (num_items + batch_size - 1) // batch_size

            for start in tqdm(range(0, num_items, batch_size), total=num_batches, desc="Generating"):
                end = min(start + batch_size, num_items)
                batch_prompts = prompts[start:end]

                # batch generate
                batch_responses = agent.batch_generate(batch_prompts, temperature=0.0)

                # attach back to original items
                for offset, response in enumerate(batch_responses):
                    item = dataset[start + offset]
                    results.append(
                        {
                            'ex_id': item['ex_id'],
                            'rot': item['rot'],
                            'story': item['story'],
                            'answer': item['answer'],
                            'pred': response,
                            'retrieved_rot': item['retrieved_rot']
                        }
                    )

            # save results
            output_path = f"outputs/{args.dataset_name}/{args.retriever}/generation/{args.model}/preds/{choice}_top_{topk}.json"
            save_json_file(results, output_path)
            print(f"Saved results to {output_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()    
    args.add_argument("--dataset_name", type=str, default="scruples")
    args.add_argument("--retriever", type=str, default="e5")
    args.add_argument("--model", type=str, default="gpt4")
    args.add_argument("--batch_size", type=int, default=16)
    args = args.parse_args()
    main(args)