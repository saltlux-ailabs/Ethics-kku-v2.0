import re
import json
import argparse

from tqdm import tqdm
# from src.ethic_agent import EthicAgent
from src.prompts import scruples_query2doc_prompt
from src.data_utils import load_json_file, save_json_file
# from src.vllm_client import VllmClient
from src.gpt4 import GPT4

# # setup
# vllm_client: VllmClient = VllmClient()
# agent: EthicAgent = EthicAgent(vllm_client)

agent: GPT4 = GPT4()


def main(args):
    # load dataset
    dataset = load_json_file(f"outputs/{args.dataset_name}/scruples_sim.json")

    # build prompts
    prompts = [scruples_query2doc_prompt(d['story']) for d in dataset]

    results = []
    batch_size = args.batch_size
    num_items = len(prompts)
    num_batches = (num_items + batch_size - 1) // batch_size

    for start in tqdm(range(0, num_items, batch_size), total=num_batches, desc="Generating"):
        end = min(start + batch_size, num_items)
        batch_prompts = prompts[start:end]

        # batch generate
        batch_responses = agent._batch_generate(batch_prompts)

        # attach back to original items
        for offset, response in enumerate(batch_responses):
            item = dataset[start + offset]
            item['generated_rot'] = response
            results.append(item)

    # save results
    save_json_file(results, f"outputs/{args.dataset_name}/query2doc.json")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="scruples")
    args.add_argument("--batch_size", type=int, default=16)
    args = args.parse_args()
    main(args)