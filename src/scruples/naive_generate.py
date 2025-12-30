import os
import argparse

from tqdm import tqdm

from src.data_utils import load_json_file, save_json_file
from src.prompts import normad_naive_generation_prompt
from src.vllm_client import VllmClient
from src.ethic_agent import EthicAgent
# from src.gpt4 import GPT4


# setup
vllm_client: VllmClient = VllmClient(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
agent: EthicAgent = EthicAgent(vllm_client)
# agent: GPT4 = GPT4()


def main(args):
    # load dataset
    dataset = load_json_file(f"outputs/{args.dataset_name}/{args.retriever}/retrieval/{args.type}.json")

    # build prompts
    system_prompts, user_prompts = [], []
    for data in dataset:
        system_prompt, user_prompt = normad_naive_generation_prompt(data)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt)

    results = []
    batch_size = args.batch_size
    num_items = len(system_prompts)
    num_batches = (num_items + batch_size - 1) // batch_size

    for start in tqdm(range(0, num_items, batch_size), total=num_batches, desc="Generating"):
        # setup the end point
        end = min(start + batch_size, num_items)

        # prepare batch prompts
        batch_system_prompts = system_prompts[start:end]
        batch_user_prompts = user_prompts[start:end]

        # batch generate
        batch_responses = agent.batch_generate(batch_system_prompts, batch_user_prompts, temperature=0.0)

        # attach back to original items
        for offset, response in enumerate(batch_responses):
            item = dataset[start + offset]
            results.append(
                {
                    'ex_id': item['ex_id'],
                    'story': item['story'],
                    'answer': item['answer'],
                    'pred': response,
                }
            )

    # save results
    output_path = f"outputs/{args.dataset_name}/{args.retriever}/generation/{args.model}/preds/{args.type}_naive.json"
    save_json_file(results, output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str, default="outputs/normad/miniLM/retrieval/3shot")
    args.add_argument("--dataset_name", type=str, default="normad")
    args.add_argument("--retriever", type=str, default="miniLM")
    args.add_argument("--model", type=str, default="gpt4")
    args.add_argument("--type", type=str, default="base")
    args.add_argument("--batch_size", type=int, default=16)
    args = args.parse_args()
    main(args)