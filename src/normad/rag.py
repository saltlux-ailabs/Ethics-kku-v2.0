import os
import argparse

from tqdm import tqdm

from src.data_utils import load_json_file, save_json_file
from src.prompts import normad_rag_prompt
from src.vllm_client import VllmClient
from src.ethic_agent import EthicAgent
# from src.gpt4 import GPT4

# setup
vllm_client: VllmClient = VllmClient(model="Qwen/Qwen3-4B-Instruct-2507")
agent: EthicAgent = EthicAgent(vllm_client)
# agent: GPT4 = GPT4(model="gpt-4o-mini")


def main(args):
    choices = ["base", "query2doc", "rewrite", "hyde", "ours"]
    for choice in choices:
        for top_k in [1, 3, 5, 10]:
            # load dataset
            input_path = f"outputs/{args.dataset_name}/{args.retriever}/retrieval/{choice}_top_{top_k}.json"
            dataset = load_json_file(input_path)
            print(f"Loaded dataset from {input_path}")

            # build prompts
            system_prompts, user_prompts = [], []
            for data in dataset:
                system_prompt, user_prompt = normad_rag_prompt(data)
                system_prompts.append(system_prompt)
                user_prompts.append(user_prompt)

            results = []
            batch_size = args.batch_size
            num_items = len(system_prompts)
            num_batches = (num_items + batch_size - 1) // batch_size

            for start in tqdm(range(0, num_items, batch_size), total=num_batches, desc="Generating"):
                end = min(start + batch_size, num_items)
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
                            'country': item['country'],
                            'situation': item['situation'],
                            'answer': item['answer'],
                            'pred': response,
                            'rot': item['rot'],
                            'retrieved_rot': item['retrieved_rot'],
                        }
                    )

            # save results
            output_path = f"outputs/{args.dataset_name}/{args.retriever}/generation/{args.model}/preds/{choice}_top_{top_k}.json"
            save_json_file(results, output_path)
            print(f"Saved results to {output_path}")
            break


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="normad")
    args.add_argument("--retriever", type=str, default="miniLM")
    args.add_argument("--model", type=str, default="gpt4")
    args.add_argument("--batch_size", type=int, default=16)
    args = args.parse_args()
    main(args)