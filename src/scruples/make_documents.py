import json
import argparse

from tqdm import tqdm
from src.data_utils import load_json_file, save_json_file


def main(args):
    # load dataset
    dataset = load_json_file(args.file_path)
    
    results = []
    for data in tqdm(dataset):
        results.append(
            {
                'doc_id': data['ex_id'],
                'rot': data['rot'],
                'rot_ir': data['rot_ir'],
            }
        )
    
    save_json_file(results, args.output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, default="data/scruples/3shot/scruples_sim.json")
    args.add_argument("--output_path", type=str, default="data/scruples/3shot/norms.json")
    args = args.parse_args()
    main(args)