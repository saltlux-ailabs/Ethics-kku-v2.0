import argparse

from src.data_utils import load_json_file, save_json_file


def main(args):
    # load dataset
    dataset = load_json_file(args.file_path)
    
    results = []
    for data in dataset:
        results.append(
            {
                'doc_id': data['ex_id'],
                'country': data['country'],
                'region': data['region'],
                'answer_rot': data['answer_rot']
            }
        )
    
    save_json_file(results, args.output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, default="data/normad/normad_region.json")
    args.add_argument("--output_path", type=str, default="data/normad/normad_region_norms.json")
    args = args.parse_args()
    main(args)