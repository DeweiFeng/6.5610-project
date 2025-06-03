import argparse
import numpy as np
import json
import csv
import os

def convert_ground_truth(reverse_index_path, ground_truth_array, output_csv_path):
    # Load reverse index mapping
    with open(reverse_index_path, "r") as f:
        reverse_index = json.load(f)

    num_queries, top_k = ground_truth_array.shape
    transformed_data = []
    
    num_queries = 100

    for i in range(num_queries):
        row = []
        for original_doc_id in ground_truth_array[i]:
            original_doc_id_str = str(original_doc_id)
            if original_doc_id_str not in reverse_index:
                raise ValueError(f"Doc ID {original_doc_id} not found in reverse index.")
            cluster_id, position = reverse_index[original_doc_id_str]
            row.extend([cluster_id, position])  # No parentheses, just values
        transformed_data.append(row)

    # Save to CSV (no header, no parentheses)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(transformed_data)

    print(f"Saved transformed ground truth to {output_csv_path}")


def main(args):
    reverse_index_path = args.reverse_index_path
    ground_truth_array = np.load(args.ground_truth_path)

    output_path = "tiptoe_baseline/msmarco_ground_truth.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    convert_ground_truth(reverse_index_path, ground_truth_array, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ground truth using reverse index.")

    parser.add_argument('--reverse_index_path', required=True, help="Path to reverse index JSON file")
    parser.add_argument('--ground_truth_path', required=True, help="Path to ground truth .npy file")

    args = parser.parse_args()
    main(args)


#ground_truth_array = np.load("./eval_data/ground_truth_1M_k10.npy")
#convert_ground_truth(reverse_index_path, ground_truth_array, "tiptoe_baseline/msmarco_ground_truth.csv")
#
