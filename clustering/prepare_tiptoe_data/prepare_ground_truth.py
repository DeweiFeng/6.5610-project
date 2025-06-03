import numpy as np
import json
import csv

def convert_ground_truth(reverse_index_path, ground_truth_array, output_csv_path):
    # Load reverse index mapping
    with open(reverse_index_path, "r") as f:
        reverse_index = json.load(f)

    num_queries, top_k = ground_truth_array.shape
    print(num_queries)
    print(top_k)
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


reverse_index_path = "tiptoe_baseline/reverse_index.json"
ground_truth_array = np.load("./eval_data/ground_truth_test_k10_reduced.npy")
convert_ground_truth(reverse_index_path, ground_truth_array, "tiptoe_baseline/msmarco_ground_truth.csv")

# Example usage:
# reverse_index_path = "clusters/reverse_index.json"
# ground_truth_array = np.array([[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]])
# convert_ground_truth(reverse_index_path, ground_truth_array, "transformed_ground_truth.csv")

