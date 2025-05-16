import pandas as pd

df_eval = pd.read_csv('baseline_msmarco/msmarco_results_cluster_only.csv', header=None)

# Merge adjacent columns into tuples: even-indexed column first
merged_cols = {f'merged_{i//2}': list(zip(df_eval.iloc[:, i], df_eval.iloc[:, i+1])) for i in range(0, 20, 2)}

df_eval = pd.DataFrame(merged_cols)

eval_rows = df_eval.values.tolist()

df_gt = pd.read_csv('baseline_msmarco/msmarco_ground_truth.csv', header=None)

merged_cols = {f'merged_{i//2}': list(zip(df_gt.iloc[:, i], df_gt.iloc[:, i+1])) for i in range(0, 20, 2)}

df_gt = pd.DataFrame(merged_cols)

gt_rows = df_gt.values.tolist()

recall_scores = []
for eval_row, gt_row in zip(eval_rows, gt_rows):
    overlap = len(set(eval_row) & set(gt_row))
    intersection = set(eval_row).intersection(set(gt_row))
    recall_scores.append(overlap / 10)

mrr_scores = []

for eval_row, gt_row in zip(eval_rows, gt_rows):
    reciprocal_rank = 0.0
    relevant = gt_row[0]
    for rank, doc_id in enumerate(eval_row[:10]):  # top-10 only
        if doc_id == relevant:
            reciprocal_rank = 1.0 / (rank+1)
            break
    mrr_scores.append(reciprocal_rank)

mrr_at_10 = sum(mrr_scores) / len(mrr_scores)
print("MRR@10:", mrr_at_10)


import numpy as np

print("RECALL@10")
print(np.mean(recall_scores))

