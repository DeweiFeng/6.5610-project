import pandas as pd
import argparse

def merge_columns(df, num_pairs=10):
    return pd.DataFrame({
        f'merged_{i}': list(zip(df.iloc[:, 2*i], df.iloc[:, 2*i + 1]))
        for i in range(num_pairs)
    })

def main(args):
    # Load and merge evaluation results
    df_eval = pd.read_csv(args.eval_path, header=None)
    df_eval = merge_columns(df_eval)
    eval_rows = df_eval.values.tolist()

    # Load and merge ground truth
    df_gt = pd.read_csv(args.ground_truth_path, header=None)
    df_gt = merge_columns(df_gt)
    gt_rows = df_gt.values.tolist()

    # Compute recall scores
    recall_scores = [
        len(set(eval_row) & set(gt_row)) / 10
        for eval_row, gt_row in zip(eval_rows, gt_rows)
    ]

    # Print mean recall if desired (or return/save if needed)
    print(f"Mean Recall@10: {sum(recall_scores)/len(recall_scores):.4f}")

    # Compute MRR

    mrr_scores = []

    for eval_row, gt_row in zip(eval_rows, gt_rows):
        reciprocal_rank = 0.0
        relevant = gt_row[0]
        for rank, doc_id in enumerate(eval_row[:10]):  # top-10 only
            if doc_id == relevant:
                reciprocal_rank = 1.0 / (rank + 1)
                break

        mrr_scores.append(reciprocal_rank)

    print(f"MRR@10: {sum(mrr_scores)/len(mrr_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recall@10 from result and ground truth CSVs.")

    parser.add_argument('--eval_path', required=True, help="Path to evaluation results CSV")
    parser.add_argument('--ground_truth_path', required=True, help="Path to ground truth CSV")

    args = parser.parse_args()
    main(args)

