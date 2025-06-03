import numpy as np
import argparse
import os

def main(args):
    # Load and slice query vectors
    query_vectors = np.load(args.query_vectors_path)[:100]

    # Load and slice cluster assignments
    query_cluster_assignments = np.load(args.cluster_assignments_path)[:100]

    # Reshape cluster IDs to column vector and concatenate with query vectors
    ids_column = query_cluster_assignments.reshape(-1, 1)
    combined = np.hstack((ids_column, query_vectors))

    # Save to CSV
    fmt = ['%d'] + ['%.6f'] * query_vectors.shape[1]
    os.makedirs("tiptoe_baseline", exist_ok=True)
    np.savetxt("tiptoe_baseline/msmarco_query.csv", combined, delimiter=",", fmt=fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MSMARCO query CSV with cluster IDs and embeddings.")

    parser.add_argument('--query_vectors_path', required=True, help="Path to reduced query vectors .npy file")
    parser.add_argument('--cluster_assignments_path', required=True, help="Path to predicted cluster IDs .npy file")

    args = parser.parse_args()
    main(args)


#import numpy as np
#
#query_vectors = np.load('./eval_data/query_test_reduced.npy')
#query_vectors = query_vectors[:100]
#
## query_cluster_assignments = np.load('./cluster_model/predicted_cluster_ids_reduced.npy')
#query_cluster_assignments = np.load('./cluster_model/baseline_predicted_cluster_ids.npy')
#query_cluster_assignments = query_cluster_assignments[:100]
###
## Reshape ids to a column vector of shape (n, 1)
#ids_column = query_cluster_assignments.reshape(-1, 1)
#
## Concatenate along axis=1 (column-wise)
#combined = np.hstack((ids_column, query_vectors))
#
#fmt = ['%d'] + ['%.6f'] * 384
#np.savetxt("tiptoe_baseline/msmarco_query.csv", combined, delimiter=",", fmt=fmt)

