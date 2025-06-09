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
    output_dir = "tiptoe_" + args.output_dir_suffix
    fmt = ['%d'] + ['%.6f'] * query_vectors.shape[1]
    np.savetxt(os.path.join(output_dir, "msmarco_query.csv"), combined, delimiter=",", fmt=fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MSMARCO query CSV with cluster IDs and embeddings.")
    parser.add_argument('--output_dir_suffix', required=True, choices=['baseline', 'baseline_learned', 'graph'])
    parser.add_argument('--query_vectors_path', required=True, help="Path to reduced query vectors .npy file")
    parser.add_argument('--cluster_assignments_path', required=True, help="Path to predicted cluster IDs .npy file")

    args = parser.parse_args()
    main(args)


