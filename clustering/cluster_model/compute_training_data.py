import numpy as np
import argparse
import os

def main(args):
    # Load inputs
    doc_cluster_assignments = np.load(args.cluster_assignments_path)
    ground_truth_test = np.load(args.ground_truth_path)
    query_vectors = np.load(args.query_vectors_path)

    # Build doc to cluster map
    doc_to_cluster = {i: cluster for i, cluster in enumerate(doc_cluster_assignments)}
    ground_truth = ground_truth_test[:, 0].flatten()

    # Assign clusters to queries
    query_cluster_assignments = np.array([doc_to_cluster[doc] for doc in ground_truth])

    # Train/val split
    y_train = query_cluster_assignments[:500000]
    y_val = query_cluster_assignments[500000:501000]

    np.save('y_train', y_train)
    np.save('y_val', y_val)

    # Split query vectors
    x_train = query_vectors[:500000]
    x_val = query_vectors[500000:501000]

    np.save('x_train', x_train)
    np.save('x_val', x_val)

    # Reduce ground truth test set
    gnd_test = ground_truth_test[500000:501000]

    np.save(args.query_test_output, x_val)
    np.save(args.ground_truth_test_output, gnd_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split clustering data.")
    
    parser.add_argument('--cluster_assignments_path', required=True, help="Path to cluster assignments .npy file")
    parser.add_argument('--ground_truth_path', required=True, help="Path to ground truth .npy file")
    parser.add_argument('--query_vectors_path', required=True, help="Path to query embeddings .npy file") 
    parser.add_argument('--query_test_output', default='../eval_data/query_test_reduced.npy', help="Output path for reduced test queries")
    parser.add_argument('--ground_truth_test_output', default='../eval_data/ground_truth_test_k10.npy', help="Output path for reduced ground truth")

    args = parser.parse_args()
    main(args)

