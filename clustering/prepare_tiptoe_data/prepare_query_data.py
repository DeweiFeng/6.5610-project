import numpy as np

query_vectors = np.load('./eval_data/query_test_reduced.npy')
query_vectors = query_vectors[:100]

# query_cluster_assignments = np.load('./cluster_model/predicted_cluster_ids_reduced.npy')
query_cluster_assignments = np.load('./cluster_model/baseline_predicted_cluster_ids.npy')
query_cluster_assignments = query_cluster_assignments[:100]

# Reshape ids to a column vector of shape (n, 1)
ids_column = query_cluster_assignments.reshape(-1, 1)

# Concatenate along axis=1 (column-wise)
combined = np.hstack((ids_column, query_vectors))

fmt = ['%d'] + ['%.6f'] * 384
np.savetxt("tiptoe_baseline/msmarco_query.csv", combined, delimiter=",", fmt=fmt)

