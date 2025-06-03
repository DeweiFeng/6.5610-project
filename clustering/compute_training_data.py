import numpy as np
import os
# Of shape (# documents x 1)
doc_cluster_assignments = np.load('./baseline_data/cluster_assignments.npy')
doc_to_cluster = {i: cluster for i, cluster in enumerate(doc_cluster_assignments)}
ground_truth_test = np.load('./baseline_data/ground_truth_1M_k10.npy')

ground_truth = ground_truth_test[:, 0]

print('***')
print(ground_truth.shape)

ground_truth = ground_truth.flatten()

query_cluster_assignments = []

for i in range(ground_truth.shape[0]):
    true_doc = ground_truth[i]
    true_cluster = doc_to_cluster[true_doc]
    query_cluster_assignments.append(true_cluster)

query_cluster_assignments = np.array(query_cluster_assignments)

# np.save('query_cluster_assignments_full', query_cluster_assignments)

y_train = query_cluster_assignments[:500000]

y_val = query_cluster_assignments[500000:501000]

num_classes = np.max(query_cluster_assignments) + 1
print('***')
print(num_classes)

print(y_train.shape)
print(y_val.shape)
np.save('./cluster_model/y_train', y_train)
np.save('./cluster_model/y_val', y_val)

query_vectors = np.load('./baseline_data/msmarco_query_embeddings_norm.npy')

x_train = query_vectors[:500000]
x_val = query_vectors[500000:501000]

np.save('./cluster_model/x_train', x_train)
np.save('./cluster_model/x_val', x_val)

print(x_train.shape)
print(x_val.shape)

gnd_test = ground_truth_test[500000:501000]

print('***')
print(gnd_test.shape)

os.makedirs('eval_data', exist_ok=True)
np.save('./eval_data/query_test_reduced', x_val)
np.save('./eval_data/ground_truth_test_k10', gnd_test)
