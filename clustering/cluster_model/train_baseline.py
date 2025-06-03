import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


centroids = np.load('../baseline_data/centroids.npy')

test_queries = np.load('../eval_data/query_test_reduced.npy')

predicted_cluster_ids = []
for query in test_queries:
    dist_to_centroids = np.linalg.norm(centroids - query, axis=1)
    cluster_id = np.argmin(dist_to_centroids)
    predicted_cluster_ids.append(cluster_id)

predicted_cluster_ids = np.array(predicted_cluster_ids)
np.save('baseline_predicted_cluster_ids', predicted_cluster_ids)


