import numpy as np

# Replace 'your_file.txt' with the path to your txt file
arr = np.loadtxt('../eval_data/knn_graph_for_metis.txt.part.1000', dtype=int)
np.save('../eval_data/cluster_assignments', arr)
