import os
import sys
import numpy as np
from sklearn.cluster import KMeans

def generate_test_files(num_vectors, dim, num_clusters, preamble, num_queries=10):
    # get the dir of preamble and create it if it does not exist
    os.makedirs(os.path.dirname(preamble), exist_ok=True)
    all_vectors = np.random.uniform(-1, 1, (num_vectors, dim))
    queries = np.random.uniform(-1, 1, (num_queries, dim))

    # normalize all vectors and queries such that l2 norm is 1, so that cosine similarity is the same as dot product
    all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_vectors)
    clusters = [[] for _ in range(num_clusters)]
    for i in range(num_vectors):
        clusters[kmeans.labels_[i]].append(all_vectors[i])

    # get the cluster ids of all query vectors
    queries_labels = kmeans.predict(queries)

    for i in range(num_clusters):
        with open(f"{preamble}_cluster_{i}.csv", "w") as f:
            np.savetxt(f, clusters[i], delimiter=",", fmt="%s")
    
    # save queries to a csv file, each row is cluster id followed by the query vector
    with open(f"{preamble}_queries.csv", "w") as f:
        for i in range(num_queries):
            f.write(f"{queries_labels[i]},")
            np.savetxt(f, queries[i].reshape(1, -1), delimiter=",", fmt="%s")

    with open(f"{preamble}_metadata.json", "w") as f:
        f.write("{\n")
        f.write(f'  "num_vectors": {num_vectors},\n')
        f.write(f'  "num_clusters": {num_clusters},\n')
        f.write(f'  "dim": {dim}\n')
        f.write("}\n")

# if __name__ == "__main__":
#     generate_test_files(100, 10, 5, "./test_data/test")

# four arguments from command line
if __name__ == "__main__":
    num_vectors = int(sys.argv[1])
    dim = int(sys.argv[2])
    num_clusters = int(sys.argv[3])
    preamble = sys.argv[4]
    generate_test_files(num_vectors, dim, num_clusters, preamble)