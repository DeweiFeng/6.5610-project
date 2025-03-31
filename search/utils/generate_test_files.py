import sys
import numpy as np
from sklearn.cluster import KMeans

def generate_test_files(num_vectors, dim, num_clusters, preamble, precision=5):
    all_vectors = np.random.uniform(-1, 1, (num_vectors, dim))
    np.savetxt(f"{preamble}_all_float.csv", all_vectors, delimiter=",", fmt="%s")

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_vectors)
    cluster_centers = kmeans.cluster_centers_
    clusters = [[] for _ in range(num_clusters)]
    for i in range(num_vectors):
        clusters[kmeans.labels_[i]].append(all_vectors[i])
    # max_cluster_size = max(len(cluster) for cluster in clusters)
    # # pad all clusters to max_cluster_size with zeros
    # for i in range(num_clusters):
    #     if len(clusters[i]) < max_cluster_size:
    #         clusters[i].extend([[0] * dim] * (max_cluster_size - len(clusters[i])))
    clusters = [np.array(cluster) for cluster in clusters]
    cluster_centers = np.array(cluster_centers)

    scale = 1 << precision
    clusters_int = [np.round(cluster * scale) for cluster in clusters]
    cluster_centers_int = np.round(cluster_centers * scale)

    for i in range(num_clusters):
        with open(f"{preamble}_cluster_{i}.csv", "w") as f:
            f.write(f"{len(clusters_int[i])}\n{dim}\n{precision}\n")
            np.savetxt(f, clusters_int[i], delimiter=",", fmt="%d")
        with open(f"{preamble}_cluster_{i}_float.csv", "w") as f:
            f.write(f"{len(clusters[i])},{dim}\n{precision}\n")
            np.savetxt(f, clusters[i], delimiter=",", fmt="%s")
    
    with open(f"{preamble}_centers.csv", "w") as f:
        f.write(f"{num_clusters}\n{dim}\n{precision}\n")
        np.savetxt(f, cluster_centers_int, delimiter=",", fmt="%d")
    
    with open(f"{preamble}_centers_float.csv", "w") as f:
        f.write(f"{num_clusters}\n{dim}\n""{precision}\n")
        np.savetxt(f, cluster_centers, delimiter=",", fmt="%s")

    with open(f"{preamble}_metadata.json", "w") as f:
        f.write("{\n")
        f.write(f'  "num_vectors": {num_vectors},\n')
        f.write(f'  "num_clusters": {num_clusters},\n')
        # f.write(f'  "cluster_size": {max_cluster_size},\n')
        f.write(f'  "dim": {dim},\n')
        f.write(f'  "prec_bits": {precision}\n')
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