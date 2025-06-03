import numpy as np
import faiss
import argparse

def k_means(doc_vectors, n_clusters, flag_spherical=False, gpu_device=0, sample_size=50000):
    d = doc_vectors.shape[1]
    print(f"Embedding dimension: {d}, total docs: {doc_vectors.shape[0]}")

    # Sample a subset for training (optional but recommended for large datasets)
    train_sample = min(sample_size, doc_vectors.shape[0])
    sample_indices = np.random.choice(doc_vectors.shape[0], train_sample, replace=False)
    train_data = doc_vectors[sample_indices].astype(np.float32)

    print(f"Training k-means on {train_data.shape[0]} samples with {n_clusters} clusters...")

    # Setup GPU resources
    res = faiss.StandardGpuResources()
    clustering = faiss.Clustering(d, n_clusters)
    clustering.niter = 2
    clustering.max_points_per_centroid = 10000000  # helps with large datasets

    # Optional: Spherical (normalize vectors to unit norm before clustering)
    if flag_spherical:
        faiss.normalize_L2(train_data)

    # Use a flat (brute-force) index on the GPU for clustering
    config = faiss.GpuIndexFlatConfig()
    config.device = gpu_device

    index_flat = faiss.GpuIndexFlatL2(res, d, config)
    clustering.train(train_data, index_flat)

    print("K-means training completed.")

    # Compute final assignments for all documents
    full_data = doc_vectors.astype(np.float32)
    if flag_spherical:
        faiss.normalize_L2(full_data)

    # Use a GPU index to assign clusters
    index_assign = faiss.GpuIndexFlatL2(res, d, config)
    centroid_matrix = faiss.vector_to_array(clustering.centroids).reshape(n_clusters, d)
    index_assign.add(centroid_matrix)
    distances, assignments = index_assign.search(full_data, 1)
    centroids_np = faiss.vector_to_array(clustering.centroids).reshape(n_clusters, -1)
    return centroids_np, assignments


def main(input_path, n_clusters, gpu_device):
    # Load vectors
    doc_vectors = np.load(input_path)
    n_clusters = 1000
    centroids, cluster_assignments = k_means(doc_vectors, n_clusters, flag_spherical=True, gpu_device=1)
    return centroids, cluster_assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run K-means clustering on document embeddings using FAISS.")
    parser.add_argument('--input', type=str, required=True, help='Path to input normalized document embeddings (npy)')
    parser.add_argument('--output_centroids', type=str, required=True, help='Path to save output centroids (npy)')
    parser.add_argument('--output_assignments', type=str, required=True, help='Path to save cluster assignments (npy)')
    parser.add_argument('--n_clusters', type=int, default=1000, help='Number of clusters to compute')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use for clustering')

    args = parser.parse_args()

    centroids, cluster_assignments = main(input_path=args.input, 
					  n_clusters=args.n_clusters,
					  gpu_device=args.gpu)

    np.save(args.output_centroids, centroids)
    np.save(args.output_assignments, cluster_assignments)


