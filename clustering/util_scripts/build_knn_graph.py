import numpy as np
import faiss
import time

# Parameters
num_embeddings = 10**6      # 1 million
embedding_dim = 384         # your actual dimension
k = 10                      # number of neighbors

# Load or generate your embeddings (ensure float32 dtype)
# Replace this with your actual embeddings
# embeddings = np.random.rand(num_embeddings, embedding_dim).astype('float32')

embeddings = np.load('../eval_data/msmarco_doc_embeddings_1M_norm.npy')

print(embeddings.shape)

# Initialize FAISS GPU resources
res = faiss.StandardGpuResources()

# Create a flat L2 index on GPU
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(embeddings)

# Run the kNN search
print("Running kNN search...")
start = time.time()
distances, indices = gpu_index.search(embeddings, k + 1)  # +1 to include self
end = time.time()
print(f"Search completed in {end - start:.2f} seconds")

# Build adjacency list (remove self-match)
adj_list = [neighbors[1:].tolist() for neighbors in indices]

print(len(adj_list))
# Print example
print(f"Neighbors of node 0: {adj_list[0]}")

np.save("adj_list.npy", np.array(adj_list, dtype=object))
