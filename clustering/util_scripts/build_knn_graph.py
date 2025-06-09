import faiss
import os
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, triu

def create_undirected_csr_from_faiss(distances, indices, num_nodes, k):
    """
    Converts FAISS kNN results into a weighted, undirected SciPy CSR matrix.
    (This function from the previous response is correct)
    """
    print("\n--- Step 1: Creating a Weighted, Undirected CSR Graph ---")
    
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]
    MAX_WEIGHT = 1000
    max_dist = np.max(neighbor_distances)
    if max_dist == 0: max_dist = 1.0
    scaled_weights = MAX_WEIGHT * (1.0 - neighbor_distances / max_dist)
    int_weights = np.maximum(1, scaled_weights.astype(int))

    source_nodes = np.arange(num_nodes).repeat(k)
    target_nodes = neighbor_indices.flatten()
    weights_flat = int_weights.flatten()

    directed_graph = coo_matrix(
        (weights_flat, (source_nodes, target_nodes)),
        shape=(num_nodes, num_nodes)
    )
    undirected_graph = directed_graph.maximum(directed_graph.T)
    graph_csr = undirected_graph.tocsr()
    
    print("CSR graph created successfully.")
    print(f" - Graph has {graph_csr.shape[0]} nodes.")
    print(f" - Graph has {graph_csr.nnz} non-zero entries before cleaning.")
    
    return graph_csr


def save_csr_to_metis_format_corrected(graph: csr_matrix, filename: str):
    """
    Saves a weighted SciPy CSR matrix to the METIS graph file format.
    
    This version includes crucial corrections:
    1. Removes self-loops.
    2. Uses a robust method to count the number of unique undirected edges.
    """
    print(f"\n--- Step 2: Saving Graph to METIS Format at '{filename}' (Corrected) ---")
    
    # --- Bug Fix Start ---

    # 1. Explicitly remove self-loops (edges from a node to itself)
    graph.setdiag(0)
    graph.eliminate_zeros()
    
    num_nodes = graph.shape[0]

    # 2. Get a robust count of unique edges.
    # The number of unique edges in an undirected graph is the number of non-zero
    # elements in its upper (or lower) triangle.
    num_edges = triu(graph, k=1).nnz
    
    print(f"Graph cleaned: {graph.nnz} non-zero entries remain after removing self-loops.")
    print(f"Robust edge count for METIS header: {num_edges}")

    # --- Bug Fix End ---

    file_path = os.path.join('../eval_data', filename)

    with open(file_path, 'w') as f:
        # Write header: num_nodes, num_edges, format_code
        header = f"{num_nodes} {num_edges} 001\n"
        f.write(header)
        
        for i in range(num_nodes):
            start, end = graph.indptr[i], graph.indptr[i+1]
            neighbors = graph.indices[start:end]
            edge_weights = graph.data[start:end]
            
            # Add 1 to neighbor index for 1-based format
            line_parts = [f"{neighbor + 1} {weight}" for neighbor, weight in zip(neighbors, edge_weights)]
            f.write(" ".join(line_parts) + "\n")
            
    print("METIS file saved successfully.")
    print(f" - To use, run: gpmetis {filename} <num_partitions>")


if __name__ == "__main__":
    # --- FAISS Setup and Search ---
    
    # Parameters
    num_embeddings = 10**6
    embedding_dim = 384
    k = 10
    
    print("--- Running FAISS kNN Search ---")
    
    # Generate random placeholder data
	
    embeddings = np.load('../eval_data/msmarco_doc_embeddings_1M_norm.npy') 
    
    # Initialize FAISS GPU resources
    res = faiss.StandardGpuResources()

    # Create a flat L2 index on GPU
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
    gpu_index = faiss.index_cpu_to_gpu(res, 1, index)
    gpu_index.add(embeddings)

    # Run the kNN search
    print("Running kNN search...")
    start = time.time()
    distances, indices = gpu_index.search(embeddings, k + 1)  # +1 to include self
    end = time.time()
    print(f"Search completed in {end - start:.2f} seconds") 
    # --- Main Workflow ---
    
    # 1. Convert FAISS output to CSR
    graph_csr_weighted = create_undirected_csr_from_faiss(distances, indices, num_embeddings, k)

    # 2. Save the CSR matrix using the CORRECTED function
    metis_filename = "knn_graph_for_metis.txt"
    save_csr_to_metis_format_corrected(graph_csr_weighted, metis_filename)
