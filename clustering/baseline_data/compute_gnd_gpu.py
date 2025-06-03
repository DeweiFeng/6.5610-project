import argparse
import faiss
import numpy as np

def main(doc_embeddings_path, query_embeddings_path, output_index_path, 
         output_doc_path, output_query_path, d=384, k=10, gpu_id=1):
    # Load data
    xb = np.load(doc_embeddings_path)
    xq = np.load(query_embeddings_path)
    
    # Downsample the documents to 1M vectors

    xb = xb[:1000000]

    # Normalize vectors
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)

    # Set up CPU index
    cpu_index = faiss.IndexFlatL2(d)

    # Move to GPU
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, cpu_index)

    # Add vectors to GPU index
    gpu_index.add(xb)

    # Search
    _, I = gpu_index.search(xq, k)

    # Save results
    np.save(output_index_path, I)
    np.save(output_doc_path, xb)
    np.save(output_query_path, xq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FAISS GPU Search with Normalized Embeddings")
    parser.add_argument('--doc', type=str, required=True, help='Path to document embeddings (npy)')
    parser.add_argument('--query', type=str, required=True, help='Path to query embeddings (npy)')
    parser.add_argument('--output_index', type=str, required=True, help='Path to save index result (npy)')
    parser.add_argument('--output_doc', type=str, required=True, help='Path to save normalized document embeddings (npy)')
    parser.add_argument('--output_query', type=str, required=True, help='Path to save normalized query embeddings (npy)')
    parser.add_argument('--dimension', type=int, default=384, help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use')

    args = parser.parse_args()
    
    xb = np.load(args.doc)
    xq = np.load(args.query)

    main(
        doc_embeddings_path=args.doc,
        query_embeddings_path=args.query,
        output_index_path=args.output_index,
        output_doc_path=args.output_doc,
        output_query_path=args.output_query,
        d=args.dimension,
        k=args.k,
        gpu_id=args.gpu
    )

