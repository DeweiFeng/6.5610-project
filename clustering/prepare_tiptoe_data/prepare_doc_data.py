import argparse
import numpy as np
import json
import os
from tqdm import tqdm

def export_cluster_data(cluster_ids, doc_vectors, output_dir, mapping_filename="reverse_index.json"):
    os.makedirs(output_dir, exist_ok=True)

    # Ensure cluster_ids is a flat 1D array of integers
    cluster_ids = np.ravel(cluster_ids).astype(int)

    reverse_mapping = {}

    # Group document indices by cluster
    clusters = {}
    for doc_id, cluster_id in enumerate(cluster_ids):
        clusters.setdefault(cluster_id, []).append(doc_id)

    for cluster_id, doc_indices in tqdm(clusters.items()):
        cluster_vectors = doc_vectors[doc_indices]

        # Save vectors using numpy's savetxt
        vector_filename = os.path.join(output_dir, f"msmarco_cluster_{cluster_id}.csv")
        np.savetxt(vector_filename, cluster_vectors, delimiter=",", fmt="%.4f")

        # Build reverse mapping for each document in this cluster
        for position_in_cluster, original_doc_id in enumerate(doc_indices):
            reverse_mapping[str(original_doc_id)] = [int(cluster_id), position_in_cluster]

        print(f"Saved cluster {cluster_id}: {len(doc_indices)} documents")

    # Save reverse mapping as JSON
    mapping_path = os.path.join(output_dir, mapping_filename)
    with open(mapping_path, "w") as f:
        json.dump(reverse_mapping, f, indent=2)

    print(f"Saved reverse mapping to {mapping_path}")


def main(args):
    cluster_ids = np.load(args.cluster_assignments_path)
    doc_vectors = np.load(args.doc_embeddings_path)
    output_dir = "tiptoe_" + args.output_dir_suffix
    export_cluster_data(cluster_ids, doc_vectors, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load cluster and document embeddings and export cluster data.")
    
    parser.add_argument('--output_dir_suffix', required=True, choices=['baseline', 'baseline_learned', 'graph'])
    parser.add_argument('--cluster_assignments_path', required=True, help="Path to cluster assignments .npy file")
    parser.add_argument('--doc_embeddings_path', required=True, help="Path to document embeddings .npy file")

    args = parser.parse_args()
    main(args)
