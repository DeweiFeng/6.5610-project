import argparse
import json
import os

def save_metadata(num_vectors, num_clusters, dim, output_path):
    metadata = {
        "num_vectors": num_vectors,
        "num_clusters": num_clusters,
        "dim": dim
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save metadata for clustering results.")
    parser.add_argument("--num_vectors", type=int, default=1000000, help="Number of vectors")
    parser.add_argument("--num_clusters", type=int, default=1000, help="Number of clusters")
    parser.add_argument("--dim", type=int, default=384, help="Dimension of vectors")
    parser.add_argument('--output_dir_suffix', required=True, choices=['baseline', 'baseline_learned', 'graph'])
    args = parser.parse_args()

    output_path = os.path.join("tiptoe_" + args.output_dir_suffix, 'msmarco_metadata.json')
    save_metadata(
        num_vectors=args.num_vectors,
        num_clusters=args.num_clusters,
        dim=args.dim,
        output_path=output_path
    )
