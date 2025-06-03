import json

def save_metadata(num_vectors, num_clusters, dim, output_path="tiptoe_baseline/msmarco_metadata.json"):
    metadata = {
        "num_vectors": num_vectors,
        "num_clusters": num_clusters,
        "dim": dim
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_path}")

# Example usage:
save_metadata(1000000, 1000, 384)
