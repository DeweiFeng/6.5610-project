import os
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from beir import util  # This is the correct import for downloading datasets

# Step 2: Load the MS MARCO dataset using BEIR
dataset = "msmarco"
data_path = os.path.join("datasets", dataset)

# Download and extract the dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, data_path)

# Load the corpus, queries, and qrels
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

# Step 3: Initialize the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to encode texts in batches
def encode_texts(texts, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Step 4: Prepare and encode documents
doc_ids = list(corpus.keys())
doc_texts = [corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "") for doc_id in doc_ids]
doc_embeddings = encode_texts(doc_texts)

# Step 5: Prepare and encode queries
query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]
query_embeddings = encode_texts(query_texts)


np.save("msmarco_doc_embeddings.npy", doc_embeddings)
np.save("msmarco_query_embeddings.npy", query_embeddings)

