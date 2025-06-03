import numpy as np
import faiss
import argparse
import os
import time
from tqdm import tqdm
from collections import defaultdict
import json
import csv


def load_vectors(filename, n, dim): 
    ext = filename.split('.')[-1]

    print("extension", ext)
    
    if ext == 'npy':
        # Load the vectors from a numpy file
        vectors = np.load(filename) 
        if vectors.shape[0] < n:
            raise ValueError("The number of vectors in the file is less than the requested number") 
        vectors = vectors[:n].astype(np.float32)
    else: 
        raise ValueError("Unknown file extension")
    
    return vectors


def clustering(vectors, n_clusters):

    print("Clustering", vectors.shape[0], "vectors into", n_clusters, "clusters")

    start = time.time()
    
    # verify the vector is of type float32
    if vectors.dtype != np.float32:
        raise ValueError("The vectors must be of type float32")

    kmeans = faiss.Kmeans(vectors.shape[1], int(n_clusters), niter=1, verbose=True, spherical=False)
    kmeans.train(vectors)
    centroids = kmeans.centroids

    np.save('baseline_centroids_v2', centroids)
    _, I = kmeans.index.search(vectors, 1) # the I contains the closest centroid for each vector
    I = I.flatten()
    
    if centroids.dtype != np.float32:
        raise ValueError("The centroids must be of type float32")

    np.save('baseline_cluster_assignments_v2', I)
    end = time.time()
    print("Clustering time:", end - start)

    return centroids, I


class ClusterSearchIndex:
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.n_clusters = np.sqrt(n).astype(int)
        self.centroids = np.load('baseline_data/centroids.npy')
        self.I = np.load('baseline_data/cluster_assignments.npy')
        # self.centroids = np.load('learned_data/centroids_reduced.npy')
        self.I = self.I.flatten()
        # self.centroids = None
        # self.I = None		

    def build_index(self, vectors):
        if self.centroids is None or self.I is None:
            self.centroids, self.I = clustering(vectors, self.n_clusters)

        self.sorted_indices = np.argsort(self.I) # are the labels from 0-n_clusters-1?
        self.sorted_labels = self.I[self.sorted_indices] # the labels of the sorted indices
        self.sorted_vectors = vectors[self.sorted_indices].copy() # make it contiguous

        size_of_each_cluster = np.bincount(self.sorted_labels)
        # print the min and the max of the size of each cluster
        print("Min size of the cluster", np.min(size_of_each_cluster))
        print("Max size of the cluster", np.max(size_of_each_cluster))
        self.offset_of_each_cluster = np.cumsum(size_of_each_cluster)
        self.offset_of_each_cluster = np.concatenate(([0], self.offset_of_each_cluster))
        # now the vectors in the i-th cluster are from offset_of_each_cluster[i] to offset_of_each_cluster[i+1]

    def search(self, query, k):
        # step 1: find the closest centroid 
        dist_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        cluster_id = np.argmin(dist_to_centroids)
        # step 2: find the k-nearest neighbors in the cluster
        cluster_start = self.offset_of_each_cluster[cluster_id]
        cluster_end = self.offset_of_each_cluster[cluster_id + 1]
        cluster_vectors = self.sorted_vectors[cluster_start:cluster_end]
        distance_to_vectors = np.linalg.norm(cluster_vectors - query, axis=1)
        top_k_offset = np.argsort(distance_to_vectors)
        if len(top_k_offset) < k:
            # append zeros to the top_k_offset
            top_k_offset = np.concatenate((top_k_offset, np.zeros(k - len(top_k_offset), dtype=np.int32)))
        else:
            top_k_offset = top_k_offset[:k]
        top_k_idx = self.sorted_indices[cluster_start + top_k_offset]

        return top_k_idx, cluster_id
    

    def search_in_cluster(self, query, cluster_id, k):
	# step 2: find the k-nearest neighbors in the cluster
        cluster_start = self.offset_of_each_cluster[cluster_id]
        cluster_end = self.offset_of_each_cluster[cluster_id + 1]
        cluster_vectors = self.sorted_vectors[cluster_start:cluster_end]
        distance_to_vectors = np.linalg.norm(cluster_vectors - query, axis=1)
        top_k_offset = np.argsort(distance_to_vectors)
        if len(top_k_offset) < k:
            # append zeros to the top_k_offset
            top_k_offset = np.concatenate((top_k_offset, np.zeros(k - len(top_k_offset), dtype=np.int32)))
        else:
            top_k_offset = top_k_offset[:k]
        top_k_idx = self.sorted_indices[cluster_start + top_k_offset]

        return top_k_idx, cluster_id


def calculate_recall(answers, gnd, k):
    # answers and gnd are both 2d np.array
    n = answers.shape[0]
    answers_focus = answers[:, :k]
    gnd_focus = gnd[:n, :k] # gnd[:n, :k]
    # gnd_focus = gnd[:n, :k]
    recall = np.zeros(n)
    for i in range(n):
        # we need to find the number of common elements between the two arrays
        # we can use the numpy intersect1d function
        recall[i] = len(np.intersect1d(answers_focus[i], gnd_focus[i], assume_unique=True)) / k
    return np.mean(recall)

def calculate_mrr(answers, gnd, k):
    n = answers.shape[0]
    answers_focus = answers[:, :k]
    gnd_focus = gnd[:n, :k]
    mrr = np.zeros(n)
    for i in range(n):
        relevant = gnd_focus[i][0] # set(gnd_focus[i])
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(answers_focus[i]):
            if doc_id == relevant:
                reciprocal_rank = 1.0 / (rank + 1)
                break
        mrr[i] = reciprocal_rank

    return np.mean(mrr)


def main():
    parser = argparse.ArgumentParser(description="tiptoe-style cluster search")
    parser.add_argument("-n", type=int, help="number of vectors")
    parser.add_argument("-dim", type=int, help="dimension of the vectors")
    parser.add_argument("-k", type=int, help="number of nearest neighbors")
    parser.add_argument("-q", type=int, help="number of queries")
    parser.add_argument("-input", type=str, help="input vector file")
    parser.add_argument("-query", type=str, help="query vector file")
    parser.add_argument("-output", type=str, help="output file")
    parser.add_argument("-report", type=str, help="report file")
    parser.add_argument("-gnd", type=str, help="ground truth file")
    parser.add_argument("-mode", choices=["learned", "baseline"])
    args = parser.parse_args()

    n = args.n
    dim = args.dim
    q = args.q
    input_file = args.input
    query_file = args.query
    output_file = args.output
    gnd_file = args.gnd

    if input_file is None or query_file is None:
        raise ValueError("The input and query files must be provided")

    print("Loading the vectors from the input file", input_file)
    vectors = load_vectors(input_file, n, dim)

    index = ClusterSearchIndex(n, dim)
    index.build_index(vectors)

    print("Loading the query vectors from the query file", query_file)
    queries = load_vectors(query_file, q, dim)
    queries = queries[:q]

    print("Performing the search")
    start = time.time()
    answers = np.zeros((queries.shape[0], args.k), dtype=np.int32)
    
    if args.mode == "learned":
        query_cluster_ids = np.load('cluster_model/predicted_cluster_ids_reduced.npy')
        for i in range(queries.shape[0]):
            answers[i], cluster_id = index.search_in_cluster(queries[i], query_cluster_ids[i], args.k)
    else:
        for i in range(queries.shape[0]):
            answers[i], cluster_id = index.search(queries[i], args.k)
    
    end = time.time()
    print("Search time:", end - start)
    print("Average search time:", (end - start) / queries.shape[0]) 

    gnd = load_vectors(gnd_file, q, args.k)
    recall = calculate_recall(answers, gnd, args.k)
    mrr = calculate_mrr(answers, gnd, args.k)
    print("Recall:", recall)
    print("MRR:", mrr)

if __name__=='__main__':
    main()
