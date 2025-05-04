"""
    **A Learning-to-Rank Formulation of Clustering-Based Approximate Nearest Neighbor Search**

    *project*:
     mips-learnt-ivf

    *authors*:
     Thomas Vecchiato, Claudio Lucchese, Franco Maria Nardini, Sebastian Bruch

    *name file*:
     main_mips.py

    *version file*:
     1.0

    *description*:
     A critical piece of the modern information retrieval puzzle is approximate nearest neighbor search.
     Its objective is to return a set of k data points that are closest to a query point, with its accuracy measured by
     the proportion of exact nearest neighbors captured in the returned set. One popular approach to this question
     is clustering: The indexing algorithm partitions data points into non-overlapping subsets and represents each
     partition by a point such as its centroid. The query processing algorithm first identifies the nearest
     clusters — a process known as routing — then performs a nearest neighbor search over those clusters only.
     In this work, we make a simple observation: The routing function solves a ranking problem.
     Its quality can therefore be assessed with a ranking metric, making the function amenable to learning-to-rank.
     Interestingly, ground-truth is often freely available: Given a query distribution in a top-k configuration,
     the ground-truth is the set of clusters that contain the exact top-k vectors. We develop this insight and apply
     it to Maximum Inner Product Search (MIPS). As we demonstrate empirically on various datasets,
     learning a simple linear function consistently improves the accuracy of clustering-based MIPS.

    *run commands*:

     1. **hdf5 format**:
     python main_mips.py --name_dataset ... --name_embedding ... --format_file hdf5 --dataset ... --algorithm ...
     --nclusters ... --top_k ... --test_split_percent ... --split_seed ... --ells ... --learner_nunits ...
     --learner_nepochs ... --compute_clusters ...

     2. **npy format**:
     python main_mips.py --name_dataset ... --name_embedding ... --format_file npy --dataset_docs ...
     --dataset_queries ... --dataset_neighbors ... --algorithm ... --nclusters ... --top_k ... --test_split_percent ...
     --split_seed ... --ells ... --learner_nunits ... --learner_nepochs ... --compute_clusters ...

"""

import numpy as np
import h5py
from absl import app, flags
import time
from tabulate import tabulate
from clustering import kmeans, k_random, auxiliary, linearlearner
from tqdm import tqdm
# from clustering import torch_learner

# names of the algorithms
AlgorithmRandom = 'random'
AlgorithmKMeans = 'kmeans'
AlgorithmSphericalKmeans = 'kmeans-spherical'
AlgorithmLinearLearner = 'linear-learner'

# name of the dataset and embedding
flags.DEFINE_string('name_dataset', None, 'Name of the dataset.')
flags.DEFINE_string('name_embedding', None, 'Name of the embedding.')

# decide the file format to import
flags.DEFINE_string('format_file', None, 'hdf5 - for the hdf5 file; npy - for the npy files.')

# dataset for hdf5
flags.DEFINE_string('dataset', None, 'Path to the dataset in hdf5 format.')

flags.DEFINE_string('documents_key', 'documents', 'Dataset key for document vectors.')
flags.DEFINE_string('train_queries_key', 'train_queries', 'Dataset key for train queries.')
flags.DEFINE_string('valid_queries_key', 'valid_queries', 'Dataset key for validation queries.')
flags.DEFINE_string('test_queries_key', 'test_queries', 'Dataset key for test queries.')
flags.DEFINE_string('train_neighbors_key', 'train_neighbors', 'Dataset key for train neighbors.')
flags.DEFINE_string('valid_neighbors_key', 'valid_neighbors', 'Dataset key for validation neighbors.')
flags.DEFINE_string('test_neighbors_key', 'test_neighbors', 'Dataset key for test neighbors.')

# docs, queries and neighbors for npy
flags.DEFINE_string('dataset_docs', None, 'Path to the dataset-docs in npy format.')
flags.DEFINE_string('dataset_queries', None, 'Path to the dataset-queries in npy format.')
flags.DEFINE_string('dataset_neighbors', None, 'Path to the dataset-neighbors in npy format.')

# setting environment
flags.DEFINE_float('test_split_percent', 20, 'Percentage of data points in the test set.')
flags.DEFINE_integer('split_seed', 42, 'Seed used when forming train-test splits.')

# linear-learner
flags.DEFINE_integer('learner_nunits', 0, 'Number of hidden units used by the linear-learner, with 0 we drop'
                                             'the hidden layer.')
flags.DEFINE_integer('learner_nepochs', 100, 'Number of epochs used by the linear-learner.')

# algorithm method
flags.DEFINE_enum('algorithm', AlgorithmKMeans,
                  [AlgorithmRandom,
                   AlgorithmKMeans,
                   AlgorithmSphericalKmeans,
                   AlgorithmLinearLearner],
                  'Indexing algorithm.')

flags.DEFINE_integer('nclusters', 1000, 'When `algorithm` is KMeans-based: Number of clusters.')

# multi-probing, set the probes
flags.DEFINE_list('ells', [1],
                  'Minimum number of documents to examine.')

# top-k docs
flags.DEFINE_integer('top_k', 1, 'Top-k documents to retrieve per query.')

# flag to skip the clustering algorithm if already computed
flags.DEFINE_integer('compute_clusters', 0, '0 - perform clustering algorithm; '
                                            '1 - take the results already computed.')

FLAGS = flags.FLAGS


def get_final_results(name_method, centroids, x_test, y_test, top_k, clusters_top_k_test=None, gpu_flag=True):
    """
    Computes the final results, where we have the accuracy of given centroids.

    :param name_method: Name of the method that generated the centroids under consideration.
    :param centroids: The centroids.
    :param x_test, y_test: The test set.
    :param top_k: The number of top documents.
    :param clusters_top_k_test: The clusters where the top documents for each query are located.
    :param gpu_flag: Flag that indicates whether to run the code using the GPU (True) or not (False).
    """

    # compute the score for each query and centroid
    print(name_method, end=' ')
    print('- run prediction with centroids...', end=' ')
    pred = auxiliary.scores_queries_centroids(centroids, x_test, gpu_flag=gpu_flag)
    print('end, shape: ', pred.shape)

    # save scores computed
    print('Saving results: score for each query and centroid.')
    np.save('./ells_stat_sig/' + name_method + '_' + FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' +
            FLAGS.algorithm + '_ells_stat_sig.npy', pred)

    # computation of the final scores
    results_ells = []
    if top_k > 1:
        for threshold in tqdm(FLAGS.ells):
            k = int(threshold)
            one_pred = auxiliary.computation_top_k_clusters(k, FLAGS.nclusters, pred)
            res = auxiliary.evaluate_ell_top_k(one_pred, clusters_top_k_test, top_k)
            results_ells.append(res)
            print('k = {0}: {1}'.format(k, res))
    else:
        for threshold in tqdm(FLAGS.ells):
            k = int(threshold)
            one_pred = auxiliary.computation_top_k_clusters(k, FLAGS.nclusters, pred)
            res = auxiliary.evaluate_ell_top_one(one_pred, y_test)
            results_ells.append(res)
            print('k = {0}: {1}'.format(k, res))

    # print the final results
    table = ([['n_k', 'acc']] + [[FLAGS.ells[i_c], results_ells[i_c]] for i_c in range(len(FLAGS.ells))])
    print(tabulate(table, headers='firstrow', tablefmt='psql'))

    # save the results
    file_result = open('./results/' + name_method + '_' + FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' +
                       FLAGS.algorithm + str(top_k) + '_results.txt', 'w')
    file_result.write(tabulate(table, headers='firstrow', tablefmt='psql'))
    file_result.close()
    print('Results saved.')


def main(_):
    """
    Main function of our algorithm, where all methods to obtain the final results are invoked.
    """
    start = time.time()

    documents = None
    queries = None
    neighbors = None


    documents = np.load(FLAGS.dataset_docs)
    queries = np.load(FLAGS.dataset_queries)
    neighbors = np.load(FLAGS.dataset_neighbors)
        
    assert len(queries) == len(neighbors)

    # run the clustering algorithm or import the clusters already computed
    print('Running the clustering algorithm or importing the clusters already computed.')

    centroids = None
    label_clustering = None

    # compute centroids and labels
    if FLAGS.compute_clusters == 1:

        # (standard or spherical) kmeans algorithm
        if FLAGS.algorithm in [AlgorithmKMeans, AlgorithmSphericalKmeans]:
            spherical = FLAGS.algorithm == AlgorithmSphericalKmeans
            centroids, label_clustering = kmeans.k_means(doc_vectors=documents,
                                                         n_clusters=FLAGS.nclusters,
                                                         flag_spherical=spherical)
            print(f'Obtained centroids with shape: {centroids.shape}')

        # shallow kmeans algorithm
        elif FLAGS.algorithm == AlgorithmRandom:
            centroids, label_clustering = k_random.random_clustering(doc_vectors=documents,
                                                                     n_clusters=FLAGS.nclusters)
            print(f'Obtained centroids with shape: {centroids.shape}')

        # save centroids and label_clustering
        centroids_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_centroids.npy'
        label_clustering_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_label_clustering.npy'

        print('Saving clusters got.')
        np.save(centroids_file, centroids)
        np.save(label_clustering_file, label_clustering)

   # load centroids and labels
    else:
        centroids_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_centroids.npy'
        label_clustering_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_label_clustering.npy'

        centroids = np.load(centroids_file)
        label_clustering = np.load(label_clustering_file)

    # data preparation
    print('Data preparation.')
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None
    clusters_top_k_test = None


    label_data = auxiliary.query_true_label(FLAGS.nclusters, label_clustering, neighbors)
    # partitioning = auxiliary.train_test_val(queries, label_data, size_split=FLAGS.test_split_percent/100)
    
    x_train = queries[:6400]
    y_train = label_data[:6400]

    x_val = queries[6400:8000]
    y_val = label_data[6400:8000]

    x_test = queries[8000:8100]
    y_test = label_data[8000:8100]

    #x_train = partitioning[0]
    #y_train = partitioning[1]
    #x_val = partitioning[2]
    #y_val = partitioning[3]
    #x_test = partitioning[4]
    #y_test = partitioning[5]
    
    np.save('data/sift/test_queries', x_test)
    np.save(FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_y_test.npy', y_test)

    train = True
    if train:
        # training linear-learner
        print('Linear Learner.')
        new_centroids = linearlearner.run_linear_learner(x_train=x_train, y_train=y_train,
                                                     x_val=x_val, y_val=y_val,
                                                     train_queries=queries,
                                                     n_clusters=FLAGS.nclusters,
                                                     n_epochs=FLAGS.learner_nepochs,
                                                     n_units=FLAGS.learner_nunits)

        print(f'Obtained centroids with shape: {new_centroids.shape}')
    
        np.save('new_centroids', new_centroids)

    else:
        new_centroids = np.load("new_centroids.npy")
    
    # results: baseline
    get_final_results('baseline', centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)
    print("baseline")
    # results: linear-learner
    get_final_results('linearlearner', new_centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)

    end = time.time()
    print(f'Done in {end - start} seconds.')
    
    from collections import defaultdict
    cluster_to_docs = defaultdict(list)
    reverse_doc_map = {}
    
    cluster_assignments = np.load('sift_euclidean_kmeans_label_clustering.npy')
    doc_vectors = np.load('data/sift/documents.npy')
    print(cluster_assignments.shape)
    for doc_id, cluster_id in enumerate(cluster_assignments):
        cluster_id = int(cluster_id)
        cluster_to_docs[cluster_id].append(doc_id)
        reverse_doc_map[doc_id] = (cluster_id, len(cluster_to_docs[cluster_id]) - 1)


    import json

    # Save
    with open('sift_doc_to_cluster_id.json', 'w') as f:
        json.dump(reverse_doc_map, f)

    # assert len(cluster_to_docs) == 1000
    assert sum(len(d) for d in cluster_to_docs.values()) == 1000000
    import os

    for cluster_id, doc_ids in cluster_to_docs.items():
        vectors_in_cluster = []

        for doc_id in doc_ids:
            doc_embedding = doc_vectors[doc_id]
            vectors_in_cluster.append(doc_embedding)

        vectors_in_cluster = np.array(vectors_in_cluster)
        print(vectors_in_cluster.shape)
        np.savetxt(f"sift_eval_artifacts/sift_cluster_{cluster_id}.csv", vectors_in_cluster, delimiter=",")
    

    original_ground_truth = np.load('data/sift/sift-128-euclidean.gtruth.npy')
    original_ground_truth = original_ground_truth[8000:8100]
    original_ground_truth = original_ground_truth[:, :10]

    cluster_ground_truth = []

    for doc_ids in original_ground_truth:
        tuple_ground_truth = []
        for doc_id in doc_ids:
            cluster_gt = reverse_doc_map[doc_id]
            tuple_ground_truth.append(cluster_gt)

        cluster_ground_truth.append(tuple_ground_truth)
    
    import csv

    with open('sift_eval_artifacts/sift_ground_truth.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in cluster_ground_truth:
            flat_row = [item for tup in row for item in tup]
            writer.writerow(flat_row)

    metadata = {"num_vectors": 1000000, "num_clusters": 5, "dim": 128}

    import json
    with open('sift_eval_artifacts/sift_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4) 

if __name__ == '__main__':
    app.run(main)
