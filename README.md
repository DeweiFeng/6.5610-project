# 6.5610-project

Dependencies: C compiler (like GCC), Go 1.20.2, SEAL compiled with `-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=off` and `-DSEAL_USE_INTEL_HEXL=ON` and Python 3.

Usage (with test data):
```bash
go run main.go -preamble=test_data/test [-topk=10] [-clusterOnly]
```

To prepare datasets other than the test data, one should save the following files in a directory of their choice:
- `<preamble>_metadata.json`
    - contains the metadata of the dataset, including the number of clusters, the number of vectors in each cluster, the dimension of the vectors, and the number of bits used to quantize the vectors
- `<preamble>_cluster_0.csv`, `<preamble>_cluster_1.csv`, ..., `<preamble>_cluster_<C-1>.csv` for `C` clusters
    - each line is a vector of floating-point numbers in that cluster
- `<preamble>_query.csv` for the query vectors
    - each line is a query, where the first number is the cluster id of the query vector, and the rest of the floating-point numbers are the query vector itself
    - you could also use the `-query` flag to specify the path to the query vectors file, in which case the program will use the specified file instead of the default one

**All vectors must be normalized to have unit l2 norm such that dot product is the same as cosine similarity.**

To run the experiments with the new dataset, one should run the following command:
```bash
go run main.go -preamble=<preamble> [-topk=10] [-clusterOnly] [-query=<path_to_query_vectors>]
```
where `<preamble>` is the prefix of the dataset files (including the directory). For example, when `preamble` is `test_data/test`, the program will look for the following files:
- `test_data/test_metadata.json`
- `test_data/test_cluster_0.csv`, `test_data/test_cluster_1.csv`, ..., `test_data/test_cluster_<C-1>.csv` for `C` clusters
- `test_data/test_query.csv` for the query vectors if not specified with `-query` flag

If using without `-clusterOnly` flag, the client will return the top-k vectors of all clusters in the bin which the query vector's cluster belongs to. If with `-clusterOnly` flag, the client will return the top-k vectors of the query vector's cluster only, which is Tiptoe's default behavior. Running without the `-clusterOnly` flag is guaranteed to improve the search recall, because it finds the top-k vectors in a larger set of relevant vectors.

After running the above command without specifying with `-query` flag, one would see two csv files. The first one is `{preamble}_results.csv` or `{preamble}_results_cluster_only.csv`. For each line, it contains the top-k vectors that the client found for the corresponding query vector. In each row, the vectors come in pairs, where the first number is the cluster id of the vector, and the second number is the index of the vector within that cluster. For example, a row of `0,1,4,0` means that the client returns two vectors, `clusters[0][1]` and `clusters[4][0]`. The second file is `{preamble}_perf.csv` or `{preamble}_perf_cluster_only.csv`, which contains the performance statistics of each query, i.e., runtimes and message sizes.

You could also specify the path to the query vectors with `-query` flag. If not specified, the program will use the default query vectors in `{preamble}_query.csv` file. To specify the path to the query vectors, you could run the following command:
```bash
go run main.go -preamble=test_data/test -query=<path_to_query_vectors> [-topk=10] [-clusterOnly]
```
where `<path_to_query_vectors>` is the path to the query vectors file. The query vectors file should be in the same format as the `{preamble}_query.csv` file.

If running with `-query` flag, the results will be saved in `{query_file_name}_results.csv` or `{query_file_name}_results_cluster_only.csv`, where `{query_file_name}` is the name of the query vectors file without the extension. For example, if the query vectors file is `test_data/some_new_queries.csv`, the results will be saved in `test_data/some_new_queries_results.csv` or `test_data/some_new_queries_results_clusterOnly.csv`, and the performance statistics will be saved in `test_data/some_new_queries_perf.csv` or `test_data/some_new_queries_perf_clusterOnly.csv`. The specified `query` file should be inside the same directory as the `preamble` files, hence, the results files will also be saved in the same directory.

## Reproducing Experimental Results from the Project Report

Now that we have established the usage of our Tiptoe implementation, we will next share the steps to
reproduce the experimental results in our [project paper](https://65610.csail.mit.edu/2025/reports/tiptoe.pdf). 

To recap, we consider three different instantiations of Tiptoe in our experiments

* Baseline: This is the standard version of Tiptoe as specified in the original paper with k-means clustering and routing via centroids. 

* K-Means + Learned Routing: This version maintains the k-means clustering but replaces the centroid-based routing with a linear model.

* Graph Partitioning + Learned Routing: This is the best version we found in our experiments. It involves first building a k-nearest neighbor graph from the document embeddings, partitioning this graph using the popular [Metis](https://github.com/KarypisLab/METIS) balanced graph partitioning library, and then training a learned linear model to route queries to these partitions. 

The relative performance of these three methods can be found in Table 1 of our paper. Furthermore, we also report the results of searching within the selected cluster versus the entire Tiptoe bin in the paper. 

### Compute Resources

Several of these steps, such as embedding the raw text data, the k-means clustering, and the model training, can be greatly accelerated on a GPU machine. Thus, we recommend running the following steps on a GPU-enabled machine 

### Preparing Baseline Data

To prepare the baseline data into Tiptoe format, run the following script which runs several steps including 1) computing the ground-truth labels, clustering the embeddings via k-means, saving the centroids, and saving the data in Tiptoe format. 

```cd clustering```

```bash full_workflow.sh baseline```

And then, in the top-level of the repository, run the Tiptoe search

```go run main.go --preamble=tiptoe_baseline/msmarco```

Or, if you want to run in clusterOnly mode append this argument to the command

```go run main.go --preamble=tiptoe_baseline/msmarco -clusterOnly```

Then, after editing the paths in `eval.sh` if necessary, run

```bash eval.sh```


### Preparing K-Means + Learned Routing Data

This step is almost identical to the baseline except change the argument in the workflow command

```bash full_workflow.sh learned```

```go run main.go --preamble=tiptoe_baseline_learned/msmarco```



### Preparing Graph Partitioning + Learned Routing Data

The final experimental configuration can also be run in a similar manner

```bash full_workflow.sh graph```

```go run main.go --preamble=tiptoe_graph/msmarco```

