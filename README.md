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