# 6.5610-project

Dependencies: C compiler (like GCC), Go 1.20.2, SEAL compiled with `-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=off` and `-DSEAL_USE_INTEL_HEXL=ON` and Python 3.

Usage (with test data):
```bash
go run main.go --preamble=test_data/test [--topk=10] [--clusterOnly]
```

To prepare datasets other than the test data, one should save the following files in a directory of their choice:
- `<preamble>_metadata.json`
    - contains the metadata of the dataset, including the number of clusters, the number of vectors in each cluster, the dimension of the vectors, and the number of bits used to quantize the vectors
- `<preamble>_cluster_0.csv`, `<preamble>_cluster_1.csv`, ..., `<preamble>_cluster_<C-1>.csv` for `C` clusters
    - each line is a vector of floating-point numbers in that cluster
- `<preamble>_query.csv` for the query vectors
    - each line is a query, where the first number is the cluster id of the query vector, and the rest of the floating-point numbers are the query vector itself

**All vectors must be normalized to have unit l2 norm such that dot product is the same as cosine similarity.**

To run the experiments with the new dataset, one should run the following command:
```bash
go run main.go --preamble=<preamble> [--topk=10] [--clusterOnly]
```
where `<preamble>` is the prefix of the dataset files (including the directory).

All vectors should have been quantized as per the `prec_bits` in the metadata file.

If using without `--clusterOnly` flag, the client will return the top-k vectors of all clusters in the bin which the query vector's cluster belongs to. If with `--clusterOnly` flag, the client will return the top-k vectors of the query vector's cluster only, which is Tiptoe's default behavior. `--clusterOnly` flag is guaranteed to improve the search recall, because it finds the top-k vectors in a larger set of relevant vectors.

After running the above command, one would see a csv file `test_data/test_results.csv` or `test_data/test_results_cluster_only.csv` of `q` lines. For each line, it contains the top-k vectors that the client found for the corresponding query vector. In each row, the vectors come in pairs, where the first number is the cluster id of the vector, and the second number is the index of the vector in that cluster. For example, a row of `0,1,4,0` means that the client returns two vectors, `clusters[0][1]` and `clusters[4][0]`.
