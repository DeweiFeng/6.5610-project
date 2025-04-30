DATASET_NAME="msmarco"
EMBEDDING_NAME="mini"
DATASET_DOCS_PATH="data/aligned_data/documents.npy"
DATASET_QUERIES_PATH="data/aligned_data/queries.npy"
DATASET_NEIGHBORS_PATH="data/aligned_data/neighbors.npy"
NUM_CLUSTERS=3000
NUM_EPOCHS=100
TOP_K=1
nprobe=1


python3 main_mips.py \
	--name_dataset ${DATASET_NAME} \
	--name_embedding ${EMBEDDING_NAME} \
	--format_file npy \
	--dataset_docs ${DATASET_DOCS_PATH} \
	--dataset_queries ${DATASET_QUERIES_PATH} \
	--dataset_neighbors ${DATASET_NEIGHBORS_PATH} \
	--algorithm kmeans \
	--nclusters ${NUM_CLUSTERS} \
	--top_k ${TOP_K} \
	--test_split_percent 20 \
	--split_seed 42 \
	--ells ${nprobe} \
	 --learner_nunits 0 \
	 --learner_nepochs ${NUM_EPOCHS} \
	 --compute_clusters 0

