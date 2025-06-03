mode=$1

# STEP 1: Assume we have a folder containing the document embeddings and query embeddings. Compute the ground truth via exact NN search
echo "Starting by computing ground truth..."
cd util_scripts 

python3 compute_gnd_gpu.py \
  --doc ../raw_data/msmarco_doc_embeddings.npy \
  --query ../raw_data/msmarco_query_embeddings.npy \
  --output_gnd ../eval_data/ground_truth_1M_k10.npy \
  --output_doc ../eval_data/msmarco_doc_embeddings_1M_norm.npy \
  --output_query ../eval_data/msmarco_query_embeddings_norm.npy

# STEP 2: Perform clustering of the document embeddings
echo "Step 2, compute clustering"
python3 kmeans_gpu.py \
  --input ../eval_data/msmarco_doc_embeddings_1M_norm.npy \
  --output_centroids ../eval_data/centroids.npy \
  --output_assignments ../eval_data/cluster_assignments.npy

# STEP 3: Compute the query eval dataset and model training data
echo "Step 3, compute training data"
cd ../cluster_model
python3 compute_training_data.py \
  --cluster_assignments_path ../eval_data/cluster_assignments.npy \
  --ground_truth_path ../eval_data/ground_truth_1M_k10.npy \
  --query_vectors_path ../eval_data/msmarco_query_embeddings_norm.npy

# STEP 4: Train the model
echo "Step 4, train the model"
if [ "${mode}" = "baseline" ]; then
    python3 train_baseline.py
else
    python3 train_model.py
fi

# Step 5: Evaluate
cd ..

python3 search.py -n 1000000 -d 384 -k 10 -q 1000 -input ./eval_data/msmarco_doc_embeddings_1M_norm.npy -query ./eval_data/query_test_reduced.npy -gnd ./eval_data/ground_truth_test_k10.npy -mode ${mode} -report ./msmarco-report.txt

# Step 6: Prepare data into tiptoe format

python3 prepare_tiptoe_data/prepare_doc_data.py \
	--cluster_assignments_path ./eval_data/cluster_assignments.npy \
	--doc_embeddings_path ./eval_data/msmarco_doc_embeddings_1M_norm.npy

if [ "${mode}" = "baseline" ]; then
python3 prepare_tiptoe_data/prepare_query_data.py --query_vectors_path ./eval_data/query_test_reduced.npy --cluster_assignments_path ./eval_data/baseline_predicted_cluster_ids.npy
else
python3 prepare_tiptoe_data/prepare_query_data.py --query_vectors_path ./eval_data/query_test_reduced.npy --cluster_assignments_path ./eval_data/predicted_cluster_ids.npy
fi

python3 prepare_tiptoe_data/prepare_ground_truth.py \
       --reverse_index_path tiptoe_baseline/reverse_index.json \
       --ground_truth_path ./eval_data/ground_truth_1M_k10.npy       

python3 prepare_tiptoe_data/prepare_metadata.py
