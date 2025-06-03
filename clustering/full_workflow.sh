# STEP 1: Assume we have a folder containing the document embeddings and query embeddings. Compute the ground truth via exact NN search
echo "Starting by computing ground truth..."
cd baseline_data 

python3 compute_gnd_gpu.py \
  --doc msmarco_doc_embeddings.npy \
  --query msmarco_query_embeddings.npy \
  --output_index ground_truth_1M_k10.npy \
  --output_doc msmarco_doc_embeddings_1M_norm.npy \
  --output_query msmarco_query_embeddings_norm.npy

# STEP 2: Perform clustering of the document embeddings

echo "Step 2, compute clustering"
python3 kmeans_gpu.py --input msmarco_doc_embeddings_1M_norm.npy --output_centroids centroids.npy --output_assignments cluster_assignments.npy 


# STEP 3: Compute the query eval dataset and model training data
cd ..
echo "Step 3, compute training data"
python3 compute_training_data.py

# STEP 4: Train the model
cd cluster_model
echo "Step 4, train the model"
python3 train_model.py
python3 train_baseline.py
# Step 5: Evaluate
cd ..

python3 cluster_search.py -n 1000000 -d 384 -k 10 -q 1000 -input ./baseline_data/msmarco_doc_embeddings_1M_norm.npy -query ./eval_data/query_test_reduced.npy -gnd ./eval_data/ground_truth_test_k10_reduced.npy -mode learned -report ./msmarco-report.txt


python3 cluster_search.py -n 1000000 -d 384 -k 10 -q 1000 -input ./baseline_data/msmarco_doc_embeddings_1M_norm.npy -query ./eval_data/query_test_reduced.npy -gnd ./eval_data/ground_truth_test_k10_reduced.npy -mode baseline -report ./msmarco-report.txt


# Step 6: Prepare data into tiptoe format

python3 prepare_tiptoe_data/prepare_doc_data.py
python3 prepare_tiptoe_data/prepare_query_data.py
python3 prepare_tiptoe_data/save_metadata.py
python3 prepare_tiptoe_data/prepare_ground_truth.py

