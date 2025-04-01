import tensorflow as tf
import struct
import tensorflow_datasets as tfds
import numpy as np

tensorflow_name_of_dataset = { 'sift' : 'sift1m', 'glove' : 'glove100_angular', 'deep': 'deep1b'}

def download_and_write_dataset(dataset_name, normalize=False):
    print("Downloading and writing " + dataset_name + " dataset")
    # Load the dataset using TensorFlow Datasets
    builder = tfds.builder(tensorflow_name_of_dataset[dataset_name])
    builder.download_and_prepare()

    with open('data/' + dataset_name + '.fbin', 'wb') as f:
        dataset = builder.as_dataset(split='database')
        
        f.write(len(dataset).to_bytes(4, byteorder='little', signed=False))
        
        dim = next(iter(dataset))['embedding'].shape[0]
        f.write(dim.to_bytes(4, byteorder='little', signed=False))
        
        for sample in dataset:
            embedding = sample['embedding'].numpy()
            if normalize:
                embedding = embedding / np.linalg.norm(embedding)
            for value in embedding:
                f.write(struct.pack('<f', value))
    print("Base set written")

    with open('data/' +dataset_name + '.query.fbin', 'wb') as qf:
        with open('data/' +dataset_name + '.ground-truth.fbin', 'wb') as gtf:
            dataset = builder.as_dataset(split='test')
            
            qf.write(len(dataset).to_bytes(4, byteorder='little', signed=False))
            gtf.write(len(dataset).to_bytes(4, byteorder='little', signed=False))
            
            dim = next(iter(dataset))['embedding'].shape[0]
            K = next(iter(dataset))['neighbors']['index'].shape[0]
            qf.write(dim.to_bytes(4, byteorder='little', signed=False))
            gtf.write(K.to_bytes(4, byteorder='little', signed=False))
            
            for sample in dataset:
                embedding = sample['embedding'].numpy()
                if normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                for value in embedding:
                    qf.write(struct.pack('<f', value))
                neighbor_indices = sample['neighbors']['index']
                for i in neighbor_indices:
                    gtf.write(int(i).to_bytes(4, byteorder='little', signed=False))
    print("Queries and ground truth written")

download_and_write_dataset('deep')
download_and_write_dataset('sift')
download_and_write_dataset('glove', normalize=True)
