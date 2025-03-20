import numpy as np
import faiss
from tqdm import tqdm

model_selected = "AutoTimes"
dataset_selected = "human"

len_series = 256
len_reduce = 16

data_num = 1_000_000
query_num = 1_000

data_pos = f"1stBSF_Data/{model_selected}/{dataset_selected}/origin_data.bin"
query_pos = f"1stBSF_Data/{model_selected}/{dataset_selected}/origin_query.bin"

# nearest
data_path = f"1stBSF_Data/{model_selected}/{dataset_selected}/origin_data.bin"
approIndex_series = f"1stBSF_Data/{model_selected}/{dataset_selected}/origin_query.bin"
exactIndex_path = f"1stBSF_Tightness/{model_selected}/{dataset_selected}/exactIndex.txt"

# split
source_path = f"1stBSF_Tightness/{model_selected}/{model_selected}/approSeries.bin"
target_paths = [
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approSeries_1.bin",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approSeries_5.bin",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approSeries_10.bin",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approSeries_50.bin",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approSeries_100.bin"
    ]
whole_size = 5000
slice_size = 1000

# getIndex
approSeries_paths = target_paths
approIndex_paths = [
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approIndex_1.txt",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approIndex_5.txt",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approIndex_10.txt",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approIndex_50.txt",
    f"1stBSF_Tightness/{model_selected}/{dataset_selected}/approIndex_100.txt"
    ]

node_nums = [1, 5, 10, 50, 100]


def nearest():
    data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 256)
    queries = np.fromfile(approIndex_series, dtype=np.float32).reshape(-1, 256)

    index = faiss.IndexFlatL2(256)
    index.add(data)

    batch_size = 100
    num_batches = (len(queries) + batch_size - 1) // batch_size

    nearest_indices = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Processing Queries", unit="batch"):
        batch_queries = queries[i:i+batch_size]
        _, batch_nearest_indices = index.search(batch_queries, 1)  # k=1
        nearest_indices.extend(batch_nearest_indices.flatten())

    np.savetxt(exactIndex_path, np.array(nearest_indices), fmt="%d")

    print(f"Processing complete. Results saved to {exactIndex_path}")
    
    
def split():
    sequences = []
    for idx in range(0, whole_size):
        sequence = np.fromfile(source_path, dtype=np.float32, count=len_reduce, offset=4 * len_reduce * idx)
        sequences.append(sequence)
    for i in range(0, 5):
        temp = np.concatenate(sequences[slice_size * i:slice_size * (i + 1)])
        temp.tofile(target_paths[i])
        

def getIndex():
    for i in range(0, 5):
        approSeries_path = approSeries_paths[i]
        data_seq = np.memmap(data_path, dtype=np.float32, mode='r', shape=(data_num, len_reduce))
        query_seq = np.fromfile(approSeries_path, dtype=np.float32, count=query_num * len_reduce).reshape(query_num, len_reduce)
        approIndex_path = approIndex_paths[i]

        with open(approIndex_path, 'a') as f:
            for query_idx, query in enumerate(query_seq):
                match = np.all(data_seq==query, axis=1)
                match_index = np.where(match)[0][0]
                if match_index.size > 0:
                    print(f"Series {query_idx} found at index {match_index}.")
                    f.write(f"{match_index}\n")
                else:
                    raise ValueError(f"Series {query_idx} failed to find the index.")
                

def tightness():
    exact_indice = []
    with open(exactIndex_path, 'r') as file:
            for line in file:
                index = int(line.strip())
                exact_indice.append(index)

    origin_data = np.fromfile(data_pos, dtype=np.float32).reshape(-1, len_series)
    origin_query = np.fromfile(query_pos, dtype=np.float32).reshape(-1, len_series)

    for appro_index_path, node_num in zip(approIndex_paths, node_nums):
        
        appro_indice = []
        with open(appro_index_path, 'r') as file:
            for line in file:
                index = int(line.strip())
                appro_indice.append(index)
                
        all_tightness = []
        for i in range(0, query_num):
            query = origin_query[i]
            exact = origin_data[exact_indice[i]]
            appro = origin_data[appro_indice[i]]
            dis1 = np.linalg.norm(query - exact)
            dis2 = np.linalg.norm(query - appro)
            if dis2 != 0:
                tightness = dis1 / dis2
            else:
                tightness = float('inf')
            all_tightness.append(tightness)
            
        tightness_mean = np.mean(all_tightness)
        print(f"1st-BSF Tightness for {node_num} nodes: {tightness_mean}")