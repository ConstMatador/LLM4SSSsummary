import os
import numpy as np


DATA_DIM = 256
DATA_SIZE = 1_000_000
QUERY_SIZE = 1000

model_selected = "S2IPLLM"
origin_data_path = f"1stBSF_Data/{model_selected}/origin_data.bin"
origin_query_path = f"1stBSF_Data/{model_selected}/origin_query.bin"

if os.path.exists(origin_data_path):
    data = np.fromfile(origin_data_path, dtype=np.float32).reshape(DATA_SIZE, DATA_DIM)
else:
    raise FileNotFoundError(f"Data file not found: {origin_data_path}")

if os.path.exists(origin_query_path):
    queries = np.fromfile(origin_query_path, dtype=np.float32).reshape(QUERY_SIZE, DATA_DIM)
else:
    raise FileNotFoundError(f"Query file not found: {origin_query_path}")

search_node_nums = [1, 5, 10, 50, 100]
file_dir = f"1stBSF_Tightness/{model_selected}/"

for node_num in search_node_nums:
    exact_file = os.path.join(file_dir, "exact.txt")
    appro_file = os.path.join(file_dir, f"appro-{node_num}.txt")
    
    if not all(os.path.exists(f) for f in [exact_file, appro_file]):
        print(f"Skipping node_num {node_num}: files missing")
        continue

    with open(exact_file, 'r') as f_exact, open(appro_file, 'r') as f_appro:
        exact_indices = [int(line.strip()) for line in f_exact]
        appro_indices = [int(line.strip()) for line in f_appro]

    if len(exact_indices) != QUERY_SIZE or len(appro_indices) != QUERY_SIZE:
        raise ValueError(f"Index file length mismatch for node_num {node_num}")

    total_ratio = 0.0
    for i in range(QUERY_SIZE):
        query_vec = queries[i]
        exact_data = data[exact_indices[i]]
        appro_data = data[appro_indices[i]]

        exact_dist = np.linalg.norm(query_vec - exact_data)
        appro_dist = np.linalg.norm(query_vec - appro_data)
        # print(f"exact_dist: {exact_dist:.4f}, appro_dist: {appro_dist:.4f}")
        ratio = exact_dist / appro_dist
        total_ratio += ratio

    avg_ratio = total_ratio / QUERY_SIZE
    print(f"NodeNum {node_num}: Avg. Exact/Appro Ratio = {avg_ratio:.4f}")