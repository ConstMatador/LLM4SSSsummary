import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


model_selected = "S2IPLLM"

data_path = f"1stBSF_Data/{model_selected}/origin_data.bin"
query_path = f"1stBSF_Data/{model_selected}/origin_query.bin"
output_path = f"1stBSF_Tightness/{model_selected}/exactIdx.txt"

data = np.fromfile(data_path, dtype=np.float32).reshape(1000000, 256)
queries = np.fromfile(query_path, dtype=np.float32).reshape(1000, 256)

tree = cKDTree(data)

nearest_indices = []
batch_size = 100

for i in tqdm(range(0, len(queries), batch_size), desc="Processing Queries", unit="batch"):
    batch_queries = queries[i:i+batch_size]
    _, batch_nearest_indices = tree.query(batch_queries, k=1)
    nearest_indices.extend(batch_nearest_indices)

with open(output_path, "w") as f:
    f.write("\n".join(map(str, nearest_indices)))
