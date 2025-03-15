import numpy as np


model_selected = "S2IPLLM"

data_path = "1stBSF_Data/" + model_selected + "/reduce_data.bin"
query_path = "1stBSF_Data/" + model_selected + "/reduce_query.bin"
output_path = "1stBSF_Tightness/" + model_selected + "/exact.txt"

data = np.fromfile(data_path, dtype=np.float32).reshape(1000000, 256)
queries = np.fromfile(query_path, dtype=np.float32).reshape(1000, 256)

nearest_indices = []
for i, q in enumerate(queries):
    distances = np.linalg.norm(data - q, axis=1)  
    nearest_idx = np.argmin(distances)  
    nearest_indices.append(nearest_idx)

with open(output_path, "w") as f:
    f.write("\n".join(map(str, nearest_indices)))
