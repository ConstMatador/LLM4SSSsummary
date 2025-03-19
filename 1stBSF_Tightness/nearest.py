import numpy as np
import faiss
from tqdm import tqdm


model_selected = "GPT4SSS"

data_path = f"1stBSF_Data/{model_selected}/origin_data.bin"
query_path = f"1stBSF_Data/{model_selected}/origin_query.bin"
output_path = f"1stBSF_Tightness/{model_selected}/exactIndex.txt"

data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 256)
queries = np.fromfile(query_path, dtype=np.float32).reshape(-1, 256)

# 使用 FAISS 构建精确 L2 最近邻搜索索引
index = faiss.IndexFlatL2(256)  # 直接计算 L2 欧几里得距离
index.add(data)  # 添加数据到索引

batch_size = 100
num_batches = (len(queries) + batch_size - 1) // batch_size

nearest_indices = []

for i in tqdm(range(0, len(queries), batch_size), desc="Processing Queries", unit="batch"):
    batch_queries = queries[i:i+batch_size]
    _, batch_nearest_indices = index.search(batch_queries, 1)  # k=1
    nearest_indices.extend(batch_nearest_indices.flatten())

np.savetxt(output_path, np.array(nearest_indices), fmt="%d")

print(f"Processing complete. Results saved to {output_path}")
