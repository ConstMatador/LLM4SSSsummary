import numpy as np


len_series = 256
model_selected = "GPT4SSS"
index_num = 1000

data_path = f"1stBSF_Data/{model_selected}/origin_data.bin"
query_path = f"1stBSF_Data/{model_selected}/origin_query.bin"
exact_index_path = f"1stBSF_Tightness/{model_selected}/exactIndex.txt"
appro_index_paths = [
    f"1stBSF_Tightness/{model_selected}/approIndex_1.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_5.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_10.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_50.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_100.txt"
]
node_nums = [1, 5, 10, 50, 100]

exact_indice = []
with open(exact_index_path, 'r') as file:
        for line in file:
            index = int(line.strip())
            exact_indice.append(index)

origin_data = np.fromfile(data_path, dtype=np.float32).reshape(-1, len_series)
origin_query = np.fromfile(query_path, dtype=np.float32).reshape(-1, len_series)

for appro_index_path, node_num in zip(appro_index_paths, node_nums):
    
    appro_indice = []
    with open(appro_index_path, 'r') as file:
        for line in file:
            index = int(line.strip())
            appro_indice.append(index)
            
    all_tightness = []
    for i in range(0, index_num):
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