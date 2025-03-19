import numpy as np


data_num = 1_000_000
query_num = 1000
len_series = 16
model_selected = "GPT4SSS"

data_path = f"1stBSF_Data/{model_selected}/reduce_data.bin"
query_paths = [
    f"1stBSF_Tightness/{model_selected}/approSeries_1.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_5.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_10.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_50.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_100.bin"
]
output_paths = [
    f"1stBSF_Tightness/{model_selected}/approIndex_1.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_5.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_10.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_50.txt",
    f"1stBSF_Tightness/{model_selected}/approIndex_100.txt"
]

for i in range(0, 5):
    query_path = query_paths[i]
    data_seq = np.memmap(data_path, dtype=np.float32, mode='r', shape=(data_num, len_series))
    query_seq = np.fromfile(query_path, dtype=np.float32, count=query_num * len_series).reshape(query_num, len_series)
    output_path = output_paths[i]
    
    with open(output_path, 'a') as f:
        for query_idx, query in enumerate(query_seq):
            match = np.all(data_seq == query, axis=1)
            match_index = np.where(match)[0][0]
            if match_index.size > 0:
                print(f"Series {query_idx} found at index {match_index}.")
                f.write(f"{match_index}\n")
            else:
                raise ValueError(f"Series {query_idx} failed to find the index.")
