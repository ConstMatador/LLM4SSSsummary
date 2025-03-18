import numpy as np

model_selected = "UniTime"

source_path = f"1stBSF_Tightness/{model_selected}/approSeries.bin"
target_paths = [
    f"1stBSF_Tightness/{model_selected}/approSeries_1.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_5.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_10.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_50.bin",
    f"1stBSF_Tightness/{model_selected}/approSeries_100.bin"
]
len_series = 16
whole_size = 5000
slice_size = 1000

sequences = []
for idx in range(0, whole_size):
    sequence = np.fromfile(source_path, dtype=np.float32, count=len_series, offset=4 * len_series * idx)
    sequences.append(sequence)

for i in range(0, 5):
    # 使用 np.concatenate 来拼接数组，而不是 np.append
    temp = np.concatenate(sequences[slice_size * i:slice_size * (i + 1)])
    temp.tofile(target_paths[i])
