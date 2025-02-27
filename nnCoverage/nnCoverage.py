import cupy as cp
import statistics
import os
import gc


os.chdir('/mnt/data/user_liangzhiyu/wangzhongzheng/LLM4SSSsummary/')

origin_data_path = "./nnCoverage/data/origin_data.bin"
origin_query_path = "./nnCoverage/data/origin_query.bin"
reduce_data_path = "./nnCoverage/data/reduce_data.bin"
reduce_query_path = "./nnCoverage/data/reduce_query.bin"

knn_origin = "./nnCoverage/result/knn_origin.txt"
knn_reduce = "./nnCoverage/result/knn_reduce.txt" 

data_size = 20000
query_size = 1000
len_series = 256
len_reduce = 16
k = 10


def get_data(data_path, data_size, length):
    data = cp.fromfile(data_path, dtype=cp.float32, count=data_size * length).reshape(-1, length)
    return data

def clear_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()  # 清理 cupy 内存
    gc.collect()  # 强制进行垃圾回收

def knn_search_batch(data, queries_batch, k):
    # data: (data_size, len), queries_batch: (batch_size, len)
    distances = cp.linalg.norm(data[:, cp.newaxis] - queries_batch, axis=2)
    # distances: (data_size, batch_size)
    knn_indices = cp.argsort(distances, axis=0)[:k]
    # knn_indices: (k, batch_size)
    knn_distances = cp.take_along_axis(distances, knn_indices, axis=0)
    # knn_distances: (k, batch_size)
    return knn_indices, knn_distances

def process_batches(data, queries, k, batch_size=100):
    knn_indices_all = []
    knn_distances_all = []
    num_batches = (queries.shape[0] + batch_size - 1) // batch_size  # 计算批次数量
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, queries.shape[0])
        queries_batch = queries[start:end]
        knn_indices_batch, knn_distances_batch = knn_search_batch(data, queries_batch, k)
        knn_indices_all.append(knn_indices_batch)
        knn_distances_all.append(knn_distances_batch)
    # 将结果拼接起来
    knn_indices_all = cp.concatenate(knn_indices_all, axis=1)
    knn_distances_all = cp.concatenate(knn_distances_all, axis=1)
    return knn_indices_all, knn_distances_all

def write_results_to_file(knn_indices, knn_distances, k, output_file):
    with open(output_file, 'w') as f:
        for query_idx in range(knn_indices.shape[1]):
            for nn_idx in range(k):
                distance = knn_distances[nn_idx, query_idx]
                index = knn_indices[nn_idx, query_idx]
                f.write(f"the [{query_idx}] query [{nn_idx}] NN is {distance:.6f} at {index}\n")
                
def read_results_from_file(input_file):
    nn_data = {}
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            query_index = int(parts[1].strip('[]'))
            nn_index = int(parts[3].strip('[]'))
            nn_value = float(parts[6])
            nn_position = int(parts[8])
            if query_index not in nn_data:
                nn_data[query_index] = []
            nn_data[query_index].append((nn_index, nn_value, nn_position))
    return nn_data
    # {
    #     query_index: [
    #         (nn_index, nn_value, nn_position),  # 最近邻的编号、距离、位置
    #         ...
    #     ],
    #     ...
    # }
    
def compare_nn_positions(nn_data1, nn_data2):
    similarity_ratios = {}
    for query_index in nn_data1:
        if query_index in nn_data2:
            nn_positions1 = [nn[2] for nn in nn_data1[query_index]]
            nn_positions2 = [nn[2] for nn in nn_data2[query_index]]
            common_positions = set(nn_positions1) & set(nn_positions2)
            similarity_ratio = len(common_positions) / len(nn_positions1)  # nn_positions1 和 nn_positions2 长度相同
            similarity_ratios[query_index] = similarity_ratio
    return statistics.mean(similarity_ratios.values())


origin_data = get_data(origin_data_path, data_size, len_series)
origin_query = get_data(origin_query_path, query_size, len_series)
reduce_data = get_data(reduce_data_path, data_size, len_reduce)
reduce_query = get_data(reduce_query_path, query_size, len_reduce)
# print(origin_data.shape, origin_query.shape)
# print(reduce_data.shape, reduce_query.shape)
knn_indices_origin, knn_distances_origin = process_batches(origin_data, origin_query, k)
knn_indices_reduce, knn_distances_reduce = process_batches(reduce_data, reduce_query, k)
# print(knn_indices_origin.shape, knn_distances_origin.shape)
# print(knn_indices_reduce.shape, knn_distances_reduce.shape)
write_results_to_file(knn_indices_origin, knn_distances_origin, k, knn_origin)
write_results_to_file(knn_indices_reduce, knn_distances_reduce, k, knn_reduce)

knn_result_origin = read_results_from_file(knn_origin)
knn_result_reduce = read_results_from_file(knn_reduce)

mean_similarity_ratio = compare_nn_positions(knn_result_origin, knn_result_reduce)

print(f"Mean similarity ratio: {mean_similarity_ratio:.6f}")