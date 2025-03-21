import sys
import argparse
import torch
import numpy as np
from torch import nn

root_dir = "/mnt/data/user_liangzhiyu/wangzhongzheng/LLM4SSSsummary/"
sys.path.append(root_dir)

from utils.conf import Configuration
from model.GPT4SSS import GPT4SSS
from model.TimeLLM import TimeLLM
from model.AutoTimes import AutoTimes
from model.UniTime import UniTime
from model.S2IPLLM import S2IPLLM


# configure
origin_data_path = "./nnCoverage/data/origin_data.bin"
origin_query_path = "./nnCoverage/data/origin_query.bin"
reduce_data_path = "./nnCoverage/data/reduce_data.bin"
reduce_query_path = "./nnCoverage/data/reduce_query.bin"

max_data_size = 1000000
data_size = 20000
max_query_size = 10000
query_size = 1000
batch_size = 200


def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1: ])
    conf = Configuration(args.confpath)
    
    model_selected = conf.getEntry("model_selected")
    device = conf.getEntry("device")
    selected_devices = conf.getEntry("GPUs")
    dataset_selected = conf.getEntry("dataset_selected")
    model_path = f"./example/{model_selected}/save/200000train_{dataset_selected}.pth"

    data_path = conf.getEntry("data_path")
    data_pos = data_path + dataset_selected + "/data.bin"
    query_pos = data_path + dataset_selected + "/query.bin"
    
    len_series = conf.getEntry("len_series")
    len_reduce = conf.getEntry("len_reduce")
    
    # getTestData Function
    def getTestData(data_pos, query_pos, data_size, query_size):
        data_indices = np.random.randint(0, max_data_size, size=data_size, dtype=np.int64)
        query_indices = np.random.randint(0, max_query_size, size=query_size, dtype=np.int64)
        
        origin_data = []
        for index in data_indices:
            sequence = np.fromfile(data_pos, dtype=np.float32, count=len_series, offset=4*len_series*index)
            if not np.isnan(np.sum(sequence)):
                origin_data.append(sequence)
        origin_data = np.array(origin_data, dtype=np.float32)
        origin_data.tofile(origin_data_path)
        
        origin_query = []
        for index in query_indices:
            sequence = np.fromfile(query_pos, dtype=np.float32, count=len_series, offset=4*len_series*index)
            if not np.isnan(np.sum(sequence)):
                origin_query.append(sequence)
        origin_query = np.array(origin_query, dtype=np.float32)
        origin_query.tofile(origin_query_path)

        origin_data, origin_query = torch.from_numpy(origin_data), torch.from_numpy(origin_query)

        return origin_data, origin_query
    # End Function
    
    
    origin_data, origin_query = getTestData(data_pos, query_pos, data_size, query_size)
    # origin_data: [1000000, 256]
    # origin_query: [1000, 256]
    origin_data = origin_data.reshape(-1, batch_size, len_series).to(device)
    # origin_data: [1000000/100, 100, 256]
    origin_query = origin_query.reshape(-1, batch_size, len_series).to(device)
    # origin_query: [1000/100, 100, 256]
    
    if model_selected == "GPT4SSS":
        model = GPT4SSS(conf)
    elif model_selected == "TimeLLM":
        model = TimeLLM(conf)
    elif model_selected == "AutoTimes":
        model = AutoTimes(conf)
    elif model_selected == "UniTime":
        model = UniTime(conf)
    elif model_selected == "S2IPLLM":
        model = S2IPLLM(conf)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=selected_devices).to(device)
        
    if model_selected == "UniTime":
            mask = torch.ones((1, len_series)).to(device)
            
    reduce_batches = []
    i = batch_size
    for batch in origin_data:
        # batch: [100, 256]
        with torch.no_grad():
            if model_selected == "UniTime":
                reduce_batch = model(batch, mask)
            else:
                reduce_batch = model(batch)
            # reduce_batch: [100, 16]
            reduce_batch = reduce_batch.cpu().numpy()
        reduce_batches.append(reduce_batch)
        print(f"data {i} completed.")
        i = i + batch_size
    # reduce_batches: [1000000/100, 100, 16]
    reduce_batches = np.array(reduce_batches, dtype=np.float32)
    reduce_data = reduce_batches.reshape(-1, len_reduce)
    # reduce_data: [1000000, 16]
    reduce_data.tofile(reduce_data_path)
    torch.cuda.empty_cache()
    
    reduce_batches = []
    i = batch_size
    for batch in origin_query:
        # batch: [100, 256]
        with torch.no_grad():
            if model_selected == "UniTime":
                reduce_batch = model(batch, mask)
            else:
                reduce_batch = model(batch)
            reduce_batch = reduce_batch.cpu().numpy()
        reduce_batches.append(reduce_batch)
        print(f"query {i} completed.")
        i = i + batch_size
    reduce_batches = np.array(reduce_batches, dtype=np.float32)
    reduce_query = reduce_batches.reshape(-1, len_reduce)
    # reduce_data: [1000, 16]
    reduce_query.tofile(reduce_query_path)
    torch.cuda.empty_cache()
    
    
if __name__ == '__main__':
    main(sys.argv)