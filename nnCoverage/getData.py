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


# configure
selected_model = "AutoTimes"

model_path = "./example/"+selected_model+"/save/200000train_human.pth"
data_path = "../Data/SEAnet/data_250M_seed1184_len256_znorm.bin"
query_path = "../Data/SEAnet/data_250k_seed14784_len256_znorm.bin"

origin_data_path = "./nnCoverage/data/origin_data.bin"
origin_query_path = "./nnCoverage/data/origin_query.bin"
reduce_data_path = "./nnCoverage/data/reduce_data.bin"
reduce_query_path = "./nnCoverage/data/reduce_query.bin"

device = "cuda:6"
selected_devices = [6]
selected_model = "AutoTimes"
max_size = 1000000
data_size = 20000
query_size = 1000
len_series = 256
len_reduce = 16


def getTestData(data_path, data_size, query_size):
    data_indices = np.random.randint(0, max_size, size=data_size, dtype=np.int64)
    query_indices = np.random.randint(0, max_size, size=query_size, dtype=np.int64)
    
    origin_data = []
    for index in data_indices:
        sequence = np.fromfile(data_path, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        if not np.isnan(np.sum(sequence)):
            origin_data.append(sequence)
            
    origin_data = np.array(origin_data, dtype=np.float32)
    origin_data.tofile(origin_data_path)
    
    origin_query = []
    for index in query_indices:
        sequence = np.fromfile(data_path, dtype = np.float32, count = len_series, offset = 4 * len_series * index)
        if not np.isnan(np.sum(sequence)):
            origin_query.append(sequence)
            
    origin_query = np.array(origin_query, dtype=np.float32)
    origin_query.tofile(origin_query_path)
    
    origin_data, origin_query = torch.from_numpy(origin_data), torch.from_numpy(origin_query)
    
    # origin_data = np.fromfile(data_path, dtype=np.float32, count=20000 * len_series).reshape(-1, len_series)
    # origin_query = np.fromfile(query_path, dtype=np.float32, count=1000 * len_series).reshape(-1, len_series)
    # origin_data, origin_query = torch.from_numpy(origin_data), torch.from_numpy(origin_query)

    return origin_data, origin_query


# main
def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters for LLM4SSS')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1: ])
    conf = Configuration(args.confpath)

    origin_data, origin_query = getTestData(data_path, data_size, query_size)

    if selected_model == "GPT4SSS":
        model = GPT4SSS(conf)
    elif selected_model == "TimeLLM":
        model = TimeLLM(conf)
    elif selected_model == "AutoTimes":
        model = AutoTimes(conf)
    elif selected_model == "UniTime":
        model = UniTime(conf)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=selected_devices).to(device)
        
    model.load_state_dict(torch.load(model_path))

    len_series = conf.getEntry('len_series')
    
    reduce_data = []
    i = 1
    for data in origin_data:
        with torch.no_grad():
            data = model(data.unsqueeze(0).to(device)).cpu()
            data = data.reshape(-1).cpu().numpy()
        reduce_data.append(data)
        print(f"data {i} completed.")
        i = i + 1
              
    reduce_data = np.array(reduce_data, dtype=np.float32)
    reduce_data.tofile(reduce_data_path)
    torch.cuda.empty_cache()
        
    reduce_query = []
    i = 1
    for data in origin_query:
        with torch.no_grad():
            data = model(data.unsqueeze(0).to(device))
            data = data.reshape(-1).cpu().numpy()
        reduce_query.append(data)
        print(f"query {i} completed.")
        i = i + 1
    
    reduce_query = np.array(reduce_query, dtype=np.float32)
    reduce_query.tofile(reduce_query_path)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main(sys.argv)