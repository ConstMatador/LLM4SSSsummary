import os
import matplotlib.pyplot as plt

save_path = "figure/figure/union.png"

def read_files(file_paths):
    data = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                data[file_path] = [float(line.strip()) for line in lines]
        else:
            print(f"File not found: {file_path}")
    return data

def plot_data(data, labels):
    plt.figure(figsize=(10, 6))
    x_values = list(range(1, 101))
    
    for file_path, values in data.items():
        label = labels.get(file_path, os.path.basename(file_path))
        plt.plot(x_values, values, label=label)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, 101, 5))
    plt.legend()
    plt.title("Loss of Experimental Models")
    plt.grid()
    plt.savefig(save_path)

file_paths = [
    "figure/error/GPT4SSS.txt",
    "figure/error/TimeLLM.txt",
    "figure/error/UniTime.txt",
    "figure/error/AutoTimes.txt"
]

labels = {
    "figure/error/GPT4SSS.txt": "GPT4SSS",
    "figure/error/TimeLLM.txt": "TimeLLM",
    "figure/error/UniTime.txt": "UniTime",
    "figure/error/AutoTimes.txt": "AutoTimes"
}

data = read_files(file_paths)
plot_data(data, labels)