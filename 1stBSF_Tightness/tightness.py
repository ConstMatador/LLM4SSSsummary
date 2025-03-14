import os

search_node_nums = [1, 5, 10, 50, 100]

total_ratio_all_files = 0.0
num_files = len(search_node_nums)

model_selected = "1stBSF_Tightness/AutoTimes/"
file_dir = f"{model_selected}"

for node_num in search_node_nums:
    file_exac = os.path.join(file_dir, f"exact-10000.txt")
    file_appro = os.path.join(file_dir, f"appro-{node_num}.txt")

    if not os.path.exists(file_exac) or not os.path.exists(file_appro):
        print(f"file {file_exac} or {file_appro} not exist.")
        continue
    
    with open(file_exac, 'r') as file_exac, open(file_appro, 'r') as file_appro:
        lines_exac = file_exac.readlines()
        lines_appro = file_appro.readlines()

    if len(lines_exac) != len(lines_appro):
        raise ValueError(f"file {file_exac} and {file_appro} have different number of lines")

    total_ratio = 0.0
    num_lines = len(lines_exac)

    for line_exac, line_appro in zip(lines_exac, lines_appro):
        distance_exac = float(line_exac.split('Distance: ')[1])
        distance_appro = float(line_appro.split('Distance: ')[1])

        total_ratio += distance_exac / distance_appro

    average_ratio = total_ratio / num_lines
    print(f"average ratio for {node_num} nodes: {average_ratio}")