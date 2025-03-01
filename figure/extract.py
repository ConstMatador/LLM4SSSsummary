import os

selected_model = "AutoTimes200"

log_path = f"figure/log/{selected_model}.log"
save_path = f"figure/error/{selected_model}.txt"

first_extracted = False

with open(log_path, "r") as log_file:
    with open(save_path, "w") as output_file:
        for line in log_file:
            if "validate trans_err: " in line:
                trans_err_value = line.split("validate trans_err: ")[1].strip()
                
                if not first_extracted:
                        first_extracted = True
                        continue
                
                output_file.write(trans_err_value + "\n")