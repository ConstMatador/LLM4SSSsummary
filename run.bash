dataset="deep1B"
model="TimeLLM"
nohup python -u LLM4SSSsummary_run.py -C conf/$dataset/$model.json > run.out 2>&1 &