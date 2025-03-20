dataset="deep1B"
model="UniTime"
nohup python -u LLM4SSSsummary_run.py -C conf/$dataset/$model.json > run.out 2>&1 &