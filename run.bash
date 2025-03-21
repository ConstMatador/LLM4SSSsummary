dataset="astro"
model="AutoTimes"
nohup python -u LLM4SSSsummary_run.py -C conf/$dataset/$model.json > run.out 2>&1 &