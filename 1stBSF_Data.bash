rm 1stBSF_Data.out
dataset="human"
model="TimeLLM"
nohup python -u ./1stBSF_Data/genData.py -C ./conf/$dataset/$model.json > 1stBSF_Data.out 2>&1 &