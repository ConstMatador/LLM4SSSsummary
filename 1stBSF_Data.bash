rm 1stBSF_Data.out
dataset="deep1B"
model="UniTime"
nohup python -u ./1stBSF_Data/genData.py -C ./conf/$dataset/$model.json > 1stBSF_Data.out 2>&1 &