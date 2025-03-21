dataset="deep1B"
model="S2IPLLM"
python ./nnCoverage/getData.py -C ./conf/$dataset/$model.json
python ./nnCoverage/nnCoverage.py -C ./conf/$dataset/$model.json