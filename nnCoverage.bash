dataset="sald"
model="AutoTimes"
python ./nnCoverage/getData.py -C ./conf/$dataset/$model.json
python ./nnCoverage/nnCoverage.py -C ./conf/$dataset/$model.json
