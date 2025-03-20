for folder in human deep1B astro; do
    mkdir -p conf/$folder
done

for folder in GPT4SSS TimeLLM AutoTimes UniTime S2IPLLM; do
    for sub in log model sample save; do
        mkdir -p example/$folder/$sub
    done
done

for folder in GPT4SSS TimeLLM AutoTimes UniTime S2IPLLM; do
    for sub in human deep1B astro; do
        mkdir -p 1stBSF_Data/$folder/$sub
        mkdir -p 1stBSF_Tightness/$folder/$sub
    done
done