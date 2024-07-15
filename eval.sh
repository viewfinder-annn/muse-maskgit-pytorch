#!/bin/bash

# 配置文件路径
cfg_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/urgentchallenge2024-55M.json"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-1-step-80000-loss-4.542117014741898/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-1-step-75000-loss-4.569438846969605/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240711-20:19/model/epoch-4-step-60000-loss-4.687433513641357/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-3-step-100000-loss-4.458391998529434/model.pt"
model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-23-step-360000-loss-4.113462843036651/model.pt"
exp_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=6

# 不需要重新采样的输入文件夹列表
input_folders_no_resample=(
    "/mnt/data2/zhangjunan/enhancement/data/voicefixer/TestSets/ALL_GSR/simulated"
)

# 循环遍历输入文件夹并执行推理任务
for input_folder in "${input_folders_no_resample[@]}"; do
    python -u infer.py --config "$cfg_path" --exp "$exp_path" --model "$model_path" --input_folder "$input_folder" --dnsmos
done

# 输入文件夹列表
input_folders_16k=(
    "/mnt/data2/zhangjunan/dns2020-repo/datasets/test_set/synthetic/no_reverb/noisy"
    "/mnt/data2/zhangjunan/dns2020-repo/datasets/test_set/synthetic/with_reverb/noisy"
    "/mnt/data2/zhangjunan/dns2020-repo/datasets/test_set/real_recordings"
)

# 循环遍历输入文件夹并执行推理任务
for input_folder in "${input_folders_16k[@]}"; do
    python -u infer.py --config "$cfg_path" --exp "$exp_path" --model "$model_path" --input_folder "$input_folder" --resample 16000 --dnsmos
done