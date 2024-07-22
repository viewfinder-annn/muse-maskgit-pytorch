#!/bin/bash

# 配置文件路径
# cfg_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/masksr-55M.json"
cfg_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/masksr-55M-encoder-loss.json"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-1-step-80000-loss-4.542117014741898/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-1-step-75000-loss-4.569438846969605/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240711-20:19/model/epoch-4-step-60000-loss-4.687433513641357/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-3-step-100000-loss-4.458391998529434/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53/model/epoch-23-step-360000-loss-4.113462843036651/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240715-12:10/model/epoch-20-step-130000-loss-4.743366591072083/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240715-12:10/model/epoch-34-step-220000-loss-4.6064291365623475/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240717-16:43/model/epoch-17-step-115000-loss-4.736460160350799/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240715-12:10/model/epoch-46-step-300000-loss-4.541741327381134/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240719-17:09/model/epoch-11-step-255000-loss-4.506019418048859/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-encoder-loss-20240719-17:07/model/epoch-7-step-175000-loss-5.093227394008636/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-encoder-loss-20240719-17:07/model/epoch-10-step-220000-loss-5.001549393939972/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-32-20240720-16:07/model/epoch-14-step-180000-loss-4.736009558677673/model.pt"
# model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-32-20240720-16:07/model/epoch-27-step-350000-loss-4.555379238224029/model.pt"
model_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-32-20240720-16:07/model/epoch-31-step-400000-loss-4.526753110647202/model.pt"
# exp_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/urgentchallenge2024-55M-20240712-11:53"
# exp_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240715-12:10"
# exp_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-20240719-17:09"
# exp_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-encoder-loss-20240719-17:07"
exp_path="/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-32-20240720-16:07"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=5

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