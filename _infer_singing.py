import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from infer import main

model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-20240713-03:40/model/epoch-121-step-400000-loss-3.7843124058246613/model.pt'
config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-55M.json'
exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-20240713-03:40'
resample = None
dnsmos = True


model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-encoder-loss-20240719-17:07/model/epoch-16-step-295000-loss-4.897931021595001/model.pt'
config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/masksr-55M-encoder-loss.json'
exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/masksr-55M-encoder-loss-20240719-17:07'
resample = None
dnsmos = True


# simulated data
input_folder = '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen/noisy'

main(model_path, config_path, input_folder, exp, resample, dnsmos)

# # real data
# input_folders = [
#     '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/v1劣化/noisy',
#     '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/v2劣化/noisy',
#     '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/v3多混响劣化/noisy',
#     '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/加混再分离劣化/noisy',
#     '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/真实室内环境数据/noisy',
#     '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/真实演唱会数据/noisy'
# ]

# for input_folder in input_folders:
#     main(model_path, config_path, input_folder, exp, resample, dnsmos)
