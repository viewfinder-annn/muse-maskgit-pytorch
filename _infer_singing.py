import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from infer import main

# model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240805-02:28-singing-120M/model/epoch-31-step-65000-loss-4.3617/model.pt'
# config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-120M.json'
# exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240805-02:28-singing-120M'
# resample = None
# dnsmos = True

# # simulated data
# input_folder = '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen/noisy'

# main(model_path, config_path, input_folder, exp, resample, dnsmos)

# model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240729-12:57-singing-55M/model/epoch-272-step-600000-loss-3.5587/model.pt'
# config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-55M.json'
# exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240729-12:57-singing-55M'
# resample = None
# dnsmos = True
# # simulated data
# input_folder = '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen/noisy'

# main(model_path, config_path, input_folder, exp, resample, dnsmos)

# real data
# input_folders = [
#     '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen/noisy',
#     # '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/v1劣化/noisy',
#     # '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/v2劣化/noisy',
#     # '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/v3多混响劣化/noisy',
#     # '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/加混再分离劣化/noisy',
#     # '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/真实室内环境数据/noisy',
#     # '/mnt/data2/zhangjunan/enhancement/data/huawei/processed/真实演唱会数据/noisy'
# ]
# model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240809-20:24-singing-120M/model/epoch-199-step-400000-loss-3.7716/model.pt'
# config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-120M.json'
# exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240809-20:24-singing-120M'
# resample = None
# dnsmos = True

# for input_folder in input_folders:
#     main(model_path, config_path, input_folder, exp, resample, dnsmos)



input_folder = '/mnt/data2/zhangjunan/dns2020-repo/datasets/test_set/real_recordings'

model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240812-23:07-urgentchallenge2024-77M/model/epoch-106-step-170000-loss-4.845/model.pt'
config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/urgentchallenge2024-77M.json'
exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/20240812-23:07-urgentchallenge2024-77M'
resample = None
dnsmos = True

main(model_path, config_path, input_folder, exp, resample, dnsmos)
