import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from infer import main

model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-20240713-03:40/model/epoch-75-step-250000-loss-3.872418055152893/model.pt'
config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-55M.json'
exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-20240713-03:40'
resample = None
dnsmos = True

# model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-encoder-loss-20240721-13:54/model/epoch-30-step-200000-loss-3.9031185120105745/model.pt'
# config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-55M-encoder-loss.json'
# exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-encoder-loss-20240721-13:54/'
# resample = None
# dnsmos = True


# simulated data
input_folder = '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen/noisy'

main(model_path, config_path, input_folder, exp, resample, dnsmos)

model_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-encoder-loss-20240722-19:04/model/epoch-6-step-250000-loss-3.8247188554286957/model.pt'
config_path = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/config/singing-55M-encoder-loss.json'
exp = '/mnt/data2/zhangjunan/muse-maskgit-pytorch/exp/singing-55M-encoder-loss-20240722-19:04'
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
