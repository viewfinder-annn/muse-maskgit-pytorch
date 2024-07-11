import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from dataset.audio_degradation_pipeline import process_from_audio_path

class AudioDegradationDataset(Dataset):
    def __init__(self, speech_list, noise_list, rir_list, degradation_config, seq_len=512*256, sr=44100, device='cpu'):
        # TODO: deal with file root
        self.speech_list, self.noise_list, self.rir_list = self.get_list(speech_list, noise_list, rir_list, '/mnt/data2/zhangjunan/urgent2024_challenge/')
        self.degradation_config = degradation_config
        self.seq_len = seq_len
        self.sr = sr
        self.device = device

    def get_list(self, speech_list, noise_list, rir_list, root):
        def parse_file(file_path):
            list_paths = []
            with open(file_path, 'r') as file:
                for line in file:
                    # 使用split()分割行，并取最后一个元素作为路径
                    rel_path = line.strip().split()[-1]
                    
                    # urgent challenge bug
                    if 'wsj0_wav' in rel_path:
                        rel_path = rel_path.replace('wsj0_wav', 'wsj0_wav/csr_1')
                    if 'wsj1_wav' in rel_path:
                        rel_path = rel_path.replace('wsj1_wav', 'wsj1_wav/csr_2_comp')
                    
                    # 将根路径与相对路径合并，形成完整路径
                    full_path =  os.path.normpath(os.path.join(root, rel_path))
                    list_paths.append(full_path)
            return list_paths

        speech_paths = parse_file(speech_list)
        noise_paths = parse_file(noise_list)
        rir_paths = parse_file(rir_list) if rir_list else None

        return speech_paths, noise_paths, rir_paths

    def __len__(self):
        return len(self.speech_list)

    def __getitem__(self, idx):
        speech_path = self.speech_list[idx]
        noise_path = random.choice(self.noise_list)
        rir_path = random.choice(self.rir_list) if self.rir_list else None
        # print(f"speech_path: {speech_path}, noise_path: {noise_path}, rir_path: {rir_path}")

        # Process the audio file with randomly selected noise and RIR
        speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(
            speech=speech_path, 
            noise=noise_path, 
            rir=rir_path, 
            fs=self.sr, 
            force_1ch=True,
            degradation_config=self.degradation_config
        )
        
        assert speech_sample.shape == noisy_speech.shape
        
        # Convert numpy arrays to torch tensors
        speech_sample = torch.from_numpy(speech_sample).float()
        noisy_speech = torch.from_numpy(noisy_speech).float()

        # Pad or truncate to sequence length
        speech_sample, noisy_speech = self.pad_or_truncate(speech_sample, noisy_speech)

        return speech_sample, noisy_speech

    def pad_or_truncate(self, clean, noisy):
        if clean.size(-1) < self.seq_len:
            clean = torch.nn.functional.pad(clean, (0, self.seq_len - clean.size(-1)))
            noisy = torch.nn.functional.pad(noisy, (0, self.seq_len - noisy.size(-1)))
        elif clean.size(-1) > self.seq_len:
            offset = np.random.randint(0, clean.size(-1) - self.seq_len)
            clean = clean[..., offset:offset+self.seq_len]
            noisy = noisy[..., offset:offset+self.seq_len]
        return clean, noisy

# # Example Usage
# speech_list = ['./path/to/speech1.wav', './path/to/speech2.wav']
# noise_list = ['./path/to/noise1.wav', './path/to/noise2.wav']
# rir_list = ['./path/to/rir1.wav', './path/to/rir2.wav']
# degradation_config = default_degradation_config

# dataset = AudioDataset(speech_list, noise_list, rir_list, degradation_config)
# speech_sample, noisy_speech = dataset[0]  # Fetch the first item