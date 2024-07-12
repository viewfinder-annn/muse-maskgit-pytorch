import torchaudio
import torch
import os
from tqdm import tqdm
import json5
import json
from trainer import pad_or_truncate, get_model
import time

def load_model(model_path, config, device):
    model = get_model(config['model'], device)  # get_model needs to be defined or imported appropriately
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer(model, input_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)
    enhanced_dir = os.path.join(output_folder, 'enhanced')
    noisy_dir = os.path.join(output_folder, 'noisy')
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    window_size = 512 * 256  # 窗口大小

    for file_name in tqdm(file_list):
        file_path = os.path.join(input_folder, file_name)
        signal, sr = torchaudio.load(file_path)
        signal = signal.to(device)

        # 重采样到模型需要的采样率
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(device)
        signal = resampler(signal)

        original_length = signal.shape[-1]  # 保存原始音频的长度

        # 用于存储处理过的音频片段
        enhanced_audio = []
        start = 0

        while start < signal.shape[-1]:
            end = start + window_size
            segment = signal[:, start:end]

            # 如果段落长度不足，则进行填充
            if segment.shape[-1] < window_size:
                segment = torch.nn.functional.pad(segment, (0, window_size - segment.shape[-1]))

            with torch.no_grad():
                # 假设model.generate返回增强的音频
                ids, output_segment = model.generate(segment.unsqueeze(0), timesteps=40)
                enhanced_audio.append(output_segment.squeeze(0))

            start += window_size  # 移动窗口，这里不重叠

        # 将所有处理过的音频片段拼接起来
        enhanced_signal = torch.cat(enhanced_audio, dim=1)

        # 剪裁到原始音频的长度
        enhanced_signal = enhanced_signal[:, :original_length]

        # 保存增强的音频和原始噪声音频
        output_file_path = os.path.join(enhanced_dir, file_name)
        torchaudio.save(output_file_path, enhanced_signal.detach().cpu(), 44100)
        torchaudio.save(os.path.join(noisy_dir, file_name), signal.detach().cpu(), 44100)
        print(f"Processed {file_name} -> {output_file_path}")

def main(model_path, config_path, input_folder, output_folder):
    config = json5.load(open(config_path))
    device = config['train']['device']
    model = load_model(model_path, config, device)
    infer(model, input_folder, output_folder, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate audio outputs from a given folder of audio files.')
    parser.add_argument('--exp', type=str, required=True, help='Path to the experiment folder.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input audio files.')
    
    args = parser.parse_args()
    
    # TODO: Add a timestamp to the output folder name, and metadata about the model/data used for inference
    output_folder = os.path.join(args.exp, 'infer', time.strftime('%Y%m%d-%H:%M'))
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    main(args.model, args.config, args.input_folder, output_folder)