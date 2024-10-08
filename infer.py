import torchaudio
import torch
import os
from tqdm import tqdm
import json5
import json
from trainer import pad_or_truncate, get_model
import time
os.environ['PYTHONIOENCODING'] = 'utf-8'

def get_model_old(config, device):
    from muse_maskgit_pytorch.muse_maskgit_pytorch_old import MaskGit, MaskGitTransformer, AudioEncoder
    import dac
    # Load DAC model
    dac_model = dac.DAC.load(config['dac_path']).to(device)
    dac_model.to(device)
    dac_model.eval()
    dac_model.requires_grad_(False)

    # Initialize transformer and audio encoder
    transformer_config = config['MaskGitTransformer']
    audio_encoder_config = config['AudioEncoder']
    transformer = MaskGitTransformer(**transformer_config)
    audio_encoder = AudioEncoder(**audio_encoder_config)

    # Initialize MaskGit model
    maskgit_config = config['MaskGit']
    model = MaskGit(
        vq_model=dac_model,
        transformer=transformer,
        audio_encoder=audio_encoder,
        **maskgit_config
    ).to(device)
    
    print(f"model Params: {round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2)}M")

    return model

def load_model(model_path, config, device):
    model_state_dict = torch.load(model_path)
    if 'module' in list(model_state_dict.keys())[0]:
        model = get_model(config['model'], device)  # get_model needs to be defined or imported appropriately
        model = torch.nn.DataParallel(model)
        model.load_state_dict(model_state_dict)
        model = model.module
    else:
        model = get_model_old(config['model'], device)
        model.load_state_dict(model_state_dict)
    model.eval()
    return model

def smooth_audio_transition(audio_chunks, overlap=1024):
    """
    使用重叠相加法（overlap-add）拼接音频段，并且通过加大overlap并在拼接前进行能量平滑来减少clip声。
    
    参数:
    audio_chunks (list of torch.Tensor): 需要拼接的音频段列表，每个段都是一个 torch 张量。
    overlap (int): 用于平滑处理的重叠部分的长度，默认为1024。

    返回:
    torch.Tensor: 拼接后的完整音频。
    """
    if len(audio_chunks) == 0:
        return torch.Tensor([])

    # Hanning 窗口，用于平滑拼接的重叠区域
    window = torch.hann_window(overlap * 2, periodic=True).to(audio_chunks[0].device)

    # 初始化拼接结果
    result = audio_chunks[0]
    
    for i in range(1, len(audio_chunks)):
        # 获取前一个片段的末尾部分
        previous_end = result[:, -overlap:].clone()
        
        # 上一个段的末尾应用 Hanning 窗口
        previous_end *= window[overlap:]
        
        # 当前段的开头应用 Hanning 窗口
        current_start = audio_chunks[i][:, :overlap].clone()
        current_start *= window[:overlap]

        # 将上一个片段的末尾和当前片段的开头叠加
        transition = previous_end + current_start
        
        # 拼接过渡区到结果中
        result[:, -overlap:] = transition
        
        # 将当前段的剩余部分拼接到结果中，保留音频长度
        result = torch.cat((result, audio_chunks[i][:, overlap:]), dim=1)

    return result

def infer(model, input_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)
    enhanced_dir = os.path.join(output_folder, 'enhanced')
    noisy_dir = os.path.join(output_folder, 'noisy')
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_folder) if f.endswith('.wav') or f.endswith('.flac') or f.endswith('.mp3')]
    assert len(file_list) > 0, f"No audio files found in {input_folder}."
    
    window_size = 512 * model.seq_len

    for file_name in tqdm(file_list):
        file_path = os.path.join(input_folder, file_name)
        signal, sr = torchaudio.load(file_path)
        signal = torch.mean(signal, dim=0, keepdim=True)  # 如果音频有多个声道，取平均值
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
        # enhanced_signal = torch.cat(enhanced_audio, dim=1)
        enhanced_signal = smooth_audio_transition(enhanced_audio, overlap=1024)

        # 剪裁到原始音频的长度
        enhanced_signal = enhanced_signal[:, :original_length]

        # 保存增强的音频和原始噪声音频
        output_file_path = os.path.join(enhanced_dir, file_name.split('.')[0] + '.wav')
        torchaudio.save(output_file_path, enhanced_signal.detach().cpu(), 44100)
        torchaudio.save(os.path.join(noisy_dir, file_name), signal.detach().cpu(), 44100)
        # print(f"Processed {file_name} -> {output_file_path}")

def main(model_path, config_path, input_folder, exp, resample=None, dnsmos=False):

    model_dir = os.path.split(os.path.dirname(model_path))[-1]
    output_folder = os.path.join(exp, 'infer', model_dir, time.strftime('%Y%m%d-%H:%M:%S'))
    os.makedirs(output_folder, exist_ok=True)
    metadata = {'model_path': model_path, 'config_path': config_path, 'input_folder': input_folder, 'output_folder': output_folder, 'resample': resample, 'dnsmos': dnsmos}
    # print(metadata)
    with open(os.path.join(output_folder, 'metadata.json'), 'w', encoding='utf-8') as f:
        # json.dump(vars(args), f, indent=4)
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    config = json5.load(open(config_path))
    device = config['train']['device']
    model = load_model(model_path, config, device)
    infer(model, input_folder, output_folder, device)
    
    final_output_folder = os.path.join(output_folder, 'enhanced')
    
    if resample is not None:
        import librosa
        import soundfile as sf
        print(f"Resampling output to {resample} Hz...")
        original_output_folder = os.path.join(output_folder, 'enhanced')
        file_list = [f for f in os.listdir(original_output_folder) if f.endswith('.wav')]
        resampled_output_folder = os.path.join(output_folder, f'enhanced_{resample}')
        os.makedirs(resampled_output_folder, exist_ok=True)
        for file_name in tqdm(file_list):
            y, sr = librosa.load(os.path.join(original_output_folder, file_name), sr=None)
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=resample)
            sf.write(os.path.join(resampled_output_folder, file_name), y_resampled, resample)
        final_output_folder = resampled_output_folder
    if dnsmos:
        from evaluation.dnsmos import calculate_dnsmos_score
        calculate_dnsmos_score(final_output_folder, './evaluation/DNSMOS', csv_path=os.path.join(output_folder, 'dnsmos.csv'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate audio outputs from a given folder of audio files.')
    parser.add_argument('--exp', type=str, required=True, help='Path to the experiment folder.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input audio files.')
    parser.add_argument('--resample', type=int, default=None, help='Resample input audio to this rate before inference.')
    parser.add_argument('--dnsmos', action='store_true', help='Compute DNSMOS scores for the output audio.')
    
    args = parser.parse_args()
    
    main(args.model, args.config, args.input_folder, args.exp, args.resample, args.dnsmos)