# modified from urgent challenge https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/simulate_data_from_param.py
import sys
sys.path.append('../')

import librosa
import numpy as np
import scipy
import soundfile as sf
import random
from tqdm.contrib.concurrent import process_map
import torchaudio
import pedalboard as pd
import math

def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result

def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )

#############################
# Augmentations per sample
#############################
def add_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    return noisy_speech, noise


def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    # print(f"speech_sample.shape: {speech_sample.shape}, rir_sample.shape: {rir_sample.shape}")
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]

def add_reverberation_v2(speech_sample, noisy_speech, rir_sample, fs):
    # print(f"speech_sample.shape: {speech_sample.shape}, rir_sample.shape: {rir_sample.shape}")
    rir_wav = rir_sample
    wav_len = speech_sample.shape[1]
    delay_idx = np.argmax(np.abs(rir_wav[0]))  # get the delay index
    delay_before_num = int(0.001 * fs)
    delay_after_num = int(0.005 * fs)
    idx_start = delay_idx - delay_before_num
    idx_end = delay_idx + delay_after_num
    if idx_start < 0:
        idx_start = 0
    early_rir = rir_wav[:, idx_start:idx_end]
    
    reverbant_speech_early = scipy.signal.fftconvolve(speech_sample, early_rir, mode="full")
    reverbant_speech = scipy.signal.fftconvolve(noisy_speech, rir_wav, mode="full")
    
    reverbant_speech = reverbant_speech[:, idx_start:idx_start + wav_len]
    reverbant_speech_early = reverbant_speech_early[:, :wav_len]
    scale = max(abs(reverbant_speech[0]))
    if scale == 0:
        scale = 1
    else:
        scale = 0.5 / scale
    reverbant_speech_early = reverbant_speech_early * scale
    reverbant_speech = reverbant_speech * scale
    return reverbant_speech, reverbant_speech_early


def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    # assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.06, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.
    speech_sample: np.ndarray, a single speech sample (1, Time)
    threshold: float, the threshold for clipping
    """
    
    threshold = random.uniform(min_quantile, max_quantile)
    ret = np.clip(speech_sample, -threshold, threshold)
    
    return ret

# Function used in EQ
def power_ratio(r: float, a: float, b: float):
    return a * math.pow((b / a), r)

def pedalboard_equalizer(wav: np.ndarray, sr: int) -> np.ndarray:
    """Use pedalboard to do equalizer"""
    board = pd.Pedalboard()

    cutoff_low_freq = 60
    cutoff_high_freq = 10000

    q_min = 2
    q_max = 5

    random_all_freq = True
    num_filters = 10
    if random_all_freq:
        key_freqs = [random.uniform(1, 12000) for _ in range(num_filters)]
    else:
        key_freqs = [
            power_ratio(float(z) / (num_filters - 1), cutoff_low_freq, cutoff_high_freq)
            for z in range(num_filters)
        ]
    q_values = [
        power_ratio(random.uniform(0, 1), q_min, q_max) for _ in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]
    # low-shelving filter
    board.append(
        pd.LowShelfFilter(
            cutoff_frequency_hz=key_freqs[0], gain_db=gains[0], q=q_values[0]
        )
    )
    # peaking filters
    for i in range(1, 9):
        board.append(
            pd.PeakFilter(
                cutoff_frequency_hz=key_freqs[i], gain_db=gains[i], q=q_values[i]
            )
        )
    # high-shelving filter
    board.append(
        pd.HighShelfFilter(
            cutoff_frequency_hz=key_freqs[9], gain_db=gains[9], q=q_values[9]
        )
    )

    # Apply the pedalboard to the audio
    processed_audio = board(wav, sr)
    return processed_audio

#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None):
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_
    # # redo using torchaudio
    # audio, fs_ = torchaudio.load(filename, backend="ffmpeg")
    # if force_1ch and audio.shape[0] > 1:
    #     audio = audio.mean(dim=0, keepdim=True)
    # audio = audio.numpy()
    # if fs is not None and fs != fs_:
    #     audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
    #     return audio, fs
    # return audio, fs_


def save_audio(audio, filename, fs):
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    sf.write(filename, audio, samplerate=fs)


#############################
# Main entry
#############################

# # from voicefixer
# snr_min = 5
# snr_max = 40

# p_reverb = 0.25

# p_clipping = 0.25
# clipping_min_quantile = 0.06
# clipping_max_quantile = 0.9

# p_bandwidth_limitation = 0.5
# bandwidth_limitation_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
# bandwidth_limitation_methods = (
#     "kaiser_best",
#     "kaiser_fast",
#     "scipy",
#     "polyphase",
# )

"""
Provided degradation functions:
    - Add noise
    - Add reverb
    - Apply bandwidth limitation
    - Add clipping
======== New Effect ========
    - Apply bitcrush
    - Add chorus
    - Add distortion
    - Apply EQ
    - Add package loss
"""
default_degradation_config = {
    # add noise
    "p_noise": 0.9,
    "snr_min": -5,
    "snr_max": 20,
    # add reverb
    "p_reverb": 0.5,
    # add clipping
    "p_clipping": 0.2,
    "clipping_min_db": -20,
    "clipping_max_db": 0,
    # apply bandwidth limitation
    "p_bandwidth_limitation": 0.2,
    "bandwidth_limitation_rates": [
        4000,
        8000,
        16000,
        22050,
        32000,
    ],
    "quality": [
        0,  # pd.Resample.Quality.ZeroOrderHold,
        1,  # pd.Resample.Quality.Linear,
        2,  # pd.Resample.Quality.CatmullRom,
        3,  # pd.Resample.Quality.Lagrange,
        4   # pd.Resample.Quality.WindowedSinc,
    ],
    # Apply bitcrush
    "p_bitcrush": 0.0,
    "bitcrush_min_bits": 3,
    "bitcrush_max_bits": 8,
    # Add chorus
    "p_chorus": 0.05,
    "rate_hz": 1.0,
    "depth": 0.25,
    "centre_delay_ms": 7.0,
    "feedback": 0.0,
    "chorus_mix": 0.5,
    # Add distortion
    "p_distortion": 0.05,
    "distortion_min_db": 5,
    "distortion_max_db": 20,
    # EQ
    "p_eq": 0.1,
    "eq_min_times": 1,
    "eq_max_times": 3,
    "eq_min_length": 0.5,
    "eq_max_length": 1,
    # package loss
    "p_pl": 0.05,
    "pl_min_ratio": 0.05,
    "pl_max_ratio": 0.1,
    "pl_min_length": 0.05,
    "pl_max_length": 0.1,
}

# def pad_or_truncate(audio, length):
#     if audio.shape[1] < length:
#         # repeat the audio
#         n_repeats = length // audio.shape[1] + 1
#         audio = np.tile(audio, (1, n_repeats))[:, :length]
#     elif audio.shape[1] > length:
#         audio = audio[:, :length]
#     return audio

def process_from_audio_path(
    vocal_path,
    noise_path,
    rir_path=None,
    fs=None,
    force_1ch=True,
    degradation_config=default_degradation_config,
    length=None,
):
    if fs is None:
        fs = sf.info(vocal_path).samplerate

    vocal, _ = read_audio(vocal_path, force_1ch=force_1ch, fs=fs)
    noise, _ = read_audio(noise_path, force_1ch=force_1ch, fs=fs)
    # if length is not None:
    #     vocal = pad_or_truncate(vocal, length)
    noisy_vocal = vocal.copy()

    # add package loss
    if random.random() < degradation_config["p_pl"]:
        # print('add pl')
        # 随机将一些帧替换为空帧
        # 定义要丢失的帧数比例
        replace_ratio = random.uniform(
            degradation_config["pl_min_ratio"], degradation_config["pl_max_ratio"]
        )

        # 计算要丢失的总帧数
        total_frames_to_replace = int(vocal.shape[1] * replace_ratio)

        # 初始化一个空的集合来存储要丢失的帧的索引
        replace_indices = set()

        # 随机生成丢失帧的起始位置和长度
        while len(replace_indices) < total_frames_to_replace:
            start_index = np.random.randint(0, vocal.shape[1])
            length = np.random.randint(fs * degradation_config["pl_min_length"], fs * degradation_config["pl_max_length"])
            end_index = min(start_index + length, vocal.shape[1])
            replace_indices.update(range(start_index, end_index))

        # 将选定的帧替换为空帧（假设空帧为零）
        noisy_vocal[0, list(replace_indices)] = 0

    # Apply EQ
    if random.random() < degradation_config["p_eq"]:
        # print('add eq')
        # 随机确定均衡器处理的次数
        num_eq = random.randint(degradation_config['eq_min_times'], degradation_config['eq_max_times'])

        processed_audio = np.array([])  # 初始化处理后音频的空数组
        last_end_sample = 0  # 跟踪上次处理后的结束位置

        for _ in range(num_eq):
            # 随机确定当前处理段的长度（秒）
            segment_duration = random.uniform(degradation_config['eq_min_length'], degradation_config['eq_max_length'])
            segment_length = int(segment_duration * fs)

            # print(last_end_sample, noisy_vocal.shape[-1], segment_length)
            # 随机选择当前处理段的起始位置
            start_sample = random.randint(last_end_sample, max(last_end_sample, noisy_vocal.shape[-1] - segment_length))
            # print("start_sample: ", start_sample)

            # 确定结束位置
            end_sample = min(start_sample + segment_length, noisy_vocal.shape[-1])

            # 从音频中切割该段
            segment = noisy_vocal[:, start_sample:end_sample]

            # 应用均衡器处理
            processed_segment = pedalboard_equalizer(segment, fs)

            # 如果是第一次处理，直接赋值，否则合并
            if processed_audio.size == 0:
                processed_audio = np.concatenate((noisy_vocal[:, :start_sample], processed_segment), axis=-1)
            else:
                processed_audio = np.concatenate((processed_audio, noisy_vocal[:, last_end_sample:start_sample], processed_segment), axis=-1)

            # 更新上次处理的结束位置
            last_end_sample = end_sample
            if last_end_sample >= noisy_vocal.shape[-1]:
                break

        # 处理完毕后，将最后一段未处理的音频添加到末尾
        if last_end_sample < noisy_vocal.shape[-1]:
            processed_audio = np.concatenate((processed_audio, noisy_vocal[:, last_end_sample:]), axis=-1)
        
        noisy_vocal = processed_audio

    # add reverb
    if rir_path is not None and random.random() < degradation_config["p_reverb"]:
        # print('add reverb')
        rir_sample = read_audio(rir_path, force_1ch=force_1ch, fs=fs)[0]
        noisy_vocal, vocal = add_reverberation_v2(vocal, noisy_vocal, rir_sample, fs)

    # Apply chorus
    if random.random() < degradation_config["p_chorus"]:
        # print('add chorus')
        noisy_vocal = pd.Chorus(
            rate_hz=random.uniform(0.1, 3.0),
            depth=random.uniform(0, 1),
            centre_delay_ms=float(random.randint(1, 30)),
            feedback=random.uniform(-0.5, 0.5),
            mix=random.uniform(0.4, 0.6),
        )(noisy_vocal, fs)

    # add noise
    if random.random() < degradation_config["p_noise"]:
        # print('add noise')
        snr = random.uniform(degradation_config["snr_min"], degradation_config["snr_max"])
        noisy_vocal, noise_sample = add_noise(noisy_vocal, noise, snr=snr, rng=np.random.default_rng())

    if random.random() < degradation_config["p_bitcrush"]:
        # print('add bitcrush')
        noisy_vocal = pd.Bitcrush(
            random.randint(
                degradation_config["bitcrush_min_bits"],
                degradation_config["bitcrush_max_bits"],
            )
        )(noisy_vocal, fs)

    if random.random() < degradation_config["p_clipping"]:
        # print('add clipping')
        noisy_vocal = pd.Clipping(
            random.uniform(
                degradation_config["clipping_min_db"],
                degradation_config["clipping_max_db"],
            )  # in decibels (about 0.1-1 of 1.0)
        )(noisy_vocal, fs)

    if random.random() < degradation_config["p_distortion"]:
        # print('add distortion')
        noisy_vocal = pd.Distortion(
            drive_db=float(random.randint(degradation_config["distortion_min_db"], degradation_config["distortion_max_db"]))
        )(noisy_vocal, fs)

    if random.random() < degradation_config["p_bandwidth_limitation"]:
        # print('add bandwidth limitation')
        fs_new = random.choice(degradation_config["bandwidth_limitation_rates"])
        res_type = random.choice(degradation_config["bandwidth_limitation_methods"])
        noisy_vocal = bandwidth_limitation(noisy_vocal, fs=fs, fs_new=fs_new, res_type=res_type)
        # tgt_fs = random.choice(degradation_config["bandwidth_limitation_rates"])
        # noisy_vocal = pd.Resample(
        #     target_sample_rate=tgt_fs,
        #     quality=pd.Resample.Quality(random.randint(0, 4))
        # )(noisy_vocal, fs)

    # normalization
    scale = 1 / max(
        np.max(np.abs(noisy_vocal)),
        np.max(np.abs(vocal)),
        np.max(np.abs(noise)),
    )

    vocal *= scale
    noise *= scale
    noisy_vocal *= scale

    return vocal, noise, noisy_vocal, fs
    
if __name__ == "__main__":
    # speech = './test_audio/p232_006.wav'
    # noise = './test_audio/p234_001_44.wav'
    # rir = './test_audio/rir_2.wav'
    
    # speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(speech, noise, rir)
    
    # save_audio(speech_sample, './test_audio/output/speech_sample.wav', fs)
    # save_audio(noise_sample, './test_audio/output/noise_sample.wav', fs)
    # save_audio(noisy_speech, './test_audio/output/noisy_speech.wav', fs)
    
    import os
    import glob
    def get_list(speech_list, noise_list, rir_list, root):
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
    
    speech_list = "/mnt/data2/zhangjunan/enhancement/data/voicefixer/processed_noise_rir/noise.scp"
    noise_list = "/mnt/data2/zhangjunan/enhancement/data/voicefixer/processed_noise_rir/noise.scp"
    rir_list = "/mnt/data2/zhangjunan/enhancement/data/voicefixer/processed_noise_rir/rir.scp"
    
    speech_list, noise_list, rir_list = get_list(speech_list, noise_list, rir_list, '/mnt/data2/zhangjunan/urgent2024_challenge/')
    src = '/mnt/data3/share/svc-data/vocalist'
    speech_list = glob.glob(os.path.join(src, '**/*.wav'), recursive=True)
    
    degradation_config = {
        # add noise
        "p_noise": 0.9,
        "snr_min": -5,
        "snr_max": 20,
        # add reverb
        "p_reverb": 0.5,
        # add clipping
        "p_clipping": 0.2,
        "clipping_min_db": -20,
        "clipping_max_db": 0,
        # apply bandwidth limitation
        "p_bandwidth_limitation": 0.2,
        "bandwidth_limitation_rates": [
            4000,
            8000,
            16000,
            22050,
            32000,
        ],
        "quality": [
            0,  # pd.Resample.Quality.ZeroOrderHold,
            1,  # pd.Resample.Quality.Linear,
            2,  # pd.Resample.Quality.CatmullRom,
            3,  # pd.Resample.Quality.Lagrange,
            4   # pd.Resample.Quality.WindowedSinc,
        ],
        # Apply bitcrush
        "p_bitcrush": 0.0,
        "bitcrush_min_bits": 3,
        "bitcrush_max_bits": 8,
        # Add chorus
        "p_chorus": 0.05,
        "rate_hz": 1.0,
        "depth": 0.25,
        "centre_delay_ms": 7.0,
        "feedback": 0.0,
        "chorus_mix": 0.5,
        # Add distortion
        "p_distortion": 0.05,
        "distortion_min_db": 5,
        "distortion_max_db": 20,
        # EQ
        "p_eq": 0.1,
        "eq_min_times": 1,
        "eq_max_times": 3,
        "eq_min_length": 0.5,
        "eq_max_length": 1,
        # package loss
        "p_pl": 0.05,
        "pl_min_ratio": 0.05,
        "pl_max_ratio": 0.1,
        "pl_min_length": 0.05,
        "pl_max_length": 0.1,
    }
    
    import random
    from tqdm import tqdm
    speech_paths = random.sample(speech_list, 200)
    dst = '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen_new'
    import shutil
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    
    os.makedirs(os.path.join(dst, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'noisy'), exist_ok=True)
    for speech_path in tqdm(speech_paths):
        noise_path = random.choice(noise_list)
        rir_path = random.choice(rir_list) if rir_list else None
        # print(f"speech_path: {speech_path}, noise_path: {noise_path}, rir_path: {rir_path}")
        
        speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(
            vocal_path=speech_path, 
            noise_path=noise_path, 
            rir_path=rir_path, 
            fs=44100, 
            force_1ch=True,
            degradation_config=degradation_config
        )
        filename = '-'.join(speech_path.split('/')[-3:])
        save_audio(speech_sample, os.path.join(dst, 'clean', filename), fs)
        save_audio(noisy_speech, os.path.join(dst, 'noisy', filename), fs)
    