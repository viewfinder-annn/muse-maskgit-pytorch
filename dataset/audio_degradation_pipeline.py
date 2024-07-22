# modified from urgent challenge https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/simulate_data_from_param.py
import sys
sys.path.append('../')

import re
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import scipy
import soundfile as sf
import random
from tqdm.contrib.concurrent import process_map
import torchaudio

from dataset.rir_utils import estimate_early_rir

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
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]


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

default_degradation_config = {
    "p_noise": 1.0,
    "snr_min": -5,
    "snr_max": 40,
    
    "p_reverb": 0.25,
    
    "p_clipping": 0.25,
    "clipping_min_quantile": 0.06,
    "clipping_max_quantile": 0.9,
    
    "p_bandwidth_limitation": 0.5,
    "bandwidth_limitation_rates": [
        8000,
        16000,
        22050,
        24000,
        32000,
        44100,
        48000,
    ],
    "bandwidth_limitation_methods": [
        "kaiser_best",
        "kaiser_fast",
        "scipy",
        "polyphase",
    ],
}

def process_from_audio_path(speech, noise, rir=None, fs=None, force_1ch=True, degradation_config=default_degradation_config): 
    if fs is None:
        fs = sf.info(speech).samplerate
    
    speech_sample = read_audio(speech, force_1ch=force_1ch, fs=fs)[0]
    noise_sample = read_audio(noise, force_1ch=force_1ch, fs=fs)[0]
    
    # add noise
    if random.random() < degradation_config["p_noise"]:
        snr = random.uniform(degradation_config["snr_min"], degradation_config["snr_max"])
        noisy_speech, noise_sample = add_noise(speech_sample, noise_sample, snr=snr, rng=np.random.default_rng())
    else:
        noisy_speech = speech_sample
    
    # add reverb
    if rir is not None and random.random() < degradation_config["p_reverb"]:
        rir_sample = read_audio(rir, force_1ch=force_1ch, fs=fs)[0]
        noisy_speech = add_reverberation(noisy_speech, rir_sample)
        early_rir_sample = estimate_early_rir(rir_sample, fs=fs)
        speech_sample = add_reverberation(speech_sample, early_rir_sample)
    # else:
    #     noisy_speech = speech_sample
    
    # # add noise
    # snr = random.uniform(snr_min, snr_max)
    # noisy_speech, noise_sample = add_noise(noisy_speech, noise_sample, snr=snr, rng=np.random.default_rng())
    
    # add clipping
    if random.random() < degradation_config["p_clipping"]:
        noisy_speech = clipping(noisy_speech, min_quantile=degradation_config["clipping_min_quantile"], max_quantile=degradation_config["clipping_max_quantile"])
    
    # add bandwidth limitation
    if random.random() < degradation_config["p_bandwidth_limitation"]:
        fs_new = random.choice(degradation_config["bandwidth_limitation_rates"])
        res_type = random.choice(degradation_config["bandwidth_limitation_methods"])
        noisy_speech = bandwidth_limitation(noisy_speech, fs=fs, fs_new=fs_new, res_type=res_type)
    
    # normalization
    scale = 1 / max(
        np.max(np.abs(noisy_speech)),
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise_sample)),
    )

    speech_sample *= scale
    noise_sample *= scale
    noisy_speech *= scale
    
    return speech_sample, noise_sample, noisy_speech, fs
    
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
            "p_noise": 1.0,
            "snr_min": -5,
            "snr_max": 20,
            
            "p_reverb": 0.5,
            
            "p_clipping": 0.25,
            "clipping_min_quantile": 0.06,
            "clipping_max_quantile": 0.9,
            
            "p_bandwidth_limitation": 0.5,
            "bandwidth_limitation_rates": [
                1000,
                2000,
                4000,
                8000,
                16000,
                22050,
                32000,
            ],
            "bandwidth_limitation_methods": [
                "kaiser_best",
                "kaiser_fast",
                "scipy",
                "polyphase",
            ],
        }
    
    import random
    from tqdm import tqdm
    speech_paths = random.sample(speech_list, 200)
    dst = '/mnt/data2/zhangjunan/enhancement/data/singing_scp/testset_unseen'
    
    os.makedirs(os.path.join(dst, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'noisy'), exist_ok=True)
    for speech_path in tqdm(speech_paths):
        noise_path = random.choice(noise_list)
        rir_path = random.choice(rir_list) if rir_list else None
        # print(f"speech_path: {speech_path}, noise_path: {noise_path}, rir_path: {rir_path}")
        
        speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(
            speech=speech_path, 
            noise=noise_path, 
            rir=rir_path, 
            fs=44100, 
            force_1ch=True,
            degradation_config=degradation_config
        )
        filename = '-'.join(speech_path.split('/')[-3:])
        save_audio(speech_sample, os.path.join(dst, 'clean', filename), fs)
        save_audio(noisy_speech, os.path.join(dst, 'noisy', filename), fs)
    