from demucs import pretrained
from demucs.apply import apply_model
from torch.utils.data import Dataset
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import torchaudio
import librosa
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

from pathlib import Path
import soundfile
import av
import numpy as np
import logging
from dataclasses import dataclass

@dataclass(frozen=True)
class AudioFileInfo:
    sample_rate: int
    duration: float
    channels: int

_av_initialized = False

def _init_av():
    global _av_initialized
    if _av_initialized:
        return
    logger = logging.getLogger('libav.mp3')
    logger.setLevel(logging.ERROR)
    _av_initialized = True

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def _av_read(filepath, seek_time: float = 0, duration: float = -1.):
    """FFMPEG-based audio file reading using PyAV bindings.
    Soundfile cannot read mp3 and av_read is more efficient than torchaudio.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate
    """
    _init_av()
    with av.open(str(filepath)) as af:
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        num_frames = int(sr * duration) if duration >= 0 else -1
        frame_offset = int(sr * seek_time)
        # we need a small negative offset otherwise we get some edge artifact
        # from the mp3 decoder.
        af.seek(int(max(0, (seek_time - 0.1)) / stream.time_base), stream=stream)
        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            current_offset = int(frame.rate * frame.pts * frame.time_base)
            strip = max(0, frame_offset - current_offset)
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != stream.channels:
                buf = buf.view(-1, stream.channels).t()
            buf = buf[:, strip:]
            frames.append(buf)
            length += buf.shape[1]
            if num_frames > 0 and length >= num_frames:
                break
        # assert frames
        # If the above assert fails, it is likely because we seeked past the end of file point,
        # in which case ffmpeg returns a single frame with only zeros, and a weird timestamp.
        # This will need proper debugging, in due time.
        if not frames:
            # return empty tensor if no frames were read
            return torch.zeros(stream.channels, 0), sr
        wav = torch.cat(frames, dim=1)
        assert wav.shape[0] == stream.channels
        if num_frames > 0:
            wav = wav[:, :num_frames]
        return f32_pcm(wav), sr

def _soundfile_info(filepath):
    info = soundfile.info(filepath)
    return AudioFileInfo(info.samplerate, info.duration, info.channels)

def audio_read(filepath, seek_time: float = 0.,
               duration: float = -1., pad: bool = False):
    """Read audio by picking the most appropriate backend tool based on the audio format.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
        pad (bool): Pad output audio if not reaching expected duration.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate.
    """
    fp = Path(filepath)
    if fp.suffix in ['.flac', '.ogg']:  # TODO: check if we can safely use av_read for .ogg
        # There is some bug with ffmpeg and reading flac
        info = _soundfile_info(filepath)
        frames = -1 if duration <= 0 else int(duration * info.sample_rate)
        frame_offset = int(seek_time * info.sample_rate)
        wav, sr = soundfile.read(filepath, start=frame_offset, frames=frames, dtype=np.float32)
        assert info.sample_rate == sr, f"Mismatch of sample rates {info.sample_rate} {sr}"
        wav = torch.from_numpy(wav).t().contiguous()
        if len(wav.shape) == 1:
            wav = torch.unsqueeze(wav, 0)
    else:
        wav, sr = _av_read(filepath, seek_time, duration)
    if pad and duration > 0:
        expected_frames = int(duration * sr)
        wav = torch.nn.functional.pad(wav, (0, expected_frames - wav.shape[-1]))
    return wav, sr

class Sing2SongDataset(Dataset):
    def __init__(self, metadatas, seq_len=10, sr=44100, device='cpu', num_samples=-1):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Sing2SongDataset.__init__()")
        
        self.metadata = []
        for metadata in metadatas:
            assert os.path.exists(metadata), f"metadata file {metadata} not found"
            with open(metadata, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata.extend(data)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] metadata loaded, {len(self.metadata)} samples")
        if num_samples != -1:
            import random
            random.shuffle(self.metadata)
            self.metadata = self.metadata[:min(num_samples, len(self.metadata))]
        self.seperator = pretrained.get_model('htdemucs')
        self.seperator.to(device)
        self.seperator.eval()
        self.seq_len = seq_len
        self.device = device
        self.sr = sr
        
        """
        metadata: a list contains dict, each dict contains:
            {
                "self_wav": "/mnt/data2/zhangjunan/sing2song_challenge/netease/mp3/english/Lil Nas X-STAR WALKIN' (League of Legends Worlds Anthem)-STAR WALKIN' (League of Legends Worlds Anthem) 英雄联盟2022全球总决赛主题曲.mp3",
                "lyric": "learned a lesson from the wise you should never take advice from a ni**a\nThat ain’t try\nThey said I  wouldn't make it out alive",
                "start": 49.648,
                "end": 58.919
            },
        """

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Dataset loaded, {len(self.metadata)} samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        utt_info = self.metadata[index]
        
        try:
            # audio_sr = torchaudio.info(utt_info["self_wav"]).sample_rate
            # duration = librosa.get_duration(path=utt_info["self_wav"])
            # if duration <= utt_info["start"]:
            #     utt_info["start"] = duration - self.seq_len
            self_wav, audio_sr = audio_read(utt_info["self_wav"])
        except Exception as e:
            print(f"Error: {e}, {utt_info['self_wav']}, {utt_info['start']}, {utt_info['end']}")
            exit(1)
        # print(f"audio_sr: {audio_sr}")
        # print(f"self_wav: {self_wav.shape}")
        if self_wav.dim() == 1:
            self_wav = self_wav.unsqueeze(0)
        if self.seq_len*audio_sr > self_wav.shape[-1]:
            self_wav = torch.nn.functional.pad(self_wav, (0, self.seq_len*audio_sr - self_wav.shape[-1]))
        self_wav = self_wav[..., :self.seq_len*audio_sr] # [channels, seq_len*sr]
        res = apply_model(self.seperator, self_wav.unsqueeze(0), device=self.device) # [1,4,2,duration*sr] 
        res.squeeze_(0)
        ref_wav = res[-1]
        self_wav = res[:-1].sum(0)
        if audio_sr != self.sr:
            # print(ref_wav.shape, self_wav.shape)
            resample = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=self.sr).to(self.device)
            self_wav = resample(self_wav.to(self.device))
            ref_wav = resample(ref_wav.to(self.device))
        
        # # add noise to ref_wav(vocal)
        # noise = torch.randn_like(self_wav)
        # # -45dB
        # noise = noise * 10**(-45/20)
        # ref_wav = ref_wav + noise
        
        # mono
        # print(self_wav.shape, ref_wav.shape)
        if len(self_wav.shape) == 2:
            # keepdim = True
            self_wav = self_wav.mean(0)
        if len(ref_wav.shape) == 2:
            ref_wav = ref_wav.mean(0)
        
        return self_wav.to(self.device), ref_wav.to(self.device)


    def __len__(self):
        return len(self.metadata)


if __name__ == "__main__":
    dataset = Sing2SongDataset(["/mnt/data2/zhangjunan/sing2song_challenge/netease/dataset/chinese_short/test.json", "/mnt/data2/zhangjunan/sing2song_challenge/netease/dataset/english_short/test.json", ], num_samples=20, seq_len=10, sr=32000)
    dst = './test_audio/testset'
    print(len(dataset))
    for i in range(len(dataset)):
        print(f"processing {i}")
        print(dataset[i][0].shape, dataset[i][1].shape) # [seq_len*sr], [seq_len*sr]
        # torchaudio.save(f"{dst}/test_{i}_self.wav", dataset[i][0], 32000)
        torchaudio.save(f"{dst}/test_{i}_ref.wav", dataset[i][1].unsqueeze(0), 32000)
    print("done")