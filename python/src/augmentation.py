import torch
import random
import torchaudio


def augment_audio(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if random.random() < 0.5:
        waveform = add_noise(waveform)
    if random.random() < 0.5:
        waveform = pitch_shift(waveform, sample_rate)
    return waveform


def add_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def pitch_shift(waveform: torch.Tensor, sample_rate: int, n_steps_range=(-2, 2)) -> torch.Tensor:
    n_steps = random.uniform(*n_steps_range)
    return torchaudio.functional.pitch_shift(waveform, sample_rate, n_steps)
