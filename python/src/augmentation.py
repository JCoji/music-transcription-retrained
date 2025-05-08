import torch
import torchaudio
import random

def augment_audio(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if random.random() < 0.5:
        n_steps = random.randint(-3, 3)
        pitch_shift = torchaudio.transforms.PitchShift(
            sample_rate=sample_rate,
            n_steps=n_steps
        )
        audio = pitch_shift(audio)

    if random.random() < 0.5:
        stretch_factor = random.uniform(0.9, 1.1)
        effects = [
            ["tempo", f"{stretch_factor}"]
        ]
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio.unsqueeze(0), sample_rate, effects)
        audio = audio.squeeze(0)

    if random.random() < 0.3:
        noise_amp = 0.005 * torch.rand(1)
        audio = audio + noise_amp * torch.randn_like(audio)

    return audio
