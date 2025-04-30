import librosa
import numpy as np
import torch

from src.config import FFT_HOP, N_FREQ_BINS_NOTES, NOTES_BINS_PER_SEMITONE


def compute_cqt(audio: np.ndarray, sr: int, hop_length: int = FFT_HOP,
                n_bins: int = N_FREQ_BINS_NOTES,
                bins_per_octave: int = 12 * NOTES_BINS_PER_SEMITONE):
    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length,
                      n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return torch.tensor(cqt_db, dtype=torch.float32)
