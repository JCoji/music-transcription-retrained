import librosa
import numpy as np
import torch
from nnAudio.features import CQT1992v2

from src.config import (
    FFT_HOP,
    N_FREQ_BINS_NOTES,
    NOTES_BINS_PER_SEMITONE,
    AUDIO_SAMPLE_RATE, GUITAR_BASE_FREQUENCY,
)

cqt_transform = CQT1992v2(
    sr=AUDIO_SAMPLE_RATE,
    hop_length=FFT_HOP,
    fmin=GUITAR_BASE_FREQUENCY,
    n_bins=N_FREQ_BINS_NOTES,
    bins_per_octave=12 * NOTES_BINS_PER_SEMITONE,
    verbose=False,
    output_format="Magnitude",
).to("cuda" if torch.cuda.is_available() else "cpu")


def compute_cqt(audio: np.ndarray) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    cqt = cqt_transform(audio_tensor)
    cqt_numpy = cqt.squeeze(0).cpu().numpy()
    cqt_db_numpy = librosa.amplitude_to_db(cqt_numpy, ref=np.max)
    cqt_db = torch.tensor(cqt_db_numpy, dtype=torch.float32, device=device)
    return cqt_db.T
