import librosa
import numpy as np
import torch
from nnAudio.features import CQT1992v2

# Import konfiguracji
from src.config import (
    FFT_HOP,
    N_FREQ_BINS_NOTES,
    NOTES_BINS_PER_SEMITONE,
    AUDIO_SAMPLE_RATE,
    GUITAR_BASE_FREQUENCY,
)

# Globalny obiekt transformaty CQT, wstępnie skonfigurowany i przeniesiony na odpowiednie urządzenie.
# Używa parametrów z pliku konfiguracyjnego src.config.
cqt_transform = CQT1992v2(
    sr=AUDIO_SAMPLE_RATE,
    hop_length=FFT_HOP,
    fmin=GUITAR_BASE_FREQUENCY,
    n_bins=N_FREQ_BINS_NOTES,
    bins_per_octave=12
    * NOTES_BINS_PER_SEMITONE,  # Zapewnia odpowiednią rozdzielczość częstotliwościową
    verbose=False,  # Wyłącza dodatkowe komunikaty z nnAudio
    output_format="Magnitude",  # Zwraca magnitudę transformaty
).to("cuda" if torch.cuda.is_available() else "cpu")


def compute_cqt(audio: np.ndarray) -> torch.Tensor:
    """
    Oblicza spektrogram CQT (Constant-Q Transform) dla danego sygnału audio.

    Funkcja wykorzystuje prekonfigurowany globalny obiekt `cqt_transform`.
    Wynik jest konwertowany do skali decybelowej.

    Args:
        audio (np.ndarray): Sygnał audio jako tablica NumPy.

    Returns:
        torch.Tensor: Obliczony spektrogram CQT w skali dB jako tensor PyTorch,
                      z transponowanymi wymiarami (czas, częstotliwość).
    """
    # Określenie urządzenia (GPU lub CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Konwersja audio NumPy do tensora PyTorch i dodanie wymiaru batcha
    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    # Obliczenie transformaty CQT przy użyciu globalnego obiektu
    cqt_magnitude = cqt_transform(audio_tensor)

    # Konwersja wyniku do NumPy w celu użycia funkcji librosa
    cqt_magnitude_numpy = cqt_magnitude.squeeze(0).cpu().numpy()

    # Konwersja magnitudy do skali decybelowej
    cqt_db_numpy = librosa.amplitude_to_db(cqt_magnitude_numpy, ref=np.max)

    # Konwersja wyniku z powrotem do tensora PyTorch
    cqt_db_tensor = torch.tensor(cqt_db_numpy, dtype=torch.float32, device=device)

    # Transpozycja wymiarów, aby uzyskać (czas, częstotliwość)
    return cqt_db_tensor.T
