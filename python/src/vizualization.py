import librosa
import numpy as np
from matplotlib import pyplot as plt
from config import Config


def create_spectrogram(audio, sr):
    """Tworzy log-mel-spectrogram."""
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=Config.N_MELS)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram


def visualize_example(audio, sr, spectrogram, onsets, pitches):
    """Wizualizacja audio, spectrogramu i onsetów."""
    plt.figure(figsize=(12, 8))

    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.vlines(onsets, -1, 1, color="r", linestyle="--", label="Onsety")
    plt.title("Sygnał audio z onsetami")
    plt.legend()

    # Spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        spectrogram, sr=sr, x_axis="time", y_axis="mel", cmap="viridis"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-mel-spectrogram")
    plt.tight_layout()
    plt.show()