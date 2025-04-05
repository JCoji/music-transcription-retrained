import librosa
import numpy as np
from matplotlib import pyplot as plt
from config import Config

def create_spectrogram(audio, sr):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=Config.N_MELS)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # Normalizacja do [0, 1]
    log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))
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


def visualize_predictions(audio, sr, spectrogram, true_onsets, predicted_onsets, track_id):
    """Porównanie prawdziwych i przewidzianych onsetów"""
    plt.figure(figsize=(14, 10))

    # Waveform z oboma zestawami onsetów
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.vlines(true_onsets, -1, 1, color="g", linestyle="--", label="True onsets", alpha=0.7)
    plt.vlines(predicted_onsets, -1, 1, color="b", linestyle=":", label="Predicted onsets", alpha=0.7)
    plt.title(f"Audio: {track_id} | Onset comparison")
    plt.legend()

    # Spectrogram z prawdziwymi onsetami
    plt.subplot(3, 1, 2)
    librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.vlines(true_onsets, 0, sr / 2, color="w", linestyle="--", alpha=0.5)
    plt.title("Spectrogram with true onsets")

    # Spectrogram z przewidzianymi onsetami
    plt.subplot(3, 1, 3)
    librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.vlines(predicted_onsets, 0, sr / 2, color="w", linestyle=":", alpha=0.5)
    plt.title("Spectrogram with predicted onsets")

    plt.tight_layout()
    plt.show()