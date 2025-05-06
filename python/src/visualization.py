import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_audio_waveform(audio, sr):
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Sygnał audio")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cqt(cqt, sr, hop_length, freq_bins):
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(
        cqt.T,  # teraz cqt.shape = (T, F) → .T = (F, T) → specshow oczekuje (Y, X)
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='log',
        y_coords=freq_bins  # Oś Y = freq_bins (ma mieć długość F)
    )
    plt.title("CQT - Constant-Q Transform")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Amplituda [dB]')
    plt.tight_layout()
    plt.show()


def plot_annotation_map(annotation_map, title, ylabel, cmap='hot'):
    plt.figure(figsize=(14, 4))
    plt.imshow(annotation_map.T, aspect='auto', origin='lower',
               cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Ramy czasowe")
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def create_annotation_maps(sample):
    notes_map = torch.zeros(sample["shape_notes"])
    contours_map = torch.zeros(sample["shape_contours"])
    notes_map[sample["note_indices"][:, 0], sample["note_indices"][:, 1]] = 1
    contours_map[sample["contour_indices"][:, 0], sample["contour_indices"][:, 1]] = 1
    return notes_map, contours_map
