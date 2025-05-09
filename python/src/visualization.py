import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_audio_waveform(audio, sr, onset_frames=None, hop_length=512):
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Sygnał audio")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)

    if onset_frames is not None:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        for t in onset_times:
            plt.axvline(x=t, color='r', linestyle='--', linewidth=1.2, alpha=0.7, label='Onset')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


def plot_cqt(cqt, sr, hop_length, freq_bins):
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(
        cqt.T,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='log',
        y_coords=freq_bins
    )
    plt.title("CQT - Constant-Q Transform")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Amplituda [dB]')
    plt.tight_layout()
    plt.show()

def plot_annotation_map(annotation_map, title, ylabel, cmap='hot', sr=44100, hop_length=512):
    T = annotation_map.shape[1]
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)

    plt.figure(figsize=(14, 4))
    plt.imshow(annotation_map, aspect='auto', origin='lower',
               cmap=cmap, interpolation='nearest',
               extent=[times[0], times[-1], 0, annotation_map.shape[0]])

    plt.title(title)
    plt.xlabel("Czas [s]")
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
