import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(y, sr, title="Sygnał audio"):
    """
    Wizualizuje przebieg sygnału audio w czasie.

    :param y: Sygnał audio.
    :param sr: Częstotliwość próbkowania.
    :param title: Tytuł wykresu.
    """
    # Tworzenie osi czasu
    time = np.arange(len(y)) / sr

    # Wykres sygnału audio
    plt.figure(figsize=(10, 4))
    plt.plot(time, y, alpha=0.8)
    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda")
    plt.title(title)
    plt.grid()
    plt.show()


def plot_onsets(y, sr, onsets, pitches=None, title="Onsety i wysokości nut"):
    """
    Wizualizuje onsety i wysokości nut na osi czasu sygnału audio.

    :param y: Sygnał audio.
    :param sr: Częstotliwość próbkowania.
    :param onsets: Lista onsetów (momentów rozpoczęcia nut).
    :param pitches: Lista wysokości nut (opcjonalnie).
    :param title: Tytuł wykresu.
    """
    # Tworzenie osi czasu
    time = np.arange(len(y)) / sr

    # Wykres sygnału audio
    plt.figure(figsize=(10, 4))
    plt.plot(time, y, alpha=0.6, label="Sygnał audio")

    # Wizualizacja onsetów
    plt.vlines(onsets, -1, 1, color='r', alpha=0.8, label="Onsety")

    # Wizualizacja wysokości nut
    if pitches is not None:
        plt.scatter(onsets, [0] * len(onsets), c=pitches, cmap='viridis', label="Wysokość nut")
        plt.colorbar(label="Wysokość nut (MIDI)")

    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()