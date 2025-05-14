import numpy as np

# Konfiguracja audio
AUDIO_SAMPLE_RATE = 22050  # Częstotliwość próbkowania audio w Hz
N_CHANNELS = 1  # Liczba kanałów audio
FFT_HOP = 512  # Przesunięcie okna FFT (hop size) w próbkach

# Konfiguracja adnotacji
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP  # Liczba klatek adnotacji na sekundę
ANNOTATION_HOP = (
    1.0 / ANNOTATIONS_FPS
)  # Czas trwania jednej klatki adnotacji w sekundach

# Zakres gitary
GUITAR_BASE_FREQUENCY = 82.41  # Bazowa częstotliwość gitary w Hz (np. E2)
GUITAR_N_SEMITONES = 64  # Liczba półtonów w zakresie gitary

# Podział na biny (koszyki częstotliwości)
NOTES_BINS_PER_SEMITONE = 3  # Liczba binów na półton dla detekcji nut
CONTOURS_BINS_PER_SEMITONE = 3  # Liczba binów na półton dla detekcji konturów


def _freq_bins(bins_per_semitone, base_frequency, n_semitones):
    """
    Oblicza środkowe częstotliwości dla każdego binu.

    Biny są rozmieszczone logarytmicznie, zgodnie ze skalą równomiernie temperowaną,
    z określoną liczbą binów na półton.

    Args:
        bins_per_semitone (int): Liczba binów na każdy półton.
        base_frequency (float): Bazowa częstotliwość (najniższa) w Hz.
        n_semitones (int): Całkowita liczba półtonów do pokrycia.

    Returns:
        np.ndarray: Tablica zawierająca środkowe częstotliwości dla każdego binu.
    """
    # Współczynnik skalujący dla kolejnych binów w ramach systemu równomiernie temperowanego
    # d odpowiada pierwiastkowi (12 * bins_per_semitone) stopnia z 2
    d = 2.0 ** (1.0 / (12 * bins_per_semitone))
    # Generowanie częstotliwości dla wszystkich binów
    return base_frequency * d ** np.arange(bins_per_semitone * n_semitones)


# Obliczone biny częstotliwości dla nut
FREQ_BINS_NOTES = _freq_bins(
    NOTES_BINS_PER_SEMITONE, GUITAR_BASE_FREQUENCY, GUITAR_N_SEMITONES
)
# Obliczone biny częstotliwości dla konturów
FREQ_BINS_CONTOURS = _freq_bins(
    CONTOURS_BINS_PER_SEMITONE, GUITAR_BASE_FREQUENCY, GUITAR_N_SEMITONES
)

# Całkowita liczba binów częstotliwości dla nut
N_FREQ_BINS_NOTES = GUITAR_N_SEMITONES * NOTES_BINS_PER_SEMITONE
# Całkowita liczba binów częstotliwości dla konturów
N_FREQ_BINS_CONTOURS = GUITAR_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
