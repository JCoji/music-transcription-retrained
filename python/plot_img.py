import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- Konfiguracja ---
# Użyj dokładnej ścieżki do Twojego pliku audio
AUDIO_FILE_PATH = r'C:\Users\lukig\OneDrive\Dokumenty\Programming\music-transcription\python\_mir_datasets_storage\audio_mono-pickup_mix\00_BN1-129-Eb_comp_mix.wav'
FIG_SAVE_PATH = 'rysunek_spektrogram_mel_z_pliku_bez_tytulu.png'

# --- Parametry Przetwarzania Audio ---
SR_TARGET = 22050
HOP_LENGTH_GLOBAL = 512

# Parametry dla Mel-spektrogramu
N_FFT_MEL = 2048
N_MELS = 128

# --- Ustawienia globalne rozmiaru czcionek ---
plt.rc('font', size=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=14)

# --- Wczytanie sygnału audio ---
try:
    y, sr = librosa.load(AUDIO_FILE_PATH, sr=SR_TARGET, duration=5) # Wczytaj np. pierwsze 5 sekund
    print(f"Wczytano plik audio: {AUDIO_FILE_PATH}, SR: {sr} Hz, Długość: {len(y)/sr:.2f}s")
    if len(y) == 0:
        print("BŁĄD: Wczytany sygnał audio jest pusty. Skrypt może nie działać poprawnie.")
        # Można tu dodać awaryjne wyjście lub wygenerowanie sygnału testowego
        # exit()
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku audio: {AUDIO_FILE_PATH}")
    print("Upewnij się, że ścieżka do pliku jest poprawna.")
    exit() # Zakończ program, jeśli plik nie zostanie znaleziony
except Exception as e:
    print(f"Błąd podczas wczytywania pliku: {e}")
    exit()

if len(y) == 0:
    print("BŁĄD: Wczytany sygnał audio jest pusty po załadowaniu. Zakończenie programu.")
    exit()

# --- Obliczanie Mel-spektrogramu ---
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT_MEL, hop_length=HOP_LENGTH_GLOBAL, n_mels=N_MELS)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)


# --- Tworzenie rysunku Mel-spektrogramu (bez tytułu na rysunku) ---
fig, ax = plt.subplots(figsize=(12, 6))

img_mel = librosa.display.specshow(S_mel_db, sr=sr, hop_length=HOP_LENGTH_GLOBAL, x_axis='time', y_axis='mel', ax=ax)

# Tytuł nie jest ustawiany na rysunku
# ax.set_title("...")

ax.set_xlabel('Czas [s]')
ax.set_ylabel('Częstotliwość [Mel]')

cb_mel = fig.colorbar(img_mel, ax=ax, format='%+2.0f dB')
cb_mel.ax.tick_params(labelsize=12)

plt.tight_layout(pad=1.0)

# Zapis rysunku do pliku
try:
    plt.savefig(FIG_SAVE_PATH, dpi=300)
    print(f"Rysunek został zapisany do pliku: {FIG_SAVE_PATH}")
except Exception as e:
    print(f"Nie udało się zapisać rysunku: {e}")

plt.show()