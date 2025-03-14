from load_data import load_audio_and_annotations
from visualization import plot_waveform, plot_onsets

# Ścieżka do folderu z danymi
data_dir = '../data/guitar_set'

# Wczytanie danych
audio_files, jams_files, y, sr, jam = load_audio_and_annotations(data_dir)

# Wyświetlenie podstawowych informacji
print(f"Wczytano plik audio: {audio_files[0]}")
print(f"Częstotliwość próbkowania: {sr} Hz")
print(f"Długość sygnału: {len(y) / sr:.2f} sekund")

# Wizualizacja sygnału audio
plot_waveform(y, sr, title=f"Przebieg sygnału audio: {audio_files[0]}")

# Wyodrębnienie onsetów i wysokości nut z adnotacji
onsets = []
pitches = []

for annotation in jam.annotations:
    if annotation.namespace == 'note_midi':
        for note in annotation.data:
            onsets.append(note.time)
            pitches.append(note.value)

# Wizualizacja onsetów i wysokości nut
plot_onsets(y, sr, onsets, pitches, title=f"Onsety i wysokości nut: {audio_files[0]}")