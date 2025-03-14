from load_data import load_audio_and_annotations

# Ścieżka do folderu z danymi
data_dir = 'data/guitar-set'

# Wczytanie danych
audio_files, jams_files, y, sr, jam = load_audio_and_annotations(data_dir)

# Wyświetlenie podstawowych informacji
print(f"Wczytano plik audio: {audio_files[0]}")
print(f"Częstotliwość próbkowania: {sr} Hz")
print(f"Długość sygnału: {len(y) / sr:.2f} sekund")

# Wyświetlenie informacji o adnotacjach
print(f"\nAdnotacje dla pliku {jams_files[0]}:")
for annotation in jam.annotations:
    print(f" - {annotation.namespace}: {len(annotation.data)} zdarzeń")