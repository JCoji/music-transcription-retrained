
from load_data import setup_guitarset, load_audio_and_annotations
from src.vizualization import create_spectrogram, visualize_example

if __name__ == "__main__":
    # 1. Inicjalizacja GuitarSet
    guitarset = setup_guitarset()

    # 2. Wczytaj pierwsze nagranie (zmień track_id dla innych)
    track_id = guitarset.track_ids[0]
    audio, sr, onsets, pitches = load_audio_and_annotations(track_id, guitarset)
    print(f"Nagranie: {track_id}, długość: {len(audio) / sr:.2f}s, liczba onsetów: {len(onsets)}")

    # 3. Przetwarzanie
    spectrogram = create_spectrogram(audio, sr)

    # 4. Weryfikacja
    print(f"Kształt spectrogramu: {spectrogram.shape} (mels x frames)")

    # 5. Wizualizacja
    # visualize_example(audio, sr, spectrogram, onsets, pitches)