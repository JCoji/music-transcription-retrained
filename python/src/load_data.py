import os
import librosa
import mirdata

from config import Config


def setup_guitarset():
    """Inicjalizuje GuitarSet i pobiera dane, jeśli brak."""
    # 1. Inicjalizacja datasetu (nawet jeśli indeksu jeszcze nie ma)
    guitarset = mirdata.initialize("guitarset", data_home=Config.DATA_DIR)

    # 2. Sprawdź, czy indeks istnieje (jeśli nie, pobierz go)
    index_path = os.path.join(Config.DATA_DIR, "mirdata-datasets", "guitarset", "index.json")
    if not os.path.exists(index_path):
        print("Pobieranie indeksu GuitarSet...")
        guitarset.download(partial_download=["index"])  # Pobierz tylko indeks

    # 3. Teraz sprawdź pełne dane audio
    example_track = guitarset.track(guitarset.track_ids[0])
    if not os.path.exists(example_track.audio_mic_path):
        print("Pobieranie pełnych danych GuitarSet (~8 GB)...")
        guitarset.download()  # Pobierz wszystko
    else:
        print("Dane GuitarSet już istnieją - pomijam pobieranie")

    return guitarset

def load_audio_and_annotations(track_id, guitarset):
    """Ładuje audio (mono-mic) i adnotacje (onsety, częstotliwości)."""
    track = guitarset.track(track_id)
    audio, sr = librosa.load(track.audio_mic_path, sr=Config.SAMPLE_RATE)

    notes = track.notes_all
    onsets = notes.intervals[:, 0]
    pitches = notes.pitches

    return audio, sr, onsets, pitches
