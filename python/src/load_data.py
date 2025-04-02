import os
import librosa
import mirdata

from config import Config

def setup_guitarset():
    """Inicjalizuje GuitarSet i pobiera dane, jeśli brak."""
    guitarset = mirdata.initialize("guitarset", data_home=Config.DATA_DIR)
    print(f"Dostępne nagrania: {guitarset.track_ids[:3]}...")

    # Sprawdź czy którykolwiek plik już istnieje
    example_track = guitarset.track(guitarset.track_ids[0])
    if not os.path.exists(example_track.audio_mic_path):
        print("Pobieranie GuitarSet (~8 GB)...")
        guitarset.download()
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
