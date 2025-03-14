import os
import librosa
import jams


def load_audio_and_annotations(data_dir):
    """
    Wczytuje pliki audio i adnotacje z folderu GuitarSet.

    :param data_dir: Ścieżka do folderu z danymi.
    :return: Lista plików audio, lista plików adnotacji, sygnał audio, częstotliwość próbkowania, obiekt JAMS.
    """
    # Wczytanie listy plików audio i adnotacji
    audio_files = [f for f in os.listdir(os.path.join(data_dir, 'audio_mono-mic')) if f.endswith('.wav')]
    jams_files = [f for f in os.listdir(os.path.join(data_dir, 'annotation')) if f.endswith('.jams')]

    # Wczytanie pierwszego pliku audio i adnotacji
    audio_file = audio_files[0]
    jams_file = jams_files[0]

    # Wczytanie pliku audio
    audio_path = os.path.join(data_dir, 'audio_mono-mic', audio_file)
    y, sr = librosa.load(audio_path, sr=16000)  # Resampling do 16 kHz

    # Wczytanie adnotacji
    jams_path = os.path.join(data_dir, 'annotation', jams_file)
    jam = jams.load(jams_path)

    return audio_files, jams_files, y, sr, jam