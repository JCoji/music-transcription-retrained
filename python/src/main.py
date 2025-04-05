import librosa
import numpy as np
from load_data import setup_guitarset, load_audio_and_annotations
from model import train_model
from vizualization import create_spectrogram, visualize_example, visualize_predictions
import tensorflow as tf


def main(guitarset, model=None):
    # Wybierz plik testowy (np. pierwszy z pominiętych w treningu)
    track_id = guitarset.track_ids[0]  # Możesz zmienić na inny
    audio, sr, true_onsets, pitches = load_audio_and_annotations(track_id, guitarset)

    # Generuj spectrogram
    spectrogram = create_spectrogram(audio, sr)

    # Przygotuj dane dla modelu (tak jak w prepare_training_data)
    frames = librosa.time_to_frames(true_onsets, sr=sr, hop_length=512)
    X_test = []
    seq_length = 128  # Musi być takie samo jak w treningu!
    for i in range(0, spectrogram.shape[1] - seq_length, seq_length // 2):
        X_test.append(spectrogram[:, i:i + seq_length].T)
    X_test = np.array(X_test)

    # Predykcja modelu
    if model:
        y_pred = model.predict(X_test)
        # Konwersja predykcji na onsety (w czasie)
        predicted_frames = []
        for i, seq in enumerate(y_pred):
            for j, prob in enumerate(seq):
                if prob > 0.5:  # Próg można dostosować
                    absolute_frame = i * (seq_length // 2) + j
                    predicted_frames.append(absolute_frame)

        predicted_onsets = librosa.frames_to_time(predicted_frames, sr=sr, hop_length=512)

        # Wizualizacja
        visualize_predictions(audio, sr, spectrogram, true_onsets, predicted_onsets, track_id)
    else:
        visualize_example(audio, sr, spectrogram, true_onsets, pitches)


if __name__ == "__main__":
    guitarset = setup_guitarset()
    model = train_model()
    main(guitarset, model)