import librosa
import mirdata
import numpy as np
from keras.src.layers import LSTM, BatchNormalization, Dropout, Dense
from keras.src.models import Sequential
from load_data import load_audio_and_annotations
from vizualization import create_spectrogram


def prepare_training_data(spectrograms, onsets, sr=44100, hop_length=512):
    max_frames = max(spec.shape[1] for spec in spectrograms)
    n_mels = spectrograms[0].shape[0]

    X = []
    y = []

    for spec, onset in zip(spectrograms, onsets):
        # Padding spectrogramu
        padded_spec = np.zeros((n_mels, max_frames))
        padded_spec[:, :spec.shape[1]] = spec

        # Binary labels
        frames = librosa.time_to_frames(onset, sr=sr, hop_length=hop_length)
        frames = frames[frames < max_frames]
        labels = np.zeros(max_frames)
        labels[frames] = 1

        X.append(padded_spec.T)
        y.append(labels)

    return np.array(X), np.expand_dims(np.array(y), -1)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model


def train_model():
    guitarset = mirdata.initialize("guitarset")
    track_ids = guitarset.track_ids[:10]  # Tylko 10 nagrań

    all_spectrograms = []
    all_onsets = []

    for track_id in track_ids:
        audio, sr, onsets, _ = load_audio_and_annotations(track_id, guitarset)
        spec = create_spectrogram(audio, sr)
        all_spectrograms.append(spec)
        all_onsets.append(onsets)
        print(f"Przetworzono {track_id}, kształt: {spec.shape}")

    X, y = prepare_training_data(all_spectrograms, all_onsets)
    print(f"Kształty: X={X.shape}, y={y.shape}")

    model = create_lstm_model((X.shape[1], X.shape[2]))
    model.summary()

    model.fit(X, y, epochs=5, batch_size=4, validation_split=0.2)

    return model