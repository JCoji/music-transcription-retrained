import librosa
import mirdata
import numpy as np
from keras.src.layers import LSTM, BatchNormalization, Dropout, Dense
from keras.src.models import Sequential
from load_data import load_audio_and_annotations
from config import Config
from vizualization import create_spectrogram
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf


def prepare_training_data(spectrograms, onsets, sr=44100, hop_length=512, seq_length=128):
    X, y = [], []

    for spec, onset in zip(spectrograms, onsets):
        # Konwersja onsetów na ramki
        frames = librosa.time_to_frames(onset, sr=sr, hop_length=hop_length)
        labels = np.zeros(spec.shape[1])
        labels[frames[frames < spec.shape[1]]] = 1  # Binary labels

        for i in range(0, spec.shape[1] - seq_length, seq_length // 2):
            X.append(spec[:, i:i + seq_length].T)
            y.append(labels[i:i + seq_length])

    return np.array(X), np.expand_dims(np.array(y), -1)  # Kształty: (n_seq, seq_length, n_mels), (n_seq, seq_length, 1)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32, return_sequences=True),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')]

    )
    return model


def train_model():
    guitarset = mirdata.initialize("guitarset", data_home=Config.DATA_DIR)
    track_ids = guitarset.track_ids[:20]

    all_spectrograms = []
    all_onsets = []

    for track_id in track_ids:
        audio, sr, onsets, _ = load_audio_and_annotations(track_id, guitarset)
        spec = create_spectrogram(audio, sr)
        all_spectrograms.append(spec)
        all_onsets.append(onsets)
        print(f"Przetworzono {track_id}, kształt: {spec.shape}")

    X, y = prepare_training_data(all_spectrograms, all_onsets)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Kształty: X={X.shape}, y={y.shape}")

    model = create_lstm_model((X.shape[1], X.shape[2]))
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Poprawione evaluate - teraz przyjmie dowolną liczbę metryk
    test_results = model.evaluate(X_val, y_val, verbose=0)

    # Nazwy metryk odpowiadają kolejności w model.compile()
    metric_names = model.metrics_names
    for name, value in zip(metric_names, test_results):
        print(f"{name}: {value:.4f}")

    return model