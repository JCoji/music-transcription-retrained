import random
from pathlib import Path
from typing import List, Dict
import librosa
import mirdata
import numpy as np
import torch
from tqdm import tqdm

from .features import compute_cqt
from .augmentation import augment_audio
from .config import (
    FREQ_BINS_NOTES,
    FREQ_BINS_CONTOURS,
    AUDIO_SAMPLE_RATE,
    ANNOTATION_HOP,
    N_FREQ_BINS_NOTES,
    N_FREQ_BINS_CONTOURS
)


def load_and_process_track(track, resample_rate: int = AUDIO_SAMPLE_RATE, apply_augmentation: bool = False) -> Dict:
    audio, sr = librosa.load(track.audio_mic_path, sr=resample_rate, mono=True)

    # Augmentacja dźwięku jeśli wymagana
    if apply_augmentation:
        audio = torch.tensor(audio, dtype=torch.float32)
        audio = augment_audio(audio, sr)
        audio = audio.numpy()

    # Obliczenie cech CQT
    features = compute_cqt(audio)

    # Obliczenie czasu trwania i siatki czasowej
    duration = librosa.get_duration(y=audio, sr=sr)
    time_grid = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
    n_time_frames = len(time_grid)

    # Konwersja nut i konturów do formatu sparse
    note_indices, note_values = track.notes_all.to_sparse_index(
        time_grid, "s", FREQ_BINS_NOTES, "hz"
    )
    onset_indices, onset_values = track.notes_all.to_sparse_index(
        time_grid, "s", FREQ_BINS_NOTES, "hz", onsets_only=True
    )
    contour_indices, contour_values = track.multif0.to_sparse_index(
        time_grid, "s", FREQ_BINS_CONTOURS, "hz"
    )

    return {
        "track_id": track.track_id,
        "audio": torch.tensor(audio, dtype=torch.float32),
        "features": features,
        "sample_rate": sr,
        "note_indices": note_indices,
        "note_values": note_values,
        "onset_indices": onset_indices,
        "onset_values": onset_values,
        "contour_indices": contour_indices,
        "contour_values": contour_values,
        "shape_notes": (n_time_frames, N_FREQ_BINS_NOTES),
        "shape_contours": (n_time_frames, N_FREQ_BINS_CONTOURS),
    }


def split_dataset(track_ids: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Dict[
    str, List[str]]:
    random.seed(seed)
    random.shuffle(track_ids)
    n_train = int(len(track_ids) * train_ratio)
    n_val = int(len(track_ids) * val_ratio)
    return {
        "train": track_ids[:n_train],
        "val": track_ids[n_train:n_train + n_val],
        "test": track_ids[n_train + n_val:],
    }


def save_track(data: Dict, destination_dir: Path, track_id: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = destination_dir / f"{track_id}.pt"
    torch.save(data, filename, _use_new_zipfile_serialization=False)


def process_split(guitarset, split_name: str, track_ids: List[str], output_dir: Path,
                  apply_augmentation: bool) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for track_id in tqdm(track_ids, desc=f"Processing {split_name}"):
        track = guitarset.track(track_id)
        data = load_and_process_track(track, apply_augmentation=apply_augmentation)
        save_track(data, split_dir, track_id)


def process_dataset(data_dir="guitarset_data", output_dir="processed_data",
                    seed=42, apply_augmentation=False,
                    max_tracks=None, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        print(f"Katalog wyjściowy '{output_dir}' już istnieje. Pomijam przetwarzanie.")
        return
    elif output_dir.exists() and overwrite:
        print(f"Nadpisuję istniejący katalog wyjściowy '{output_dir}'.")

    # Inicjalizacja datasetu GuitarSet
    guitarset = mirdata.initialize("guitarset", data_home=data_dir)
    if not guitarset.validate():
        guitarset.download()

    all_track_ids = guitarset.track_ids

    # Ograniczenie liczby utworów jeśli wymagane
    if max_tracks is not None and max_tracks > 0:
        random.seed(seed)
        all_track_ids = random.sample(all_track_ids, min(max_tracks, len(all_track_ids)))

    # Podział datasetu
    splits = split_dataset(all_track_ids, seed=seed)

    # Przetwarzanie każdego podziału
    for split_name, track_ids in splits.items():
        process_split(
            guitarset,
            split_name,
            track_ids,
            output_dir,
            apply_augmentation=apply_augmentation,
        )
