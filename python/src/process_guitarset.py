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
    N_FREQ_BINS_CONTOURS, FFT_HOP
)


def load_and_process_track(track, resample_rate: int = AUDIO_SAMPLE_RATE, apply_augmentation: bool = False) -> Dict:
    audio, sr = librosa.load(track.audio_mic_path, sr=resample_rate, mono=True)

    if apply_augmentation:
        audio = torch.tensor(audio, dtype=torch.float32)
        audio = augment_audio(audio, sr)
        audio = audio.numpy()

    # CQT/mel features
    features = compute_cqt(audio, sr, hop_length=FFT_HOP)

    # Reszta bez zmian
    duration = librosa.get_duration(y=audio, sr=sr)
    time_grid = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
    n_time_frames = len(time_grid)

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


def save_batch(data_list: List[Dict], destination_dir: Path, batch_name: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = destination_dir / f"batch_{batch_name}.pt"
    torch.save(data_list, filename)


def process_split(guitarset, split_name: str, track_ids: List[str], output_dir: Path,
                  apply_augmentation: bool, batch_size: int = 32) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    current_batch = []
    batch_num = 0

    for track_id in tqdm(track_ids, desc=f"Processing {split_name}"):
        track = guitarset.track(track_id)
        data = load_and_process_track(track, apply_augmentation=apply_augmentation)
        current_batch.append(data)

        if len(current_batch) >= batch_size:
            save_batch(current_batch, split_dir, f"{batch_num}")
            batch_num += 1
            current_batch = []

    if current_batch:
        save_batch(current_batch, split_dir, f"{batch_num}_final")


def process_dataset(data_dir="guitarset_data", output_dir="processed_data",
                    batch_size=32, seed=42, apply_augmentation=False,
                    max_tracks=None, overwrite=False):
    """Main function for notebook usage

    Args:
        data_dir: Directory with GuitarSet data
        output_dir: Output directory for processed data
        batch_size: Size of processing batches
        seed: Random seed for reproducibility
        apply_augmentation: Whether to apply audio augmentation
        max_tracks: Maximum number of tracks to process (None for all)
        overwrite: Whether to overwrite existing output directory
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        print(f"Output directory '{output_dir}' already exists. Skipping processing.")
        return
    elif output_dir.exists() and overwrite:
        print(f"Overwriting existing output directory '{output_dir}'.")

    guitarset = mirdata.initialize("guitarset", data_home=data_dir)
    if not guitarset.validate():
        guitarset.download()

    all_track_ids = guitarset.track_ids

    if max_tracks is not None and max_tracks > 0:
        random.seed(seed)
        all_track_ids = random.sample(all_track_ids, min(max_tracks, len(all_track_ids)))

    splits = split_dataset(all_track_ids, seed=seed)

    for split_name, track_ids in splits.items():
        process_split(
            guitarset,
            split_name,
            track_ids,
            output_dir,
            apply_augmentation=apply_augmentation,
            batch_size=batch_size
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="guitarset_data")
    parser.add_argument("--output-dir", default="processed_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-augmentation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Maximum number of tracks to process (None for all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        apply_augmentation=args.apply_augmentation,
        max_tracks=args.max_tracks,
        overwrite=args.overwrite
    )
