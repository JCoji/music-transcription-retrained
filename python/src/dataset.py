from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset


def load_all_batches(split_dir: Path) -> List[Dict]:
    all_data = []
    for batch_file in sorted(split_dir.glob("*.pt")):
        batch_data = torch.load(batch_file, weights_only=False)
        all_data.extend(batch_data)
    return all_data


class GuitarSetDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            "audio": sample["audio"],  # [T]
            "features": sample["features"],  # [n_bins, n_frames]
            "note_indices": sample["note_indices"],  # (N, 2)
            "note_values": sample["note_values"],  # (N,)
            "onset_indices": sample["onset_indices"],
            "onset_values": sample["onset_values"],
            "contour_indices": sample["contour_indices"],
            "contour_values": sample["contour_values"],
            "shape_notes": sample["shape_notes"],
            "shape_contours": sample["shape_contours"],
            "track_id": sample["track_id"],
            "sample_rate": sample["sample_rate"]
        }
