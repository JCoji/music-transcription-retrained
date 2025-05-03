from pathlib import Path
from typing import Union, Dict, Any
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class GuitarSetDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path], file_extension: str = "*.pt"):
        self.root_dir = Path(root_dir)
        self.file_paths = sorted(self.root_dir.rglob(file_extension))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_paths[idx]
        return torch.load(file_path, weights_only=False)


def collate_fn(batch):
    # Wyodrębnienie pól
    audios = [item["audio"] for item in batch]
    features = [item["features"] for item in batch]
    notes_maps = [create_sparse_map(item["note_indices"], item["note_values"], item["shape_notes"]) for item in batch]
    contours_maps = [create_sparse_map(item["contour_indices"], item["contour_values"], item["shape_contours"]) for item
                     in batch]

    # Padding po wymiarze czasowym
    features_padded = pad_sequence(features, batch_first=True)  # (B, T, F)
    notes_padded = pad_sequence(notes_maps, batch_first=True)
    contours_padded = pad_sequence(contours_maps, batch_first=True)

    # Maski (1 tam, gdzie prawdziwe dane)
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    mask = torch.arange(features_padded.shape[1])[None, :] < lengths[:, None]  # (B, T)

    return {
        "audio": audios,
        "features": features_padded,
        "notes": notes_padded,
        "contours": contours_padded,
        "mask": mask,
        "lengths": lengths
    }


def create_sparse_map(indices, values, shape):
    dense = torch.zeros(shape, dtype=torch.float32)
    if len(indices) > 0:
        indices = torch.tensor(indices, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        dense[indices[:, 0], indices[:, 1]] = values
    return dense
