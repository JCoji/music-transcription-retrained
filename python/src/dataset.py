from pathlib import Path
from typing import Union, Dict, Any, List
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GuitarSetDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path], file_extension: str = "*.pt"):
        # Inicjalizacja datasetu z plikami w podanym katalogu
        self.root_dir = Path(root_dir)
        self.file_paths = sorted(self.root_dir.rglob(file_extension))

    def __len__(self):
        # Zwraca liczbę plików w datasetcie
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Ładuje pojedynczy plik z datasetu
        file_path = self.file_paths[idx]
        return torch.load(file_path, map_location='cpu', weights_only=False)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Funkcja pomocnicza do przesuwania indeksów w batchu
    def shift_indices(indices, batch_idx, time_offset):
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices)
        shifted = torch.stack([
            indices[:, 0] + time_offset,
            indices[:, 1]
        ], dim=1)
        batch_column = torch.full((indices.shape[0], 1), batch_idx, dtype=torch.long)
        return torch.cat([batch_column, shifted], dim=1)  # shape [N, 3]

    # Przygotowanie list cech i audio z batcha
    features_list = [item["features"] for item in batch]
    audio_list = [item["audio"] for item in batch]
    feature_lengths = [feat.shape[0] for feat in features_list]
    max_len = max(feature_lengths)
    B = len(batch)
    F_notes = batch[0]["shape_notes"][1]
    F_contours = batch[0]["shape_contours"][1]

    # Dopasowanie sekwencji do tej samej długości
    padded_features = pad_sequence(features_list, batch_first=True)       # [B, T, F]
    padded_audio = pad_sequence(audio_list, batch_first=True)             # [B, T_audio]

    device = padded_features.device

    # Tworzenie maski dla prawidłowych długości
    mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    for i, length in enumerate(feature_lengths):
        mask[i, :length] = 1.0

    # Inicjalizacja tensorów dla nut, onsetów i konturów
    note_dense = torch.zeros((B, max_len, F_notes), dtype=torch.float32, device=device)
    onset_dense = torch.zeros((B, max_len, F_notes), dtype=torch.float32, device=device)
    contour_dense = torch.zeros((B, max_len, F_contours), dtype=torch.float32, device=device)

    # Przetwarzanie każdego elementu w batchu
    for i, item in enumerate(batch):
        # Przygotowanie indeksów i wartości dla sparse tensors
        note_indices = shift_indices(item["note_indices"], i, 0).to(device).T   # shape [3, N]
        onset_indices = shift_indices(item["onset_indices"], i, 0).to(device).T
        contour_indices = shift_indices(item["contour_indices"], i, 0).to(device).T

        note_values = torch.tensor(item["note_values"], dtype=torch.float32, device=device)
        onset_values = torch.tensor(item["onset_values"], dtype=torch.float32, device=device)
        contour_values = torch.tensor(item["contour_values"], dtype=torch.float32, device=device)

        # Konwersja sparse tensors na dense
        note_sparse = torch.sparse_coo_tensor(
            note_indices, note_values, size=(B, max_len, F_notes), device=device
        ).to_dense()
        onset_sparse = torch.sparse_coo_tensor(
            onset_indices, onset_values, size=(B, max_len, F_notes), device=device
        ).to_dense()
        contour_sparse = torch.sparse_coo_tensor(
            contour_indices, contour_values, size=(B, max_len, F_contours), device=device
        ).to_dense()

        # Akumulacja wyników
        note_dense += note_sparse
        onset_dense += onset_sparse
        contour_dense += contour_sparse

    # Zwracanie słownika z wynikami
    return {
        "features": padded_features,              # [B, T, F]
        "audio": padded_audio,                    # [B, T_audio]
        "feature_lengths": torch.tensor(feature_lengths, device=device),  # [B]
        "notes": note_dense,                      # [B, T, F_notes]
        "onsets": onset_dense,                    # [B, T, F_notes]
        "contours": contour_dense,                # [B, T, F_contours]
        "mask": mask                              # [B, T]
    }