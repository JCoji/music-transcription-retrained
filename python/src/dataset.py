import json
from pathlib import Path
from typing import List, Tuple, Iterator, Dict, Any, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm


class BucketBatchSampler(Sampler[List[int]]):
    """
    Sampler, który grupuje próbki o podobnych długościach w "kubełki" (buckets),
    a następnie tworzy batche z tych kubełków. Pomaga to zminimalizować padding.
    """

    def __init__(
        self,
        length_cache: Dict[int, int],
        batch_size: int,
        num_samples: int,
        drop_last: bool = False,
        bucket_size_multiplier: int = 100,
        shuffle: bool = True,
    ):
        """
        Inicjalizuje BucketBatchSampler.

        Args:
            length_cache (Dict[int, int]): Słownik mapujący indeks próbki na jej długość.
            batch_size (int): Rozmiar każdego batcha.
            num_samples (int): Całkowita liczba próbek w zbiorze danych.
            drop_last (bool): Jeśli True, ostatni niepełny batch zostanie pominięty.
            bucket_size_multiplier (int): Mnożnik rozmiaru batcha określający rozmiar kubełka.
                                          Kubełek będzie miał rozmiar batch_size * bucket_size_multiplier.
            shuffle (bool): Jeśli True, dane w kubełkach oraz kolejność batchy będą tasowane.
        """
        super().__init__()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket_size_multiplier = bucket_size_multiplier
        self.shuffle = shuffle
        self.num_samples = num_samples

        # Zakładamy poprawność przekazanego cache'a długości.
        self.lengths = np.array([length_cache[i] for i in range(self.num_samples)])
        self.sorted_indices = np.argsort(
            self.lengths
        )  # Indeksy posortowane według długości próbek

        # Obliczenie całkowitej liczby batchy
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (
                self.num_samples + self.batch_size - 1
            ) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        """Zwraca iterator po batchach indeksów próbek."""
        bucket_size = self.batch_size * self.bucket_size_multiplier
        num_buckets = (self.num_samples + bucket_size - 1) // bucket_size
        all_batches = []

        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = min((i + 1) * bucket_size, self.num_samples)
            bucket = self.sorted_indices[start_idx:end_idx].tolist()
            if self.shuffle:
                np.random.shuffle(bucket)  # Tasowanie próbek wewnątrz kubełka

            for j in range(0, len(bucket), self.batch_size):
                batch = bucket[j : j + self.batch_size]
                # Pomijanie ostatniego batcha, jeśli jest niepełny i drop_last=True
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if batch:  # Upewniamy się, że batch nie jest pusty
                    all_batches.append(batch)

        if self.shuffle:
            np.random.shuffle(all_batches)  # Tasowanie kolejności batchy

        yield from all_batches

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę batchy."""
        return self.num_batches


class GuitarSetDataset(Dataset):
    """
    Klasa Dataset dla zbioru GuitarSet, wczytująca przetworzone dane.
    Automatycznie tworzy i wczytuje cache długości sekwencji dla szybszego działania.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        file_extension: str = "*.pt",
        cache_file_name: str = "length_cache.json",
    ):
        """
        Inicjalizuje GuitarSetDataset.

        Args:
            root_dir (Union[str, Path]): Ścieżka do głównego katalogu ze zbioru danych.
            file_extension (str): Rozszerzenie plików do wczytania (domyślnie "*.pt").
            cache_file_name (str): Nazwa pliku cache dla długości sekwencji.
        """
        self.root_dir = Path(root_dir)
        # Zakładamy, że katalog root_dir istnieje i zawiera pliki danych.
        self.file_paths = sorted(list(self.root_dir.rglob(file_extension)))
        self.cache_path = self.root_dir / cache_file_name
        self.lengths_cache = self._load_or_create_length_cache()

    def _load_or_create_length_cache(self) -> Dict[int, int]:
        """
        Wczytuje cache długości sekwencji z pliku, jeśli istnieje,
        lub tworzy nowy cache, jeśli plik nie istnieje.
        Zakładamy, że istniejący cache jest poprawny.
        """
        if self.cache_path.exists():
            # Zakładamy, że cache jest poprawny, jeśli plik istnieje.
            with open(self.cache_path, "r") as f:
                lengths_cache_str_keys = json.load(f)
            # Konwersja kluczy na int, ponieważ JSON przechowuje klucze jako stringi
            lengths_cache = {int(k): v for k, v in lengths_cache_str_keys.items()}
            if len(lengths_cache) != len(self.file_paths):
                print(
                    f"OSTRZEŻENIE: Rozmiar wczytanego cache ({len(lengths_cache)}) różni się od liczby plików ({len(self.file_paths)}). Może być konieczne usunięcie pliku cache i ponowne uruchomienie."
                )
            return lengths_cache
        else:
            return self._create_length_cache()

    def _create_length_cache(self) -> Dict[int, int]:
        """
        Tworzy cache długości sekwencji, iterując przez wszystkie pliki danych.
        Zakładamy poprawność struktury plików danych.
        """
        lengths_cache = {}
        # Użycie tqdm do wyświetlania paska postępu podczas tworzenia cache'a.
        for idx, file_path in enumerate(
            tqdm(self.file_paths, desc="Tworzenie cache'a długości sekwencji")
        ):
            # Wczytujemy tylko dane potrzebne do określenia długości (klucz 'features')
            data = torch.load(file_path, map_location="cpu", weights_only=False)
            lengths_cache[idx] = data["features"].shape[
                0
            ]  # Długość sekwencji to pierwszy wymiar 'features'

        with open(self.cache_path, "w") as f:
            json.dump(lengths_cache, f)  # Zapis cache'a do pliku JSON

        return lengths_cache

    def __len__(self):
        """Zwraca całkowitą liczbę próbek w zbiorze danych."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Ładuje i zwraca pojedynczą próbkę danych o zadanym indeksie.
        Zakładamy poprawność indeksu, pliku oraz struktury danych w pliku.
        """
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        # Konwersja 'features' do tensora float32, jeśli jest to ndarray
        features_data = data["features"]
        if isinstance(features_data, np.ndarray):
            data["features"] = torch.from_numpy(features_data).to(torch.float32)
        elif isinstance(features_data, torch.Tensor):
            data["features"] = features_data.to(
                torch.float32
            )  # Upewnienie się, że typ to float32

        # Usuwanie niepotrzebnych kluczy z wczytanych danych
        keys_to_remove = ["audio", "sample_rate", "track_id", "feature_length"]
        for key in keys_to_remove:
            data.pop(
                key, None
            )  # Użycie pop(key, None) jest bezpieczne, nie rzuci błędu jeśli klucza nie ma.
        return data


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Funkcja łącząca listę próbek (słowników) w pojedynczy batch (słownik tensorów).
    Obsługuje padding sekwencji o zmiennej długości oraz konwersję danych sparse do dense.
    Wszystkie operacje i wynikowe tensory są domyślnie na CPU.
    """
    if not batch:  # Pusty batch
        return {}
    target_device = "cpu"  # Wszystkie tensory wynikowe będą na CPU

    features_list = [item["features"] for item in batch]
    feature_lengths = [feat.shape[0] for feat in features_list]
    shape_notes = batch[0]["shape_notes"]
    shape_contours = batch[0]["shape_contours"]
    F_notes = shape_notes[1]  # Liczba binów częstotliwości dla nut
    F_contours = shape_contours[1]  # Liczba binów częstotliwości dla konturów

    max_len = (
        max(feature_lengths) if feature_lengths else 0
    )  # Maksymalna długość sekwencji w batchu
    B = len(batch)  # Rozmiar batcha

    # Padding sekwencji 'features' do maksymalnej długości
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Tworzenie maski dla paddingu
    mask = torch.zeros((B, max_len), dtype=torch.bool, device=target_device)
    for i, length in enumerate(feature_lengths):
        mask[i, :length] = True

    all_note_indices, all_note_values = [], []
    all_onset_indices, all_onset_values = [], []
    all_contour_indices, all_contour_values = [], []

    def process_sparse_data(
        item: Dict[str, Any], key_indices: str, key_values: str, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Przetwarza dane sparse (indeksy, wartości) dla pojedynczej próbki."""
        # Zakładamy istnienie kluczy 'key_indices' i 'key_values'.
        indices = item[key_indices]
        values = item[key_values]

        # Konwersja do tensorów PyTorch, jeśli są to tablice NumPy
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).to(device=target_device)
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(device=target_device)
        indices = indices.to(device=target_device, dtype=torch.long)
        values = values.to(device=target_device, dtype=torch.float32)

        if indices.shape[0] > 0:  # Jeśli istnieją jakiekolwiek dane sparse
            # Dodanie indeksu batcha do pierwszej kolumny indeksów
            batch_column = torch.full(
                (indices.shape[0], 1), batch_idx, dtype=torch.long, device=target_device
            )
            shifted_indices = torch.cat([batch_column, indices], dim=1)
            return (
                shifted_indices.T,
                values,
            )  # Zwracamy transponowane indeksy (format COO)
        else:
            # Zwracanie pustych tensorów, jeśli brak danych sparse dla tej próbki
            return torch.empty(
                (3, 0), dtype=torch.long, device=target_device
            ), torch.empty((0,), dtype=torch.float32, device=target_device)

    for i, item in enumerate(batch):
        note_idx_T, note_val = process_sparse_data(
            item, "note_indices", "note_values", i
        )
        onset_idx_T, onset_val = process_sparse_data(
            item, "onset_indices", "onset_values", i
        )
        contour_idx_T, contour_val = process_sparse_data(
            item, "contour_indices", "contour_values", i
        )

        all_note_indices.append(note_idx_T)
        all_note_values.append(note_val)
        all_onset_indices.append(onset_idx_T)
        all_onset_values.append(onset_val)
        all_contour_indices.append(contour_idx_T)
        all_contour_values.append(contour_val)

    def create_dense_from_sparse(
        all_indices_T: List[torch.Tensor],
        all_values: List[torch.Tensor],
        size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Tworzy gęsty tensor z listy zebranych danych sparse."""
        # Jeśli wszystkie listy indeksów są puste, zwróć tensor zerowy
        if not any(idx.numel() > 0 for idx in all_indices_T):
            return torch.zeros(size, dtype=torch.float32, device=target_device)

        # Zakładamy, że konkatenacja się powiedzie i wymiary będą zgodne.
        indices_T = torch.cat(all_indices_T, dim=1)
        values = torch.cat(all_values, dim=0)

        # Tworzenie tensora sparse COO na CPU, a następnie konwersja do gęstego
        sparse_tensor = torch.sparse_coo_tensor(
            indices_T, values, size=size, device=target_device
        )
        return sparse_tensor.coalesce().to_dense()

    notes_dense = create_dense_from_sparse(
        all_note_indices, all_note_values, (B, max_len, F_notes)
    )
    onsets_dense = create_dense_from_sparse(
        all_onset_indices, all_onset_values, (B, max_len, F_notes)
    )
    contours_dense = create_dense_from_sparse(
        all_contour_indices, all_contour_values, (B, max_len, F_contours)
    )

    return {
        "features": padded_features.to(target_device),  # Upewnienie się, że jest na CPU
        "feature_lengths": torch.tensor(
            feature_lengths, dtype=torch.long, device=target_device
        ),
        "notes": notes_dense.to(target_device),
        "onsets": onsets_dense.to(target_device),
        "contours": contours_dense.to(target_device),
        "mask": mask.to(target_device),
    }
