import json
import random  # Dodajemy import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
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
    Dodano możliwość augmentacji onsetów dla zbioru treningowego.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        file_extension: str = "*.pt",
        cache_file_name: str = "length_cache.json",
        is_train_dataset: bool = False,
        augment_onsets_prob: float = 0.0,
        onset_blur_window_size: int = 1,
        onset_blur_probability: float = 0.6,  # Prawd. rozszerzenia o 1 ramkę w danym kierunku
        onset_augmented_value_decay: float = 0.4,  # Jak bardzo spada wartość aug. onsetu z odległością
    ):
        """
        Inicjalizuje GuitarSetDataset.

        Args:
            root_dir (Union[str, Path]): Ścieżka do głównego katalogu ze zbioru danych.
            file_extension (str): Rozszerzenie plików do wczytania (domyślnie "*.pt").
            cache_file_name (str): Nazwa pliku cache dla długości sekwencji.
            is_train_dataset (bool): Flaga wskazująca, czy jest to zbiór treningowy (dla augmentacji).
            augment_onsets_prob (float): Prawdopodobieństwo zastosowania augmentacji dla danej próbki.
            onset_blur_window_size (int): Promień rozmycia onsetu. Np. 1 oznacza, że onset
                                          w ramce `t` może zostać rozszerzony na `t-1, t, t+1`.
            onset_blur_probability (float): Prawdopodobieństwo rozszerzenia onsetu o jedną ramkę
                                            w danym kierunku (lewo/prawo) dla każdego kroku w oknie.
            onset_augmented_value_decay (float): Współczynnik, o który maleje wartość augmentowanego
                                                 onetu dla każdej dodatkowej ramki odległości od oryginału.
                                                 Wartość = original_value * (1.0 - decay * shift).
        """
        self.root_dir = Path(root_dir)
        self.file_paths = sorted(list(self.root_dir.rglob(file_extension)))
        self.cache_path = self.root_dir / cache_file_name
        self.lengths_cache = self._load_or_create_length_cache()
        self.is_train_dataset = is_train_dataset
        self.augment_onsets_prob = augment_onsets_prob
        self.onset_blur_window_size = onset_blur_window_size
        self.onset_blur_probability = onset_blur_probability
        self.onset_augmented_value_decay = onset_augmented_value_decay

    def _load_or_create_length_cache(self) -> Dict[int, int]:
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                lengths_cache_str_keys = json.load(f)
            lengths_cache = {int(k): v for k, v in lengths_cache_str_keys.items()}
            if len(lengths_cache) != len(self.file_paths):
                print(
                    f"OSTRZEŻENIE: Rozmiar wczytanego cache ({len(lengths_cache)}) różni się od liczby plików ({len(self.file_paths)}). Może być konieczne usunięcie pliku cache i ponowne uruchomienie."
                )
            return lengths_cache
        else:
            return self._create_length_cache()

    def _create_length_cache(self) -> Dict[int, int]:
        lengths_cache = {}
        for idx, file_path in enumerate(
            tqdm(self.file_paths, desc="Tworzenie cache'a długości sekwencji")
        ):
            data = torch.load(
                file_path, map_location="cpu", weights_only=False
            )  # weights_only=False jest domyślne, ale dla jasności
            lengths_cache[idx] = data["features"].shape[0]
        with open(self.cache_path, "w") as f:
            json.dump(lengths_cache, f)
        return lengths_cache

    def __len__(self):
        return len(self.file_paths)

    def _augment_onsets(
        self, onset_indices: np.ndarray, onset_values: np.ndarray, max_time_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stosuje augmentację do danych onsetów poprzez ich rozszerzenie/rozmycie.
        """
        if onset_indices.size == 0:
            return onset_indices, onset_values

        # Konwersja do list dla łatwiejszej manipulacji, potem z powrotem do ndarray
        augmented_onset_indices_list = []
        augmented_onset_values_list = []

        # Zbiór do śledzenia już dodanych par (time_frame, freq_bin), aby uniknąć duplikatów
        # i nadpisywania oryginalnych wartości przez augmentowane o niższej wartości
        # Klucz: (time_frame, freq_bin), Wartość: najwyższa dotychczasowa wartość dla tego onsetu
        existing_onsets_map = {}

        if (
            onset_indices.ndim == 2 and onset_indices.shape[1] == 2
        ):  # (N, 2) -> (time, freq_bin)
            for i in range(onset_indices.shape[0]):
                time_frame, freq_bin = int(onset_indices[i, 0]), int(
                    onset_indices[i, 1]
                )
                original_value = float(onset_values[i])

                # Dodajemy/aktualizujemy oryginalny onset
                current_max_val = existing_onsets_map.get((time_frame, freq_bin), 0.0)
                if original_value > current_max_val:
                    existing_onsets_map[(time_frame, freq_bin)] = original_value

                # Rozmycie/rozszerzenie onsetu
                for shift in range(1, self.onset_blur_window_size + 1):
                    # Rozszerzenie w lewo
                    if random.random() < self.onset_blur_probability:
                        new_time_frame_prev = time_frame - shift
                        if new_time_frame_prev >= 0:
                            # Wartość maleje z odległością
                            augmented_value = original_value * max(
                                0, (1.0 - self.onset_augmented_value_decay * shift)
                            )
                            current_max_val = existing_onsets_map.get(
                                (new_time_frame_prev, freq_bin), 0.0
                            )
                            if (
                                augmented_value > current_max_val
                            ):  # Dodajemy tylko jeśli nowa wartość jest lepsza
                                existing_onsets_map[(new_time_frame_prev, freq_bin)] = (
                                    augmented_value
                                )
                    # Rozszerzenie w prawo
                    if random.random() < self.onset_blur_probability:
                        new_time_frame_next = time_frame + shift
                        if new_time_frame_next < max_time_frames:
                            augmented_value = original_value * max(
                                0, (1.0 - self.onset_augmented_value_decay * shift)
                            )
                            current_max_val = existing_onsets_map.get(
                                (new_time_frame_next, freq_bin), 0.0
                            )
                            if augmented_value > current_max_val:
                                existing_onsets_map[(new_time_frame_next, freq_bin)] = (
                                    augmented_value
                                )
        else:  # Jeśli onset_indices nie jest w oczekiwanym formacie, zwróć oryginał
            return onset_indices, onset_values

        # Konwersja mapy z powrotem do list, a następnie do ndarray
        if not existing_onsets_map:
            return np.array([]).reshape(0, 2).astype(np.int64), np.array([]).astype(
                np.float32
            )

        for (tf, fb), val in existing_onsets_map.items():
            if val > 1e-3:  # Dodajemy tylko jeśli wartość jest znacząca
                augmented_onset_indices_list.append([tf, fb])
                augmented_onset_values_list.append(val)

        if not augmented_onset_indices_list:
            return np.array([]).reshape(0, 2).astype(np.int64), np.array([]).astype(
                np.float32
            )

        return np.array(augmented_onset_indices_list, dtype=np.int64), np.array(
            augmented_onset_values_list, dtype=np.float32
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_paths[idx]
        # Wczytujemy dane, weights_only=False jest istotne, jeśli chcemy modyfikować dane przed zwróceniem
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        features_data = data["features"]
        if isinstance(features_data, np.ndarray):
            data["features"] = torch.from_numpy(features_data).to(torch.float32)
        elif isinstance(features_data, torch.Tensor):  # Upewnij się, że jest to float32
            data["features"] = features_data.to(torch.float32)

        # Augmentacja onsetów tylko dla zbioru treningowego i z zadanym prawdopodobieństwem
        if self.is_train_dataset and random.random() < self.augment_onsets_prob:
            if (
                "onset_indices" in data
                and "onset_values" in data
                and isinstance(data["onset_indices"], np.ndarray)
                and isinstance(data["onset_values"], np.ndarray)
            ):
                max_time_frames = data["features"].shape[0]
                # Tworzymy kopie, aby nie modyfikować danych w miejscu, jeśli ten sam plik .pt byłby wczytywany wielokrotnie bez augmentacji
                onset_indices_copy = data["onset_indices"].copy()
                onset_values_copy = data["onset_values"].copy()

                augmented_onset_indices, augmented_onset_values = self._augment_onsets(
                    onset_indices_copy, onset_values_copy, max_time_frames
                )
                data["onset_indices"] = augmented_onset_indices
                data["onset_values"] = augmented_onset_values

        keys_to_remove_original = [
            "audio",
            "sample_rate",
        ]
        for key in keys_to_remove_original:
            data.pop(key, None)

        return data


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Funkcja łącząca listę próbek (słowników) w pojedynczy batch (słownik tensorów).
    Obsługuje padding sekwencji o zmiennej długości oraz konwersję danych sparse do dense.
    Wszystkie operacje i wynikowe tensory są domyślnie na CPU.
    """
    if not batch:
        return {}
    target_device = "cpu"  # Wszystkie tensory wynikowe będą na CPU
    track_ids_list = [
        item.get("track_id", f"unknown_track_{i}") for i, item in enumerate(batch)
    ]
    features_list = [item["features"] for item in batch]
    feature_lengths = [feat.shape[0] for feat in features_list]

    # Sprawdzenie czy batch nie jest pusty przed dostępem do batch[0]
    if not batch[0]:
        return {}  # lub rzuć błąd

    shape_notes = batch[0].get("shape_notes")
    shape_contours = batch[0].get("shape_contours")

    if shape_notes is None or shape_contours is None:
        # Jeśli brakuje kształtów, użyj domyślnych lub zgłoś błąd.
        # To powinno być obsługiwane podczas tworzenia danych .pt
        # Dla przykładu, można spróbować odzyskać z pierwszego elementu, który ma te klucze
        first_valid_item = next(
            (
                item
                for item in batch
                if item.get("shape_notes") is not None
                and item.get("shape_contours") is not None
            ),
            None,
        )
        if first_valid_item:
            shape_notes = first_valid_item["shape_notes"]
            shape_contours = first_valid_item["shape_contours"]
        else:
            # To jest problematyczne, należy ustawić wartości domyślne lub przerwać
            # print("OSTRZEŻENIE: Brak 'shape_notes' lub 'shape_contours' w batchu. Ustawiam domyślne F_bins.")
            # Poniższe wartości powinny być spójne z konfiguracją N_FREQ_BINS_NOTES i N_FREQ_BINS_CONTOURS
            # np. z pliku config.py. Załóżmy, że je zaimportowaliśmy lub mamy je jako argumenty.
            # import v1.config as cfg # Przykładowo
            # F_notes = cfg.N_FREQ_BINS_NOTES
            # F_contours = cfg.N_FREQ_BINS_CONTOURS
            # Bezpieczniej jest przerwać lub logować błąd, jeśli to krytyczne.
            # Na potrzeby przykładu, jeśli nie ma skąd wziąć, to problem.
            # Zakładając, że problem nie wystąpi jeśli dane są poprawnie przygotowane:
            if not features_list:  # Jeśli lista cech jest pusta
                F_notes = 192  # Placeholder, musi być zgodne z rzeczywistymi danymi
                F_contours = 192  # Placeholder
            else:
                # Próba odgadnięcia z samych danych, jeśli 'shape_notes' brakuje
                # To jest bardzo ryzykowne i niezalecane w produkcji
                # Lepiej upewnić się, że 'shape_notes' i 'shape_contours' są zawsze obecne w danych .pt
                F_notes = (
                    batch[0]["notes"].shape[2]
                    if "notes" in batch[0] and batch[0]["notes"].ndim == 3
                    else 192
                )
                F_contours = (
                    batch[0]["contours"].shape[2]
                    if "contours" in batch[0] and batch[0]["contours"].ndim == 3
                    else 192
                )

    F_notes = shape_notes[1]  # Liczba binów częstotliwości dla nut
    F_contours = shape_contours[1]  # Liczba binów częstotliwości dla konturów

    max_len = max(feature_lengths) if feature_lengths else 0
    B = len(batch)

    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    mask = torch.zeros((B, max_len), dtype=torch.bool, device=target_device)
    for i, length in enumerate(feature_lengths):
        mask[i, :length] = True

    all_note_indices, all_note_values = [], []
    all_onset_indices, all_onset_values = [], []
    all_contour_indices, all_contour_values = [], []

    def process_sparse_data(
        item: Dict[str, Any], key_indices: str, key_values: str, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = item.get(key_indices)  # Użyj .get() dla bezpieczeństwa
        values = item.get(key_values)

        if indices is None or values is None:  # Jeśli brakuje danych sparse dla klucza
            # print(f"OSTRZEŻENIE: Brak klucza '{key_indices}' lub '{key_values}' w próbce {item.get('track_id', 'N/A')}. Zwracam puste tensory.")
            return torch.empty(
                (3, 0), dtype=torch.long, device=target_device
            ), torch.empty((0,), dtype=torch.float32, device=target_device)

        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).to(device=target_device)
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(device=target_device)

        # Upewnij się, że typy są poprawne, nawet jeśli już są tensorami
        indices = indices.to(device=target_device, dtype=torch.long)
        values = values.to(device=target_device, dtype=torch.float32)

        if (
            indices.numel() > 0 and values.numel() > 0
        ):  # Sprawdź czy tensory nie są puste
            # Upewnij się, że indices ma odpowiedni kształt (N, 2)
            if indices.ndim == 1:  # Jeśli jest jednowymiarowy, spróbuj go przekształcić
                if indices.shape[0] % 2 == 0:
                    indices = indices.view(-1, 2)
                else:
                    # print(f"OSTRZEŻENIE: Niespodziewany kształt 1D dla indeksów {key_indices}: {indices.shape}. Zwracam puste.")
                    return torch.empty(
                        (3, 0), dtype=torch.long, device=target_device
                    ), torch.empty((0,), dtype=torch.float32, device=target_device)
            elif indices.ndim != 2 or indices.shape[1] != 2:
                # print(f"OSTRZEŻENIE: Niespodziewany kształt ND dla indeksów {key_indices}: {indices.shape}. Zwracam puste.")
                return torch.empty(
                    (3, 0), dtype=torch.long, device=target_device
                ), torch.empty((0,), dtype=torch.float32, device=target_device)

            batch_column = torch.full(
                (indices.shape[0], 1), batch_idx, dtype=torch.long, device=target_device
            )
            shifted_indices = torch.cat([batch_column, indices], dim=1)
            return (
                shifted_indices.T,  # Transponowane dla formatu COO (3, N)
                values,
            )
        else:  # Jeśli indices lub values są puste (np. shape (0,) lub (0,2) )
            return torch.empty(
                (3, 0),
                dtype=torch.long,
                device=target_device,  # Poprawka na (3,0) dla COO formatu
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
        # Filtruj puste tensory przed konkatenacją, aby uniknąć błędów z torch.cat
        # Puste tensory indeksów powinny mieć kształt (3,0)
        valid_indices_T = [
            idx
            for idx in all_indices_T
            if idx.numel() > 0 and idx.shape[0] == 3 and idx.shape[1] > 0
        ]
        valid_values = [
            val
            for val, idx in zip(all_values, all_indices_T)
            if idx.numel() > 0 and idx.shape[0] == 3 and idx.shape[1] > 0
        ]

        if not valid_indices_T:  # Jeśli po filtracji nic nie zostało
            return torch.zeros(size, dtype=torch.float32, device=target_device)

        indices_T = torch.cat(valid_indices_T, dim=1)
        values = torch.cat(valid_values, dim=0)

        sparse_tensor = torch.sparse_coo_tensor(
            indices_T, values, size=size, device=target_device
        )
        return sparse_tensor.coalesce().to_dense()

    notes_dense = create_dense_from_sparse(
        all_note_indices, all_note_values, (B, max_len, F_notes)
    )
    onsets_dense = create_dense_from_sparse(
        all_onset_indices,
        all_onset_values,
        (B, max_len, F_notes),  # Onsets mają tyle samo binów co nuty
    )
    contours_dense = create_dense_from_sparse(
        all_contour_indices, all_contour_values, (B, max_len, F_contours)
    )

    return {
        "track_ids": track_ids_list,
        "features": padded_features.to(target_device),
        "feature_lengths": torch.tensor(
            feature_lengths, dtype=torch.long, device=target_device
        ),
        "notes": notes_dense.to(target_device),
        "onsets": onsets_dense.to(target_device),
        "contours": contours_dense.to(target_device),
        "mask": mask.to(target_device),
    }
