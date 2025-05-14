import random
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple  # Dodano Tuple dla type hintingu

import librosa
import mirdata  # Narzędzie do pracy ze zbiorami danych muzycznych
import numpy as np
import torch
from tqdm import tqdm  # Biblioteka do wyświetlania pasków postępu

# Import konfiguracji i funkcji pomocniczych z bieżącego pakietu
from .config import (
    FREQ_BINS_NOTES,
    FREQ_BINS_CONTOURS,
    AUDIO_SAMPLE_RATE,
    ANNOTATION_HOP,
    N_FREQ_BINS_NOTES,
    N_FREQ_BINS_CONTOURS,
)
from .features import compute_cqt


def _process_sparse_annotations(
    indices: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:  # Poprawiony type hint na Tuple
    """
    Przetwarza rzadkie adnotacje (sparse annotations).

    Główne zadania to obsługa pustych danych wejściowych, binar_valuesyzacja wartości
    (wszystkie niezerowe wartości stają się 1.0) oraz zapewnienie unikalności
    wierszy w tablicy indeksów.

    Args:
        indices (np.ndarray): Tablica NumPy z indeksami (np. czas, częstotliwość).
                              Oczekiwany kształt (N, 2).
        values (np.ndarray): Tablica NumPy z wartościami odpowiadającymi indeksom.
                             Oczekiwany kształt (N,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Krotka zawierająca przetworzone tablice:
                                       - unikalne_indeksy (np.ndarray)
                                       - odpowiadające_im_zbinaryzowane_wartości (np.ndarray)
    """
    # Obsługa przypadku, gdy brakuje danych wejściowych (indeksów lub wartości)
    if values is None or indices is None or indices.shape[0] == 0:
        empty_indices = (
            np.empty((0, 2), dtype=int)
            if indices is None or indices.shape[0] == 0
            else indices.astype(
                int
            )  # Zapewnienie typu int, jeśli indices nie jest puste ale values jest None
        )
        empty_values = (
            np.empty((0,), dtype=float)
            if values is None or values.shape[0] == 0
            else values.astype(
                float
            )  # Zapewnienie typu float, jeśli values nie jest puste ale indices jest None
        )
        return empty_indices, empty_values

    # Binar_valuesyzacja wartości: wartości > 0 stają się 1.0, reszta 0.0
    processed_values = np.zeros_like(values, dtype=np.float32)
    processed_values[values > 0] = 1.0

    # Zachowanie tylko unikalnych wierszy w indeksach i odpowiadających im wartości
    # `np.unique` z `axis=0` zwraca unikalne wiersze oraz indeksy tych wierszy w oryginalnej tablicy
    unique_rows, unique_indices_idx = np.unique(indices, axis=0, return_index=True)

    final_indices = unique_rows
    final_values = processed_values[unique_indices_idx]

    return final_indices, final_values


def load_and_process_track(
    track: mirdata,
    resample_rate: int = AUDIO_SAMPLE_RATE,
) -> Dict[str, Any]:
    """
    Wczytuje ścieżkę audio z obiektu `track` (mirdata), przetwarza ją
    (oblicza spektrogram CQT) oraz dopasowuje czasowo adnotacje.

    Args:
        track (mirdata.Track): Obiekt ścieżki z biblioteki mirdata.
        resample_rate (int): Docelowa częstotliwość próbkowania audio.

    Returns:
        Dict[str, Any]: Słownik zawierający przetworzone dane:
                        cechy CQT, długość cech, rzadkie indeksy i wartości
                        dla nut, onsetów i konturów, oraz kształty docelowe.
    """
    # Wczytanie i resampling sygnału audio (mikrofonowego) do mono
    audio_mic, _ = librosa.load(track.audio_mic_path, sr=resample_rate, mono=True)

    # Normalizacja audio do zakresu [-1, 1]
    max_abs_val = np.max(np.abs(audio_mic))
    if max_abs_val > 1e-6:  # Unikanie dzielenia przez zero dla cichych sygnałów
        audio_mic = audio_mic / max_abs_val
    else:
        audio_mic = np.zeros_like(audio_mic)  # Jeśli sygnał jest praktycznie ciszą

    # Obliczenie spektrogramu CQT
    features_mic_tensor = compute_cqt(audio_mic)
    if (
        features_mic_tensor.ndim == 3
    ):  # Usunięcie ewentualnego wymiaru batcha, jeśli CQT go dodało
        features_mic_tensor = features_mic_tensor.squeeze(0)

    n_time_frames = features_mic_tensor.shape[0]  # Liczba ramek czasowych w CQT
    # Siatka czasowa dla adnotacji, zsynchronizowana z ramkami CQT
    time_grid = np.arange(n_time_frames) * ANNOTATION_HOP

    # Konwersja adnotacji mirdata (nuty, onSety, kontury) do formatu sparse
    # (indeksy czas-częstotliwość i odpowiadające im wartości)
    note_indices, note_values = track.notes_all.to_sparse_index(
        time_grid, "s", FREQ_BINS_NOTES, "hz"
    )
    onset_indices, onset_values = track.notes_all.to_sparse_index(
        time_grid, "s", FREQ_BINS_NOTES, "hz", onsets_only=True
    )
    contour_indices_raw, contour_values_raw = track.multif0.to_sparse_index(
        time_grid, "s", FREQ_BINS_CONTOURS, "hz"
    )

    # Dalsze przetwarzanie rzadkich adnotacji (binaryzacja, unikalność)
    note_indices, note_values = _process_sparse_annotations(note_indices, note_values)
    onset_indices, onset_values = _process_sparse_annotations(
        onset_indices, onset_values
    )

    # Przetwarzanie konturów, z obsługą przypadku braku danych
    if contour_values_raw is not None and contour_indices_raw is not None:
        contour_indices, contour_values = _process_sparse_annotations(
            contour_indices_raw, contour_values_raw
        )
    else:  # Jeśli brak adnotacji konturów, zwróć puste tablice
        contour_indices = np.empty((0, 2), dtype=int)
        contour_values = np.empty((0,), dtype=float)

    return {
        "features": features_mic_tensor.cpu().numpy(),  # Cechy CQT jako NumPy array
        "feature_length": n_time_frames,  # Długość sekwencji cech
        "note_indices": note_indices,  # Indeksy rzadkich nut
        "note_values": note_values,  # Wartości rzadkich nut (zbinaryzowane)
        "onset_indices": onset_indices,  # Indeksy rzadkich onsetów
        "onset_values": onset_values,  # Wartości rzadkich onsetów (zbinaryzowane)
        "contour_indices": contour_indices,  # Indeksy rzadkich konturów
        "contour_values": contour_values,  # Wartości rzadkich konturów (zbinaryzowane)
        "shape_notes": (
            n_time_frames,
            N_FREQ_BINS_NOTES,
        ),  # Docelowy kształt dla gęstych nut/onsetów
        "shape_contours": (
            n_time_frames,
            N_FREQ_BINS_CONTOURS,
        ),  # Docelowy kształt dla gęstych konturów
    }


def split_dataset(
    track_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Dzieli listę identyfikatorów ścieżek na zbiory: treningowy, walidacyjny i testowy.

    Podział jest wykonywany losowo, ale z ustalonym ziarnem dla powtarzalności.
    Stosunek testowy jest wyliczany jako reszta po odjęciu treningowego i walidacyjnego.

    Args:
        track_ids (List[str]): Lista wszystkich identyfikatorów ścieżek do podziału.
        train_ratio (float): Procentowy udział zbioru treningowego.
        val_ratio (float): Procentowy udział zbioru walidacyjnego.
        seed (int): Ziarno dla generatora liczb losowych.

    Returns:
        Dict[str, List[str]]: Słownik zawierający listy ID ścieżek dla każdego podziału
                              ('train', 'val', 'test').
    """
    random.seed(seed)  # Ustawienie ziarna dla powtarzalności losowania
    shuffled_ids = random.sample(track_ids, len(track_ids))  # Tasowanie ID ścieżek

    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Reszta trafia do zbioru testowego

    return {
        "train": shuffled_ids[:n_train],
        "val": shuffled_ids[n_train : n_train + n_val],
        "test": shuffled_ids[n_train + n_val :],  # Wszystkie pozostałe ID
    }


def save_track(data: Dict[str, Any], destination_dir: Path, track_id: str) -> None:
    """
    Zapisuje przetworzone dane pojedynczej ścieżki do pliku binarnego .pt.

    Jeśli katalog docelowy nie istnieje, zostanie utworzony.

    Args:
        data (Dict[str, Any]): Słownik z przetworzonymi danymi ścieżki.
        destination_dir (Path): Katalog, w którym ma być zapisany plik.
        track_id (str): Identyfikator ścieżki, używany jako nazwa pliku.
    """
    destination_dir.mkdir(
        parents=True, exist_ok=True
    )  # Utworzenie katalogu, jeśli nie istnieje
    filename = destination_dir / f"{track_id}.pt"
    # Użycie _use_new_zipfile_serialization=True dla potencjalnie lepszej kompatybilności/wydajności
    torch.save(data, filename, _use_new_zipfile_serialization=True)


def process_split(
    guitarset: mirdata,
    split_name: str,
    track_ids: List[str],
    output_dir: Path,
) -> None:
    """
    Przetwarza wszystkie ścieżki w danym podziale zbioru danych (np. 'train', 'val', 'test').

    Dla każdej ścieżki wczytuje ją, przetwarza, a następnie zapisuje wynik
    w odpowiednim podkatalogu w `output_dir`.

    Args:
        guitarset (mirdata.Dataset): Zainicjalizowany obiekt zbioru danych GuitarSet.
        split_name (str): Nazwa podziału (np. "train", "val", "test").
        track_ids (List[str]): Lista ID ścieżek należących do tego podziału.
        output_dir (Path): Główny katalog wyjściowy dla przetworzonych danych.
    """
    split_dir = output_dir / split_name  # Katalog dla konkretnego podziału

    # Iteracja przez ID ścieżek z użyciem paska postępu tqdm
    for track_id in tqdm(
        track_ids, desc=f"Przetwarzanie podziału '{split_name}'", unit="ścieżka"
    ):
        track = guitarset.track(track_id)  # Pobranie obiektu ścieżki z mirdata
        processed_data = load_and_process_track(track)  # Wczytanie i przetwarzanie
        save_track(processed_data, split_dir, track_id)  # Zapis przetworzonych danych


def process_dataset(
    data_dir: Union[str, Path] = "guitarset_data",
    output_dir: Union[str, Path] = "processed_data",
    seed: int = 42,
    max_tracks: Union[int, None] = None,
    overwrite: bool = False,
) -> None:
    """
    Główna funkcja do przetwarzania całego zbioru danych GuitarSet.

    Kroki:
    1. Inicjalizuje obiekt zbioru danych GuitarSet (mirdata).
    2. Wyklucza zdefiniowane problematyczne pliki.
    3. Opcjonalnie ogranicza liczbę przetwarzanych ścieżek.
    4. Dzieli zbiór na części treningową, walidacyjną i testową.
    5. Przetwarza i zapisuje każdą ścieżkę w odpowiednich podkatalogach.
    Obsługuje logikę nadpisywania istniejących danych.

    Args:
        data_dir (Union[str, Path]): Ścieżka do katalogu z danymi GuitarSet (dla mirdata).
        output_dir (Union[str, Path]): Ścieżka do katalogu, gdzie zostaną zapisane przetworzone pliki.
        seed (int): Ziarno losowości dla podziału danych i ewentualnego wyboru podzbioru ścieżek.
        max_tracks (Union[int, None]): Maksymalna liczba ścieżek do przetworzenia.
                                       Jeśli None, przetwarzane są wszystkie dostępne ścieżki.
        overwrite (bool): Jeśli True, istniejące przetworzone pliki mogą zostać nadpisane.
                          Jeśli False i katalog wyjściowy z plikami .pt istnieje, przetwarzanie jest pomijane.
    """
    output_dir_path = Path(output_dir)
    data_dir_path = Path(data_dir)

    # Sprawdzenie, czy przetwarzać dane, w zależności od parametru `overwrite` i zawartości katalogu wyjściowego
    if output_dir_path.exists() and not overwrite:
        if any(
            output_dir_path.rglob("*.pt")
        ):  # Sprawdzenie czy są jakiekolwiek pliki .pt
            print(
                f"Katalog wyjściowy '{output_dir_path}' istnieje i zawiera pliki .pt. "
                f"Parametr 'overwrite' ma wartość Fałsz. Pomijanie przetwarzania."
            )
            return  # Zakończenie funkcji, jeśli nie trzeba przetwarzać
        else:
            print(
                f"Katalog wyjściowy '{output_dir_path}' istnieje, ale jest pusty lub nie zawiera plików .pt. Kontynuowanie przetwarzania."
            )
    elif output_dir_path.exists() and overwrite:
        print(
            f"Parametr 'overwrite' ma wartość Prawda. Istniejące pliki w '{output_dir_path}' mogą zostać nadpisane."
        )

    output_dir_path.mkdir(
        parents=True, exist_ok=True
    )  # Utworzenie katalogu wyjściowego

    # Inicjalizacja obiektu GuitarSet z mirdata
    guitarset = mirdata.initialize("guitarset", data_home=str(data_dir_path))
    all_track_ids = guitarset.track_ids

    # Lista predefiniowanych problematycznych plików, które mają być wykluczone z przetwarzania
    problematic_files = ["04_BN3-154-E_comp", "04_Jazz1-200-B_comp"]
    all_track_ids = [
        track_id for track_id in all_track_ids if track_id not in problematic_files
    ]
    print(
        f"Po wykluczeniu problematycznych plików, znaleziono {len(all_track_ids)} ścieżek do przetworzenia."
    )

    # Opcjonalne ograniczenie liczby przetwarzanych ścieżek (np. do celów testowych)
    if max_tracks is not None and 0 < max_tracks < len(all_track_ids):
        print(f"Ograniczanie przetwarzania do {max_tracks} losowo wybranych ścieżek.")
        random.seed(seed)  # Użycie ziarna dla powtarzalności wyboru
        all_track_ids = random.sample(all_track_ids, max_tracks)

    # Podział ID ścieżek na zbiory treningowy, walidacyjny i testowy
    splits = split_dataset(all_track_ids, seed=seed)
    print(
        f"Podział danych: Treningowe: {len(splits['train'])}, "
        f"Walidacyjne: {len(splits['val'])}, Testowe: {len(splits['test'])}"
    )

    # Przetwarzanie każdego podziału
    for split_name, track_ids_list in splits.items():
        if (
            track_ids_list
        ):  # Sprawdzenie, czy lista ID dla danego podziału nie jest pusta
            print(f"Rozpoczynanie przetwarzania podziału: '{split_name}'")
            process_split(guitarset, split_name, track_ids_list, output_dir_path)
        else:
            print(f"Brak ścieżek do przetworzenia dla podziału: '{split_name}'.")

    print(
        f"Zakończono pomyślnie przetwarzanie zbioru danych. Wyniki w: '{output_dir_path}'."
    )
