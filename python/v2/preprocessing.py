# preprocessing.py

import os
import jams
import numpy as np
import torch
import librosa
import mirdata
from sklearn.model_selection import train_test_split
from tqdm import tqdm

OPEN_STRING_PITCHES = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

def prepare_track_splits(
    data_home,
    problematic_files,
    test_split_size,
    validation_split_size,
    random_seed,
    output_base_dir_for_id_files=None,
):
    """
    Przygotowuje podział identyfikatorów utworów na zbiory treningowe, walidacyjne i testowe.

    Args:
        data_home (str): Ścieżka do katalogu głównego danych GuitarSet.
        problematic_files (list): Lista identyfikatorów plików do wykluczenia.
        test_split_size (float): Procent danych przeznaczonych na zbiór testowy.
        validation_split_size (float): Procent danych przeznaczonych na zbiór walidacyjny (z całości).
        random_seed (int): Ziarno losowości dla powtarzalności podziału.
        output_base_dir_for_id_files (str, optional): Katalog do zapisu plików z ID utworów dla każdego podzbioru. Domyślnie None.

    Returns:
        dict: Słownik zawierający listy ID utworów dla 'train', 'validation' i 'test'.
              np. {'train': ['id1', 'id2'], 'validation': ['id3'], 'test': ['id4']}
    """
    try:
        # Inicjalizacja mirdata i wczytanie wszystkich ID utworów
        guitarset_loader = mirdata.initialize("guitarset", data_home=data_home)
        all_track_ids = guitarset_loader.track_ids
        print(f"Znaleziono {len(all_track_ids)} wszystkich track_id w GuitarSet.")
    except Exception as e:
        print(f"Błąd podczas inicjalizacji mirdata lub pobierania track_ids: {e}")
        return {"train": [], "validation": [], "test": []}

    # Filtrowanie problematycznych plików
    filtered_track_ids = []
    for track_id in all_track_ids:
        track_id_base = os.path.splitext(os.path.basename(track_id))[0]
        if track_id_base not in problematic_files:
            filtered_track_ids.append(track_id)
    print(
        f"Liczba track_id po usunięciu problematycznych plików: {len(filtered_track_ids)}"
    )

    track_ids_split = {"train": [], "validation": [], "test": []}

    if not filtered_track_ids:
        print("Brak track_id do przetworzenia po filtracji.")
        return track_ids_split

    # Podział na zbiór testowy i resztę (treningowo-walidacyjny)
    train_val_ids, test_ids = train_test_split(
        filtered_track_ids,
        test_size=test_split_size,
        random_state=random_seed,
        shuffle=True,
    )
    track_ids_split["test"] = test_ids

    # Podział reszty na zbiór treningowy i walidacyjny
    if (1 - test_split_size) > 0 and len(train_val_ids) > 0:
        # Obliczenie względnego rozmiaru zbioru walidacyjnego w stosunku do puli treningowo-walidacyjnej
        relative_val_split_size = (
            validation_split_size / (1 - test_split_size)
            if (1 - test_split_size) > 0
            else 0
        )
        if relative_val_split_size > 0 and len(train_val_ids) > 1: # Potrzebne co najmniej 2 próbki do podziału
            train_ids, validation_ids = train_test_split(
                train_val_ids,
                test_size=relative_val_split_size,
                random_state=random_seed,
                shuffle=True,
            )
            track_ids_split["train"] = train_ids
            track_ids_split["validation"] = validation_ids
        elif relative_val_split_size == 0 and len(train_val_ids) > 0: # Brak zbioru walidacyjnego
            track_ids_split["train"] = train_val_ids
            track_ids_split["validation"] = []
        elif len(train_val_ids) > 0: # Za mało danych na walidacyjny lub walidacyjny nie jest wymagany
            track_ids_split["train"] = train_val_ids
            track_ids_split["validation"] = []

    elif len(train_val_ids) > 0: # Jeśli cały zbiór testowy to 100%, reszta idzie do treningowego
        track_ids_split["train"] = train_val_ids
        track_ids_split["validation"] = []

    print(f"\nPodział na zbiory:")
    print(f"  Treningowy: {len(track_ids_split['train'])} utworów")
    print(f"  Walidacyjny: {len(track_ids_split['validation'])} utworów")
    print(f"  Testowy: {len(track_ids_split['test'])} utworów")

    # Zapis list ID utworów do plików tekstowych, jeśli podano katalog
    if output_base_dir_for_id_files:
        if not os.path.exists(output_base_dir_for_id_files):
            os.makedirs(output_base_dir_for_id_files, exist_ok=True)
        for split_name, ids in track_ids_split.items():
            split_specific_dir = os.path.join(output_base_dir_for_id_files, split_name)
            if not os.path.exists(split_specific_dir):
                os.makedirs(split_specific_dir, exist_ok=True) # Tworzenie podkatalogów dla każdego splitu
            with open(
                os.path.join(output_base_dir_for_id_files, f"{split_name}_ids.txt"), "w"
            ) as f:
                for track_id in ids:
                    f.write(f"{track_id}\n")
        print(
            f"\nZapisano listy track_id dla każdego zbioru w katalogu: {output_base_dir_for_id_files}"
        )
    return track_ids_split


def extract_annotations_from_jams(jams_path):
    """
    Ekstrahuje adnotacje dotyczące nut (początek, koniec, struna, próg, wysokość MIDI) z pliku JAMS.

    Args:
        jams_path (str): Ścieżka do pliku JAMS z adnotacjami.

    Returns:
        list: Lista krotek, gdzie każda krotka reprezentuje nutę i zawiera:
              (onset_sec, offset_sec, string_num, fret_num, pitch_midi).
              Zwraca pustą listę w przypadku braku odpowiednich adnotacji.
    """
    notes = []
    jam = jams.load(jams_path) # Wczytanie pliku JAMS
    note_midi_annotations = jam.search(namespace="note_midi") # Wyszukiwanie adnotacji typu 'note_midi'

    for annotation_obj in note_midi_annotations:
        # Sprawdzenie, czy adnotacja ma metadane i źródło danych (informację o strunie)
        if not (
            annotation_obj.annotation_metadata
            and hasattr(annotation_obj.annotation_metadata, "data_source")
        ):
            continue
        string_num = int(annotation_obj.annotation_metadata.data_source)
        # Pominięcie adnotacji dla strun spoza zdefiniowanego zakresu
        if string_num not in OPEN_STRING_PITCHES:
            continue

        open_string_pitch = OPEN_STRING_PITCHES[string_num]
        for obs in annotation_obj.data: # Iteracja po poszczególnych obserwacjach (nutach) w adnotacji
            onset_sec = float(obs.time)
            duration_sec = float(obs.duration)
            offset_sec = onset_sec + duration_sec
            pitch_midi = float(obs.value)
            # Obliczenie numeru progu na podstawie wysokości dźwięku i stroju pustej struny
            fret_num = int(round(pitch_midi - open_string_pitch))
            if fret_num < 0: # Korekta dla ewentualnych negatywnych wartości progów (np. z powodu odstrojenia)
                fret_num = 0
            notes.append((onset_sec, offset_sec, string_num, fret_num, pitch_midi))

    notes.sort(key=lambda x: x[0]) # Sortowanie nut według czasu rozpoczęcia
    return notes


def process_track(track_obj, output_file_basename, sr, n_fft, hop_length, n_mels):
    """
    Przetwarza pojedynczy utwór: wczytuje audio, ekstrahuje cechy (log-mel spektrogram)
    oraz etykiety (adnotacje nutowe), a następnie zapisuje je do plików.

    Args:
        track_obj (mirdata.Track): Obiekt utworu z mirdata.
        output_file_basename (str): Bazowa ścieżka (bez rozszerzenia) do zapisu plików wyjściowych.
        sr (int): Docelowa częstotliwość próbkowania audio.
        n_fft (int): Długość okna FFT.
        hop_length (int): Przesunięcie okna (hop size).
        n_mels (int): Liczba pasm Mel.

    Returns:
        str: Status przetwarzania: "processed", "skipped" (jeśli pliki już istnieją) lub "error".
    """
    features_path = f"{output_file_basename}_features.pt" # Ścieżka do pliku z cechami
    labels_path = f"{output_file_basename}_labels.pt"   # Ścieżka do pliku z etykietami

    # Pomiń, jeśli pliki cech i etykiet już istnieją
    if os.path.exists(features_path) and os.path.exists(labels_path):
        return "skipped"

    # Ustalenie ścieżki do pliku audio (preferowany mix, fallback na mic)
    audio_file_path = None
    if (
        hasattr(track_obj, "audio_mix_path")
        and track_obj.audio_mix_path
        and os.path.exists(track_obj.audio_mix_path)
    ):
        audio_file_path = track_obj.audio_mix_path
    elif (
        hasattr(track_obj, "audio_mic_path")
        and track_obj.audio_mic_path
        and os.path.exists(track_obj.audio_mic_path)
    ):
        audio_file_path = track_obj.audio_mic_path

    if not audio_file_path:
        print(f"  Błąd: Brak dostępnego pliku audio dla {track_obj.track_id}")
        return "error"

    try:
        # Wczytanie audio i konwersja do mono
        audio, _ = librosa.load(audio_file_path, sr=sr, mono=True)
        # Obliczenie mel-spektrogramu
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        # Konwersja na skalę logarytmiczną (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    except Exception as e:
        print(f"  Błąd podczas przetwarzania audio dla {track_obj.track_id}: {e}")
        return "error"

    # Sprawdzenie dostępności pliku JAMS
    if (
        not hasattr(track_obj, "jams_path")
        or not track_obj.jams_path
        or not os.path.exists(track_obj.jams_path)
    ):
        print(f"  Błąd: Brak pliku JAMS dla {track_obj.track_id}")
        return "error"

    try:
        # Ekstrakcja adnotacji z pliku JAMS
        annotations = extract_annotations_from_jams(track_obj.jams_path)
    except Exception as e:
        print(f"  Błąd podczas ekstrakcji adnotacji JAMS dla {track_obj.track_id}: {e}")
        return "error"

    if not annotations:
        print(f"  Błąd: Brak adnotacji w pliku JAMS dla {track_obj.track_id}")
        return "error"

    try:
        # Zapis cech (log-mel spektrogramu) jako tensora PyTorch
        torch.save(
            torch.tensor(log_mel_spectrogram, dtype=torch.float32), features_path
        )
        # Zapis etykiet (adnotacji) jako tensora PyTorch
        labels_array = np.array(annotations, dtype=np.float32)
        torch.save(torch.from_numpy(labels_array), labels_path)
        return "processed"
    except Exception as e:
        print(f"  Błąd podczas zapisywania cech/etykiet dla {track_obj.track_id}: {e}")
        return "error"


def preprocess_guitarset(
    data_home,
    output_base_dir,
    track_ids_split,
    sample_rate,
    n_fft,
    hop_length,
    n_mels,
):
    """
    Wykonuje pełny proces preprocessingu dla datasetu GuitarSet, iterując po
    podziałach na zbiory (treningowy, walidacyjny, testowy) i przetwarzając każdy utwór.

    Args:
        data_home (str): Ścieżka do katalogu głównego danych GuitarSet.
        output_base_dir (str): Główny katalog, w którym zostaną zapisane przetworzone dane (cechy i etykiety).
        track_ids_split (dict): Słownik z listami ID utworów dla 'train', 'validation', 'test'.
        sample_rate (int): Docelowa częstotliwość próbkowania audio.
        n_fft (int): Długość okna FFT.
        hop_length (int): Przesunięcie okna (hop size).
        n_mels (int): Liczba pasm Mel.

    Returns:
        None: Funkcja nie zwraca wartości, ale zapisuje pliki i drukuje podsumowanie.
    """
    print(f"Rozpoczynanie preprocessingu GuitarSet.")
    print(f"Katalog danych (data_home): {data_home}")
    print(f"Główny katalog wyjściowy: {output_base_dir}")

    try:
        # Inicjalizacja obiektu datasetu GuitarSet
        guitarset = mirdata.initialize("guitarset", data_home=data_home)
    except Exception as e:
        print(f"Krytyczny błąd: Nie udało się zainicjalizować GuitarSet. Błąd: {e}")
        return

    # Statystyki przetwarzania
    stats = {
        "train": {"processed": 0, "skipped": 0, "errors": 0},
        "validation": {"processed": 0, "skipped": 0, "errors": 0},
        "test": {"processed": 0, "skipped": 0, "errors": 0},
    }

    # Iteracja po każdym podzbiorze (train, validation, test)
    for split_name, track_id_list in track_ids_split.items():
        if not track_id_list:
            print(f"\nBrak utworów do przetworzenia w zbiorze: {split_name}")
            continue

        print(f"\nPrzetwarzanie zbioru: {split_name} ({len(track_id_list)} utworów)")

        # Tworzenie katalogu wyjściowego dla danego podzbioru
        split_output_dir = os.path.join(output_base_dir, split_name)
        if not os.path.exists(split_output_dir):
            try:
                os.makedirs(split_output_dir)
            except OSError as e:
                print(
                    f"Krytyczny błąd: Nie można utworzyć katalogu {split_output_dir}. Błąd: {e}"
                )
                continue # Przejdź do następnego splitu, jeśli nie można utworzyć katalogu

        # Iteracja po ID utworów w danym podzbiorze z paskiem postępu
        for track_id in tqdm(
            track_id_list,
            desc=f"Processing {split_name}",
            total=len(track_id_list),
            unit="track",
        ):
            try:
                track_obj = guitarset.track(track_id) # Pobranie obiektu utworu
                track_id_base = os.path.splitext(os.path.basename(track_id))[0] # Bazowa nazwa pliku
                output_file_basename = os.path.join(split_output_dir, track_id_base)

                # Przetwarzanie pojedynczego utworu
                status = process_track(
                    track_obj,
                    output_file_basename,
                    sample_rate,
                    n_fft,
                    hop_length,
                    n_mels,
                )
                # Aktualizacja statystyk
                if status == "processed":
                    stats[split_name]["processed"] += 1
                elif status == "skipped":
                    stats[split_name]["skipped"] += 1
                else:  # error
                    stats[split_name]["errors"] += 1
            except Exception as e:
                # Obsługa nieoczekiwanych błędów na poziomie pętli po utworach
                print(
                    f"  Nieoczekiwany błąd (poza process_track) dla utworu {track_id}: {e}"
                )
                stats[split_name]["errors"] += 1
        print() # Nowa linia po pasku postępu dla danego splitu

    # Wyświetlanie podsumowania preprocessingu
    print("\n--- Podsumowanie preprocessingu ---")
    total_tracks_in_split = sum(len(v) for v in track_ids_split.values())
    print(
        f"Liczba wszystkich utworów przeznaczonych do przetworzenia (po podziale): {total_tracks_in_split}"
    )
    for split_name_key in ["train", "validation", "test"]:
        if split_name_key in stats: # Wyświetl statystyki tylko dla istniejących splitów
            print(f"  Zbiór {split_name_key}:")
            print(f"    Nowo przetworzono: {stats[split_name_key]['processed']}")
            print(f"    Pominięto (już istniały): {stats[split_name_key]['skipped']}")
            print(f"    Błędy: {stats[split_name_key]['errors']}")
    print(f"Przetworzone dane zostały zapisane w katalogu: {output_base_dir}")