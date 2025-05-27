import torch

def validate_tensor_shapes_and_types(data, track_id, params):
    """
    Waliduje kształty i typy danych (dtype) tensorów w pojedynczej próbce danych.

    Args:
        data (dict): Słownik zawierający tensory do walidacji (np. 'cqt', 'mel', 'onsets').
        track_id (str): Identyfikator ścieżki, używany w komunikatach o błędach/ostrzeżeniach.
        params (dict): Słownik zawierający oczekiwane wymiary, np.:
                       {'N_BINS_CQT': int, 'N_MELS_MEL': int, 'N_PITCH_BINS': int, 'NUM_STRINGS': int}.

    Returns:
        bool: True, jeśli wszystkie sprawdzenia kształtów i typów zakończą się pomyślnie, w przeciwnym razie False.
    """
    valid = True
    num_frames = -1 # Liczba ramek, zostanie wywnioskowana z CQT lub Mel

    # Walidacja tensora CQT
    if 'cqt' in data and isinstance(data['cqt'], torch.Tensor):
        if data['cqt'].ndim == 2 and data['cqt'].shape[0] == params['N_BINS_CQT']:
            num_frames = data['cqt'].shape[1] # Ustalenie liczby ramek na podstawie CQT
            if data['cqt'].dtype != torch.float32:
                print(f"  [FAIL] {track_id} - cqt: typ danych to {data['cqt'].dtype}, oczekiwano torch.float32")
                valid = False
        else:
            print(f"  [FAIL] {track_id} - cqt: nieoczekiwany kształt {data['cqt'].shape}. Oczekiwano [{params['N_BINS_CQT']}, num_frames]")
            valid = False
    else:
        print(f"  [FAIL] {track_id} - cqt: brakujący klucz lub obiekt nie jest tensorem")
        valid = False

    # Walidacja tensora Mel-spektrogramu
    if 'mel' in data and isinstance(data['mel'], torch.Tensor):
        # Jeśli num_frames nie zostało ustalone z CQT, spróbuj z Mel (choć to mniej typowe)
        if num_frames == -1 and data['mel'].ndim == 2:
            num_frames = data['mel'].shape[1]

        if not (data['mel'].ndim == 2 and data['mel'].shape[0] == params['N_MELS_MEL'] and data['mel'].shape[1] == num_frames):
            print(f"  [FAIL] {track_id} - mel: nieoczekiwany kształt {data['mel'].shape}. Oczekiwano [{params['N_MELS_MEL']}, {num_frames}]")
            valid = False
        if data['mel'].dtype != torch.float32:
            print(f"  [FAIL] {track_id} - mel: typ danych to {data['mel'].dtype}, oczekiwano torch.float32")
            valid = False
    else:
        print(f"  [FAIL] {track_id} - mel: brakujący klucz lub obiekt nie jest tensorem")
        valid = False

    # Jeśli liczba ramek nadal nie jest znana, nie można kontynuować walidacji zależnych od ramek
    if num_frames == -1:
        print(f"  [FAIL] {track_id} - Nie można ustalić liczby ramek (num_frames) z CQT lub Mel. Pomijanie dalszych sprawdzeń zależnych od ramek.")
        return False

    # Walidacja tensora 'pitch_active'
    if 'pitch_active' in data and isinstance(data['pitch_active'], torch.Tensor):
        if not (data['pitch_active'].ndim == 2 and data['pitch_active'].shape[0] == num_frames and data['pitch_active'].shape[1] == params['N_PITCH_BINS']):
            print(f"  [FAIL] {track_id} - pitch_active: nieoczekiwany kształt {data['pitch_active'].shape}. Oczekiwano [{num_frames}, {params['N_PITCH_BINS']}]")
            valid = False
        if data['pitch_active'].dtype != torch.float32: # Oczekiwane dla BCEWithLogitsLoss
            print(f"  [FAIL] {track_id} - pitch_active: typ danych to {data['pitch_active'].dtype}, oczekiwano torch.float32")
            valid = False
    else:
        print(f"  [FAIL] {track_id} - pitch_active: brakujący klucz lub obiekt nie jest tensorem")
        valid = False

    # Walidacja tensora 'onsets'
    if 'onsets' in data and isinstance(data['onsets'], torch.Tensor):
        if not (data['onsets'].ndim == 2 and data['onsets'].shape[0] == num_frames and data['onsets'].shape[1] == params['N_PITCH_BINS']):
            print(f"  [FAIL] {track_id} - onsets: nieoczekiwany kształt {data['onsets'].shape}. Oczekiwano [{num_frames}, {params['N_PITCH_BINS']}]")
            valid = False
        if data['onsets'].dtype != torch.float32: # Oczekiwane dla BCEWithLogitsLoss
            print(f"  [FAIL] {track_id} - onsets: typ danych to {data['onsets'].dtype}, oczekiwano torch.float32")
            valid = False
    else:
        print(f"  [FAIL] {track_id} - onsets: brakujący klucz lub obiekt nie jest tensorem")
        valid = False

    # Walidacja tensora 'offsets'
    if 'offsets' in data and isinstance(data['offsets'], torch.Tensor):
        if not (data['offsets'].ndim == 2 and data['offsets'].shape[0] == num_frames and data['offsets'].shape[1] == params['N_PITCH_BINS']):
            print(f"  [FAIL] {track_id} - offsets: nieoczekiwany kształt {data['offsets'].shape}. Oczekiwano [{num_frames}, {params['N_PITCH_BINS']}]")
            valid = False
        if data['offsets'].dtype != torch.float32: # Oczekiwane dla BCEWithLogitsLoss, jeśli modelowane
            print(f"  [FAIL] {track_id} - offsets: typ danych to {data['offsets'].dtype}, oczekiwano torch.float32")
            valid = False
    else:
        print(f"  [FAIL] {track_id} - offsets: brakujący klucz lub obiekt nie jest tensorem")
        valid = False

    # Walidacja tensora 'string_contours'
    if 'string_contours' in data and isinstance(data['string_contours'], torch.Tensor):
        if not (data['string_contours'].ndim == 2 and data['string_contours'].shape[0] == num_frames and data['string_contours'].shape[1] == params['NUM_STRINGS']):
            print(f"  [FAIL] {track_id} - string_contours: nieoczekiwany kształt {data['string_contours'].shape}. Oczekiwano [{num_frames}, {params['NUM_STRINGS']}]")
            valid = False
        if data['string_contours'].dtype != torch.float32:
            print(f"  [FAIL] {track_id} - string_contours: typ danych to {data['string_contours'].dtype}, oczekiwano torch.float32")
            valid = False
    else:
        print(f"  [FAIL] {track_id} - string_contours: brakujący klucz lub obiekt nie jest tensorem")
        valid = False

    if num_frames == 0: # Ostrzeżenie, jeśli ścieżka nie ma ramek
        print(f"  [WARN] {track_id} - Ścieżka ma 0 ramek. Może to być problematyczne.")
        # Niekoniecznie błąd krytyczny, ale warto odnotować

    return valid

def validate_tensor_values(data, track_id):
    """
    Waliduje zakresy wartości oraz podstawowe statystyki (NaN, Inf) tensorów w pojedynczej próbce danych.

    Args:
        data (dict): Słownik zawierający tensory do walidacji.
        track_id (str): Identyfikator ścieżki, używany w komunikatach.

    Returns:
        bool: True, jeśli wszystkie sprawdzenia wartości zakończą się pomyślnie (lub tylko z ostrzeżeniami),
              False w przypadku krytycznych błędów wartości.
    """
    valid = True
    # Sprawdzenie tensorów binarnych (powinny zawierać tylko 0.0 lub 1.0)
    for key in ['pitch_active', 'onsets', 'offsets']:
        if key in data and isinstance(data[key], torch.Tensor):
            tensor = data[key]
            # Sprawdzenie, czy wszystkie wartości to 0.0 lub 1.0
            if not (torch.all((tensor == 0.0) | (tensor == 1.0))):
                print(f"  [FAIL] {track_id} - {key}: wartości nie są wyłącznie 0.0 lub 1.0. Min: {tensor.min()}, Max: {tensor.max()}")
                valid = False
            # Ostrzeżenie, jeśli tensor (poza offsets) zawiera same zera
            if tensor.sum() == 0 and key != 'offsets':
                print(f"  [WARN] {track_id} - {key}: wszystkie wartości to 0. Brak wykrytych {key} w tej ścieżce.")
        elif key not in data: # Ten przypadek powinien być już obsłużony przez validate_tensor_shapes_and_types
            pass # Błąd braku klucza zostałby już zgłoszony

    # Sprawdzenie tensora 'string_contours' (wartości powinny być nieujemne)
    if 'string_contours' in data and isinstance(data['string_contours'], torch.Tensor):
        tensor = data['string_contours']
        if torch.any(tensor < 0.0): # Sprawdzenie, czy istnieją wartości ujemne
            print(f"  [FAIL] {track_id} - string_contours: zawiera wartości ujemne. Min: {tensor.min()}")
            valid = False
        if tensor.sum() == 0.0: # Ostrzeżenie, jeśli wszystkie wartości to zero
            print(f"  [WARN] {track_id} - string_contours: wszystkie wartości to 0. Brak wykrytych konturów.")
    elif 'string_contours' not in data:
        pass

    # Sprawdzenie tensorów cech (CQT, Mel) pod kątem NaN lub Inf
    for key in ['cqt', 'mel']:
        if key in data and isinstance(data[key], torch.Tensor):
            tensor = data[key]
            if torch.isnan(tensor).any(): # Sprawdzenie obecności NaN
                print(f"  [FAIL] {track_id} - {key}: zawiera wartości NaN.")
                valid = False
            if torch.isinf(tensor).any(): # Sprawdzenie obecności Inf
                print(f"  [FAIL] {track_id} - {key}: zawiera wartości Inf.")
                valid = False
        elif key not in data:
            pass

    return valid

def run_full_validation(dataset, validation_params):
    """
    Przeprowadza pełną walidację danych w dostarczonym datasecie.

    Iteruje po wszystkich próbkach w datasecie, sprawdzając obecność oczekiwanych kluczy,
    kształty i typy tensorów oraz zakresy ich wartości. Gromadzi również podstawowe
    statystyki dotyczące danych. Wyniki walidacji i statystyki są drukowane na konsolę.

    Args:
        dataset (torch.utils.data.Dataset): Instancja datasetu PyTorch do walidacji.
                                             Oczekuje się, że `dataset[i]` zwraca słownik.
        validation_params (dict): Słownik zawierający parametry niezbędne do walidacji
                                  kształtów (np. N_BINS_CQT, N_MELS_MEL, N_PITCH_BINS, NUM_STRINGS).
    """
    print(f"\n--- Rozpoczynanie Pełnej Walidacji Danych ({len(dataset)} ścieżek) ---")
    if len(dataset) == 0:
        print("Dataset jest pusty. Nic do walidacji.")
        return

    all_tracks_valid_shape_type = True # Flaga ogólnej poprawności kształtów/typów
    all_tracks_valid_values = True     # Flaga ogólnej poprawności wartości

    # Inicjalizacja liczników statystyk
    total_frames = 0
    total_onsets = 0
    total_active_pitches = 0
    total_offsets = 0
    frames_with_any_onset = 0
    frames_with_any_active_pitch = 0
    frames_with_any_contour = 0

    min_frames = float('inf')
    max_frames = 0
    track_id_min_frames = ""
    track_id_max_frames = ""

    for i in range(len(dataset)): # Iteracja po wszystkich próbkach w datasecie
        try:
            data_item = dataset[i] # Pobranie próbki
            # Próba uzyskania ID ścieżki, jeśli dataset udostępnia taką metodę lub atrybut
            track_id = data_item.get('track_id',
                                      dataset.get_track_id(i) if hasattr(dataset, 'get_track_id') else
                                      (dataset.track_ids_base[i] if hasattr(dataset, 'track_ids_base') and i < len(dataset.track_ids_base) else f"unknown_track_{i}")
                                     )
            print(f"\nWalidacja ścieżki {i+1}/{len(dataset)}: {track_id}")

            # 1. Walidacja obecności kluczy w próbce danych
            expected_keys = ['track_id', 'cqt', 'mel', 'pitch_active', 'onsets', 'offsets', 'string_contours']
            keys_ok = True
            for key in expected_keys:
                if key not in data_item:
                    print(f"  [FAIL] {track_id} - Brakujący klucz: {key}")
                    keys_ok = False
            if not keys_ok:
                all_tracks_valid_shape_type = False # Traktuj brak klucza jako błąd strukturalny
                continue # Przejdź do następnej próbki

            # 2. Walidacja kształtów i typów tensorów
            if not validate_tensor_shapes_and_types(data_item, track_id, validation_params):
                all_tracks_valid_shape_type = False
                # Można zdecydować, czy kontynuować do walidacji wartości, jeśli kształty są złe

            # 3. Walidacja wartości tensorów
            if not validate_tensor_values(data_item, track_id):
                all_tracks_valid_values = False

            # 4. Zbieranie statystyk (tylko jeśli podstawowe struktury są obecne)
            if 'cqt' in data_item and isinstance(data_item['cqt'], torch.Tensor) and data_item['cqt'].ndim == 2:
                current_frames = data_item['cqt'].shape[1]
                total_frames += current_frames
                if current_frames < min_frames: min_frames = current_frames; track_id_min_frames = track_id
                if current_frames > max_frames: max_frames = current_frames; track_id_max_frames = track_id

                if 'onsets' in data_item and isinstance(data_item['onsets'], torch.Tensor):
                    onsets_sum = data_item['onsets'].sum().item()
                    total_onsets += onsets_sum
                    # Liczba ramek, w których występuje co najmniej jeden onset
                    frames_with_any_onset += (data_item['onsets'].sum(dim=1) > 0).sum().item()

                if 'pitch_active' in data_item and isinstance(data_item['pitch_active'], torch.Tensor):
                    total_active_pitches += data_item['pitch_active'].sum().item()
                    frames_with_any_active_pitch += (data_item['pitch_active'].sum(dim=1) > 0).sum().item()

                if 'offsets' in data_item and isinstance(data_item['offsets'], torch.Tensor):
                    total_offsets += data_item['offsets'].sum().item()

                if 'string_contours' in data_item and isinstance(data_item['string_contours'], torch.Tensor):
                    # Liczba ramek, gdzie co najmniej jedna struna ma niezerową częstotliwość konturu
                    frames_with_any_contour += (data_item['string_contours'] > 0).any(dim=1).sum().item()

        except Exception as e: # Obsługa błędów podczas ładowania/przetwarzania próbki
            track_id_for_error = f"unknown_track_index_{i}"
            if hasattr(dataset, 'get_track_id'): track_id_for_error = dataset.get_track_id(i)
            elif hasattr(dataset, 'track_ids_base') and i < len(dataset.track_ids_base): track_id_for_error = dataset.track_ids_base[i]

            print(f"  [FATAL ERROR] {track_id_for_error} - Nie można załadować lub przetworzyć próbki: {e}")
            all_tracks_valid_shape_type = False # Błąd ładowania jest traktowany jako krytyczny

    # Wydrukowanie podsumowania walidacji
    print("\n--- Podsumowanie Walidacji ---")
    if len(dataset) > 0:
        print(f"Przetworzono {len(dataset)} ścieżek.")
        if all_tracks_valid_shape_type:
            print("[OK] Wszystkie przetworzone ścieżki mają poprawne klucze, kształty tensorów i typy danych.")
        else:
            print("[BŁĄD] Znaleziono problemy z kluczami, kształtami tensorów lub typami danych w niektórych ścieżkach.")

        if all_tracks_valid_values:
            print("[OK] Wszystkie tensory w przetworzonych ścieżkach mają poprawne zakresy wartości (binarne, nieujemne itp.).")
        else:
            print("[BŁĄD/OSTRZEŻENIE] Znaleziono problemy z wartościami tensorów w niektórych ścieżkach (np. poza zakresem, same zera).")

        print("\nStatystyki Ogólne:")
        print(f"  Całkowita liczba ramek we wszystkich poprawnie przetworzonych ścieżkach: {total_frames}")
        if len(dataset) > 0 and total_frames > 0 : # Unikaj dzielenia przez zero
            print(f"  Średnia liczba ramek na ścieżkę: {total_frames / len(dataset):.2f}")
        print(f"  Minimalna liczba ramek w ścieżce: {min_frames if min_frames != float('inf') else 'N/A'} (ścieżka: {track_id_min_frames})")
        print(f"  Maksymalna liczba ramek w ścieżce: {max_frames if max_frames > 0 else 'N/A'} (ścieżka: {track_id_max_frames})")

        print("\nStatystyki Etykiet (dla poprawnie przetworzonych ścieżek):")
        print(f"  Całkowita liczba zdarzeń onsetów: {total_onsets}")
        print(f"  Całkowita liczba zdarzeń aktywnych pitchy: {total_active_pitches}")
        print(f"  Całkowita liczba zdarzeń offsetów: {total_offsets}")
        if total_frames > 0: # Unikaj dzielenia przez zero
            print(f"  Procent ramek z co najmniej jednym onsetem: {(frames_with_any_onset / total_frames) * 100:.2f}%")
            print(f"  Procent ramek z co najmniej jednym aktywnym pitchem: {(frames_with_any_active_pitch / total_frames) * 100:.2f}%")
            print(f"  Procent ramek z co najmniej jednym aktywnym konturem struny: {(frames_with_any_contour / total_frames) * 100:.2f}%")
        else:
            print("  Brak ramek do obliczenia procentowych statystyk etykiet.")
    else:
        print("Dataset jest pusty.")
    print("--- Koniec Walidacji ---")