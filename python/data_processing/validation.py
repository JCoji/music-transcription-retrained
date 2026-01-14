import torch


def validate_tensor_shapes_and_types(data_sample, track_identifier, shape_params):
    is_valid = True
    calculated_num_frames = -1

    if 'cqt' in data_sample and isinstance(data_sample['cqt'], torch.Tensor):
        if data_sample['cqt'].ndim == 2 and data_sample['cqt'].shape[0] == shape_params['N_BINS_CQT']:
            calculated_num_frames = data_sample['cqt'].shape[1]
            if data_sample['cqt'].dtype != torch.float32:
                print(
                    f"  [FAIL] {track_identifier} - cqt: typ danych to {data_sample['cqt'].dtype}, oczekiwano torch.float32")
                is_valid = False
        else:
            print(
                f"  [FAIL] {track_identifier} - cqt: nieoczekiwany kształt {data_sample['cqt'].shape}. Oczekiwano [{shape_params['N_BINS_CQT']}, num_frames]")
            is_valid = False
    else:
        print(f"  [FAIL] {track_identifier} - cqt: brakujący klucz lub obiekt nie jest tensorem")
        is_valid = False

    if calculated_num_frames == -1:
        print(f"  [FAIL] {track_identifier} - Nie można ustalić liczby ramek (calculated_num_frames) z CQT.")
        return False

    for data_key, expected_shape_dim1_key, expected_tensor_dtype in [
        ('pitch_active', 'N_PITCH_BINS', torch.float32),
        ('onsets', 'N_PITCH_BINS', torch.float32),
        ('offsets', 'N_PITCH_BINS', torch.float32),
        ('string_contours', 'NUM_STRINGS', torch.float32)
    ]:
        if data_key in data_sample and isinstance(data_sample[data_key], torch.Tensor):
            tensor_to_check = data_sample[data_key]
            expected_dim1_size = shape_params[expected_shape_dim1_key]
            if not (tensor_to_check.ndim == 2 and tensor_to_check.shape[0] == calculated_num_frames and
                    tensor_to_check.shape[1] == expected_dim1_size):
                print(
                    f"  [FAIL] {track_identifier} - {data_key}: nieoczekiwany kształt {tensor_to_check.shape}. Oczekiwano [{calculated_num_frames}, {expected_dim1_size}]")
                is_valid = False
            if tensor_to_check.dtype != expected_tensor_dtype:
                print(
                    f"  [FAIL] {track_identifier} - {data_key}: typ danych to {tensor_to_check.dtype}, oczekiwano {expected_tensor_dtype}")
                is_valid = False
        else:
            print(f"  [FAIL] {track_identifier} - {data_key}: brakujący klucz lub obiekt nie jest tensorem")
            is_valid = False

    if calculated_num_frames == 0:
        print(f"  [WARN] {track_identifier} - Ścieżka ma 0 ramek.")
    return is_valid


def validate_tensor_values(data_sample, track_identifier):
    is_valid = True
    for data_key in ['pitch_active', 'onsets', 'offsets']:
        if data_key in data_sample and isinstance(data_sample[data_key], torch.Tensor):
            tensor_to_check = data_sample[data_key]
            if not (torch.all((tensor_to_check == 0.0) | (tensor_to_check == 1.0))):
                print(
                    f"  [FAIL] {track_identifier} - {data_key}: wartości nie są wyłącznie 0.0 lub 1.0. Min: {tensor_to_check.min()}, Max: {tensor_to_check.max()}")
                is_valid = False
            if tensor_to_check.sum() == 0 and data_key != 'offsets':
                print(f"  [WARN] {track_identifier} - {data_key}: wszystkie wartości to 0.")

    if 'string_contours' in data_sample and isinstance(data_sample['string_contours'], torch.Tensor):
        tensor_to_check = data_sample['string_contours']
        if torch.any(tensor_to_check < 0.0):
            print(
                f"  [FAIL] {track_identifier} - string_contours: zawiera wartości ujemne. Min: {tensor_to_check.min()}")
            is_valid = False
        if tensor_to_check.sum() == 0.0:
            print(f"  [WARN] {track_identifier} - string_contours: wszystkie wartości to 0.")

    if 'cqt' in data_sample and isinstance(data_sample['cqt'], torch.Tensor):
        tensor_to_check = data_sample['cqt']
        if torch.isnan(tensor_to_check).any():
            print(f"  [FAIL] {track_identifier} - cqt: zawiera wartości NaN.")
            is_valid = False
        if torch.isinf(tensor_to_check).any():
            print(f"  [FAIL] {track_identifier} - cqt: zawiera wartości Inf.")
            is_valid = False
    return is_valid


def run_full_data_validation(dataset_to_validate, validation_shape_params):
    print(f"\n--- Rozpoczynanie Pełnej Walidacji Danych ({len(dataset_to_validate)} ścieżek) ---")
    if not dataset_to_validate:
        print("Dataset jest pusty. Nic do walidacji.")
        return

    all_items_valid_shape_type = True
    all_items_valid_values = True
    validation_stats = {
        'total_frames': 0, 'total_onsets': 0, 'total_active_pitches': 0,
        'total_offsets': 0, 'frames_with_any_onset': 0,
        'frames_with_any_active_pitch': 0, 'frames_with_any_contour': 0,
        'min_frames': float('inf'), 'max_frames': 0,
        'track_id_min_frames': "", 'track_id_max_frames': ""
    }

    for item_idx in range(len(dataset_to_validate)):
        try:
            current_data_item = dataset_to_validate[item_idx]
            current_track_id = current_data_item.get('track_id', f"unknown_track_idx_{item_idx}")
            if hasattr(dataset_to_validate, 'get_track_id_for_item'):
                current_track_id = dataset_to_validate.get_track_id_for_item(item_idx)
            elif hasattr(dataset_to_validate, 'base_track_ids') and item_idx < len(dataset_to_validate.base_track_ids):
                current_track_id = dataset_to_validate.base_track_ids[item_idx]

            print(f"\nWalidacja ścieżki {item_idx + 1}/{len(dataset_to_validate)}: {current_track_id}")

            required_keys = ['track_id', 'cqt', 'pitch_active', 'onsets', 'offsets', 'string_contours']
            if not all(key in current_data_item for key in required_keys):
                missing_keys_list = [key for key in required_keys if key not in current_data_item]
                print(f"  [FAIL] {current_track_id} - Brakujące klucze: {', '.join(missing_keys_list)}")
                all_items_valid_shape_type = False
                continue

            if not validate_tensor_shapes_and_types(current_data_item, current_track_id, validation_shape_params):
                all_items_valid_shape_type = False
            if not validate_tensor_values(current_data_item, current_track_id):
                all_items_valid_values = False

            if 'cqt' in current_data_item and isinstance(current_data_item['cqt'], torch.Tensor) and current_data_item[
                'cqt'].ndim == 2:
                num_frames_in_item = current_data_item['cqt'].shape[1]
                validation_stats['total_frames'] += num_frames_in_item
                if num_frames_in_item < validation_stats['min_frames']:
                    validation_stats['min_frames'] = num_frames_in_item
                    validation_stats['track_id_min_frames'] = current_track_id
                if num_frames_in_item > validation_stats['max_frames']:
                    validation_stats['max_frames'] = num_frames_in_item
                    validation_stats['track_id_max_frames'] = current_track_id

                for key_name, total_stat_key, frames_any_stat_key in [
                    ('onsets', 'total_onsets', 'frames_with_any_onset'),
                    ('pitch_active', 'total_active_pitches', 'frames_with_any_active_pitch'),
                    ('offsets', 'total_offsets', None)
                ]:
                    if key_name in current_data_item and isinstance(current_data_item[key_name], torch.Tensor):
                        validation_stats[total_stat_key] += current_data_item[key_name].sum().item()
                        if frames_any_stat_key:
                            validation_stats[frames_any_stat_key] += (
                                        current_data_item[key_name].sum(dim=1) > 0).sum().item()

                if 'string_contours' in current_data_item and isinstance(current_data_item['string_contours'],
                                                                         torch.Tensor):
                    validation_stats['frames_with_any_contour'] += (current_data_item['string_contours'] > 0).any(
                        dim=1).sum().item()

        except Exception as e:
            error_track_id = f"unknown_track_index_{item_idx}"
            if hasattr(dataset_to_validate, 'get_track_id_for_item'):
                error_track_id = dataset_to_validate.get_track_id_for_item(item_idx)
            elif hasattr(dataset_to_validate, 'base_track_ids') and item_idx < len(dataset_to_validate.base_track_ids):
                error_track_id = dataset_to_validate.base_track_ids[item_idx]
            print(f"  [FATAL ERROR] {error_track_id} - Nie można załadować lub przetworzyć próbki: {e}")
            all_items_valid_shape_type = False

    print("\n--- Podsumowanie Walidacji ---")
    if not dataset_to_validate:
        print("Dataset jest pusty.")
        return

    print(f"Przetworzono {len(dataset_to_validate)} ścieżek.")
    print(
        "[OK] Wszystkie przetworzone ścieżki mają poprawne klucze, kształty tensorów i typy danych." if all_items_valid_shape_type else "[BŁĄD] Znaleziono problemy z kluczami, kształtami tensorów lub typami danych.")
    print(
        "[OK] Wszystkie tensory w przetworzonych ścieżkach mają poprawne zakresy wartości." if all_items_valid_values else "[BŁĄD/OSTRZEŻENIE] Znaleziono problemy z wartościami tensorów.")

    print("\nStatystyki Ogólne:")
    print(f"  Całkowita liczba ramek: {validation_stats['total_frames']}")
    if len(dataset_to_validate) > 0 and validation_stats['total_frames'] > 0:
        print(f"  Średnia liczba ramek na ścieżkę: {validation_stats['total_frames'] / len(dataset_to_validate):.2f}")
    print(
        f"  Min ramek: {validation_stats['min_frames'] if validation_stats['min_frames'] != float('inf') else 'N/A'} (ścieżka: {validation_stats['track_id_min_frames']})")
    print(
        f"  Max ramek: {validation_stats['max_frames'] if validation_stats['max_frames'] > 0 else 'N/A'} (ścieżka: {validation_stats['track_id_max_frames']})")

    print("\nStatystyki Etykiet:")
    print(f"  Łącznie onsets: {validation_stats['total_onsets']}")
    print(f"  Łącznie active pitches: {validation_stats['total_active_pitches']}")
    print(f"  Łącznie offsets: {validation_stats['total_offsets']}")
    if validation_stats['total_frames'] > 0:
        for stat_key, friendly_name in [
            ('frames_with_any_onset', 'onsetem'),
            ('frames_with_any_active_pitch', 'aktywnym pitchem'),
            ('frames_with_any_contour', 'aktywnym konturem struny')
        ]:
            print(
                f"  Procent ramek z co najmniej jednym {friendly_name}: {(validation_stats[stat_key] / validation_stats['total_frames']) * 100:.2f}%")
    else:
        print("  Brak ramek do obliczenia procentowych statystyk etykiet.")
    print("--- Koniec Walidacji ---")