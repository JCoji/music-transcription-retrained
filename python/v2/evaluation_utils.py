import os
import torch
import numpy as np
import shutil
import mirdata

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    print("Ostrzeżenie: Biblioteka pretty_midi nie jest zainstalowana. Generowanie plików MIDI nie będzie możliwe.")

OPEN_STRING_PITCHES_MIDI = {
    0: 40,  # E2
    1: 45,  # A2
    2: 50,  # D3
    3: 55,  # G3
    4: 59,  # B3
    5: 64   # E4
}

def load_best_model(model_class, model_init_params, model_path, device):
    """
    Wczytuje zapisany stan modelu PyTorch.

    Args:
        model_class (torch.nn.Module): Klasa modelu do zainicjalizowania.
        model_init_params (dict): Słownik parametrów inicjalizacyjnych dla klasy modelu.
        model_path (str): Ścieżka do zapisanego pliku stanu modelu (.pth lub .pt).
        device (torch.device): Urządzenie (CPU lub GPU), na które ma być załadowany model.

    Returns:
        torch.nn.Module or None: Wczytany model w trybie ewaluacji lub None w przypadku błędu.
    """
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu w {model_path}")
        return None
    try:
        # Inicjalizacja instancji modelu
        loaded_model = model_class(**model_init_params)
        # Wczytanie zapisanych wag modelu
        # Użycie weights_only=True dla bezpieczeństwa, jeśli plik jest zaufany
        loaded_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        # Przeniesienie modelu na określone urządzenie
        loaded_model.to(device)
        # Przełączenie modelu w tryb ewaluacji (ważne dla warstw jak Dropout, BatchNorm)
        loaded_model.eval()
        print(f"Model pomyślnie wczytany z {model_path} i przełączony w tryb ewaluacji.")
        return loaded_model
    except Exception as e:
        print(f"Błąd podczas wczytywania lub inicjalizacji modelu z {model_path}: {e}")
        import traceback
        traceback.print_exc() # Wypisanie pełnego śladu stosu dla lepszego debugowania
        return None

def find_optimal_onset_threshold(model_to_eval, dataloader, device,
                                 min_thresh=0.05, max_thresh=0.95, step=0.01):
    """
    Znajduje optymalny próg dla detekcji onsetów na podstawie miary F1-score.

    Args:
        model_to_eval (torch.nn.Module): Model do ewaluacji.
        dataloader (torch.utils.data.DataLoader): DataLoader dostarczający dane walidacyjne/testowe.
        device (torch.device): Urządzenie, na którym model wykonuje obliczenia.
        min_thresh (float, optional): Minimalna wartość progu do przetestowania. Domyślnie 0.05.
        max_thresh (float, optional): Maksymalna wartość progu do przetestowania. Domyślnie 0.95.
        step (float, optional): Krok zmiany progu. Domyślnie 0.01.

    Returns:
        tuple: (optimal_threshold, best_f1_score)
               - optimal_threshold (float): Znaleziony optymalny próg.
               - best_f1_score (float): Najlepszy uzyskany F1-score dla tego progu.
    """
    print("\nRozpoczynanie wyszukiwania optymalnego progu dla onsetów...")
    all_onset_probs_list = []
    all_onset_targets_list = []
    model_to_eval.eval() # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad(): # Wyłączenie obliczania gradientów
        for i, (features, labels_tuple) in enumerate(dataloader):
            features = features.to(device)
            onset_targets_batch, _ = labels_tuple # Interesują nas tylko etykiety onsetów
            # Predykcja modelu
            onset_logits, _ = model_to_eval(features)
            onset_probs_batch = torch.sigmoid(onset_logits) # Konwersja logitów na prawdopodobieństwa
            # Gromadzenie predykcji i etykiet
            all_onset_probs_list.append(onset_probs_batch.cpu().reshape(-1))
            all_onset_targets_list.append(onset_targets_batch.cpu().reshape(-1))

    if not all_onset_probs_list:
        print("Brak danych do analizy progu. Zwracam domyślny próg 0.5.")
        return 0.5, 0.0

    try:
        # Konkatenacja wyników z wszystkich batchy
        flat_onset_probs = torch.cat(all_onset_probs_list)
        flat_onset_targets = torch.cat(all_onset_targets_list)
    except RuntimeError as e: # Obsługa błędu, gdyby listy były puste lub tensory miały niekompatybilne kształty
        print(f"Błąd RuntimeError podczas konkatenacji tensorów: {e}\nZwracam domyślny próg 0.5.")
        return 0.5, 0.0

    print(f"Zebrano {flat_onset_probs.shape[0]} punktów danych (ramka*struna) do optymalizacji progu.")
    possible_thresholds = np.arange(min_thresh, max_thresh + step, step)
    best_f1_score = -1.0
    optimal_threshold = 0.5 # Domyślna wartość

    print("Testowanie różnych progów dla onsetów...")
    # Iteracja po możliwych wartościach progów
    for threshold_val in possible_thresholds:
        onset_preds_binary = (flat_onset_probs > threshold_val).float()
        # Obliczanie True Positives, False Positives, False Negatives
        tp = ((onset_preds_binary == 1) & (flat_onset_targets == 1)).sum().item()
        fp = ((onset_preds_binary == 1) & (flat_onset_targets == 0)).sum().item()
        fn = ((onset_preds_binary == 0) & (flat_onset_targets == 1)).sum().item()
        # Obliczanie precyzji i czułości
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Obliczanie F1-score
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
        if f1 > best_f1_score:
            best_f1_score = f1
            optimal_threshold = threshold_val

    print(f"Optymalny próg dla onsetów znaleziony: {optimal_threshold:.2f} (F1-score: {best_f1_score:.4f})")
    return optimal_threshold, best_f1_score

def evaluate_model_on_test_set(
        model_to_eval, test_dataloader, device,
        fret_num_classes, optimal_onset_threshold,
):
    """
    Ewaluuje model na zbiorze testowym, obliczając metryki dla detekcji onsetów i klasyfikacji progów.

    Args:
        model_to_eval (torch.nn.Module): Model do ewaluacji.
        test_dataloader (torch.utils.data.DataLoader): DataLoader dla zbioru testowego.
        device (torch.device): Urządzenie (CPU/GPU).
        fret_num_classes (int): Liczba klas progów (wliczając klasę "cisza").
        optimal_onset_threshold (float): Optymalny próg dla detekcji onsetów.

    Returns:
        dict: Słownik zawierający obliczone metryki (precyzja, czułość, F1, dokładność dla onsetów;
              dokładność ogólna i dla aktywnych ramek dla progów).
    """
    print(f"\nRozpoczynanie ewaluacji na zbiorze testowym z progiem onsetów: {optimal_onset_threshold:.2f}...")
    model_to_eval.eval() # Tryb ewaluacji
    # Inicjalizacja liczników metryk
    onset_tp_total = 0; onset_fp_total = 0; onset_fn_total = 0; onset_tn_total = 0
    fret_correct_predictions_total = 0; fret_total_elements = 0
    fret_correct_predictions_active = 0; fret_total_active_frames_total = 0

    with torch.no_grad(): # Bez obliczania gradientów
        for features, labels_tuple in test_dataloader:
            features = features.to(device)
            onset_targets_batch, fret_targets_batch = labels_tuple
            onset_targets_batch = onset_targets_batch.to(device)
            fret_targets_batch = fret_targets_batch.to(device)

            # Predykcje modelu
            onset_logits, fret_logits = model_to_eval(features)
            onset_probs = torch.sigmoid(onset_logits)
            onset_preds_binary_batch = (onset_probs > optimal_onset_threshold).float()

            # Agregacja statystyk dla onsetów
            onset_tp_total += ((onset_preds_binary_batch == 1) & (onset_targets_batch == 1)).sum().item()
            onset_fp_total += ((onset_preds_binary_batch == 1) & (onset_targets_batch == 0)).sum().item()
            onset_fn_total += ((onset_preds_binary_batch == 0) & (onset_targets_batch == 1)).sum().item()
            onset_tn_total += ((onset_preds_binary_batch == 0) & (onset_targets_batch == 0)).sum().item()

            # Agregacja statystyk dla progów
            fret_pred_indices_batch = torch.argmax(fret_logits, dim=-1)
            fret_correct_predictions_total += (fret_pred_indices_batch == fret_targets_batch).sum().item()
            fret_total_elements += fret_targets_batch.numel()

            # Statystyki dla progów tylko w aktywnych ramkach (gdzie nie ma ciszy w ground truth)
            mask_active_batch = (fret_targets_batch != (fret_num_classes - 1)) # Ostatnia klasa to "cisza"
            fret_correct_predictions_active += ((fret_pred_indices_batch == fret_targets_batch) & mask_active_batch).sum().item()
            fret_total_active_frames_total += mask_active_batch.sum().item()

    # Obliczanie finalnych metryk dla onsetów
    onset_precision = onset_tp_total / (onset_tp_total + onset_fp_total) if (onset_tp_total + onset_fp_total) > 0 else 0.0
    onset_recall = onset_tp_total / (onset_tp_total + onset_fn_total) if (onset_tp_total + onset_fn_total) > 0 else 0.0
    onset_f1 = 2 * (onset_precision * onset_recall) / (onset_precision + onset_recall) if (onset_precision + onset_recall) > 0 else 0.0
    total_onset_elements_for_accuracy = onset_tp_total + onset_tn_total + onset_fp_total + onset_fn_total
    onset_accuracy_corrected = (onset_tp_total + onset_tn_total) / total_onset_elements_for_accuracy if total_onset_elements_for_accuracy > 0 else 0.0

    # Obliczanie finalnych metryk dla progów
    fret_accuracy_overall = fret_correct_predictions_total / fret_total_elements if fret_total_elements > 0 else 0.0
    fret_accuracy_active = fret_correct_predictions_active / fret_total_active_frames_total if fret_total_active_frames_total > 0 else 0.0

    test_metrics = {
        "test_onset_precision": onset_precision, "test_onset_recall": onset_recall,
        "test_onset_f1": onset_f1, "test_onset_accuracy": onset_accuracy_corrected,
        "test_fret_accuracy_overall": fret_accuracy_overall, "test_fret_accuracy_active": fret_accuracy_active
    }

    print("\n--- Wyniki ewaluacji na zbiorze testowym ---")
    for name, value in test_metrics.items(): print(f"  {name}: {value:.4f}")
    return test_metrics

def convert_frames_to_notes(onset_preds_binary, fret_pred_indices,
                            hop_length, sample_rate, max_frets,
                            min_note_duration_frames=2):
    """
    Konwertuje sekwencje predykcji (onsety i progi w ramkach) na listę obiektów nut MIDI.

    Args:
        onset_preds_binary (torch.Tensor): Tensor binarnych predykcji onsetów [liczba_ramek, liczba_strun].
        fret_pred_indices (torch.Tensor): Tensor predykcji indeksów progów [liczba_ramek, liczba_strun].
        hop_length (int): Przesunięcie okna analizy (w próbkach).
        sample_rate (int): Częstotliwość próbkowania audio (w Hz).
        max_frets (int): Maksymalny numer progu uwzględniany (np. 20). Klasa ciszy to max_frets + 1.
        min_note_duration_frames (int, optional): Minimalna długość nuty w ramkach, aby została uwzględniona. Domyślnie 2.

    Returns:
        list: Lista słowników, gdzie każdy słownik zawiera 'note_obj' (pretty_midi.Note) i 'string' (indeks struny).
              Zwraca pustą listę, jeśli pretty_midi nie jest dostępne lub nie wykryto nut.
    """
    if not PRETTY_MIDI_AVAILABLE:
        print("  pretty_midi nie jest dostępne. Nie można konwertować na nuty MIDI.")
        return []

    num_frames, num_strings = onset_preds_binary.shape
    time_per_frame = hop_length / sample_rate # Czas trwania jednej ramki w sekundach
    notes_list = []
    silence_fret_idx = max_frets + 1 # Indeks klasy oznaczającej ciszę

    for s_idx in range(num_strings): # Iteracja po każdej strunie
        active_note_start_frame = None # Ramka rozpoczęcia aktualnie aktywnej nuty
        active_note_fret = None      # Próg aktualnie aktywnej nuty

        for frame_idx in range(num_frames): # Iteracja po każdej ramce czasowej
            is_onset = onset_preds_binary[frame_idx, s_idx].item() > 0.5 # Czy w tej ramce jest onset
            current_fret = fret_pred_indices[frame_idx, s_idx].item()   # Przewidziany próg w tej ramce

            # Jeśli nuta jest aktualnie aktywna, sprawdź, czy powinna zostać zakończona
            if active_note_start_frame is not None:
                terminate_note = False
                if is_onset and frame_idx > active_note_start_frame : # Nowy onset na tej samej strunie kończy poprzednią nutę
                    terminate_note = True
                elif current_fret == silence_fret_idx: # Zmiana na próg "cisza" kończy nutę
                    terminate_note = True
                elif frame_idx == num_frames - 1: # Koniec utworu, zakończ wszystkie aktywne nuty
                    terminate_note = True
                # Można dodać inne warunki, np. zmiana progu bez onsetu (legato)

                if terminate_note:
                    start_time = active_note_start_frame * time_per_frame
                    end_time = frame_idx * time_per_frame # Nuta kończy się tuż przed bieżącą ramką
                    duration_frames = frame_idx - active_note_start_frame

                    if duration_frames >= min_note_duration_frames and active_note_fret != silence_fret_idx:
                        pitch_val = OPEN_STRING_PITCHES_MIDI[s_idx] + active_note_fret
                        note = pretty_midi.Note(
                            velocity=100, # Domyślna głośność
                            pitch=int(round(pitch_val)),
                            start=start_time,
                            end=end_time
                        )
                        notes_list.append({'note_obj': note, 'string': s_idx})
                    # Zresetuj stan aktywnej nuty
                    active_note_start_frame = None
                    active_note_fret = None

            # Jeśli wykryto onset i nie jest to onset "ciszy", rozpocznij nową nutę
            if is_onset and current_fret != silence_fret_idx:
                # Jeśli poprzednia nuta była aktywna i nie została zakończona (np. przez brak onsetu, ale próg się zmienił),
                # zakończ ją teraz. Ta logika jest częściowo pokryta powyżej, ale dla pewności.
                if active_note_start_frame is not None and frame_idx > active_note_start_frame:
                    start_time = active_note_start_frame * time_per_frame
                    end_time = frame_idx * time_per_frame
                    duration_frames = frame_idx - active_note_start_frame
                    if duration_frames >= min_note_duration_frames and active_note_fret != silence_fret_idx:
                        pitch_val = OPEN_STRING_PITCHES_MIDI[s_idx] + active_note_fret
                        note = pretty_midi.Note(
                            velocity=100, pitch=int(round(pitch_val)),
                            start=start_time, end=end_time
                        )
                        notes_list.append({'note_obj': note, 'string': s_idx})

                # Rozpocznij nową nutę
                active_note_start_frame = frame_idx
                active_note_fret = current_fret

        # Jeśli ostatnia nuta na strunie była aktywna do samego końca utworu
        if active_note_start_frame is not None:
            start_time = active_note_start_frame * time_per_frame
            end_time = num_frames * time_per_frame # Nuta trwa do końca ostatniej ramki
            duration_frames = num_frames - active_note_start_frame
            if duration_frames >= min_note_duration_frames and active_note_fret != silence_fret_idx:
                pitch_val = OPEN_STRING_PITCHES_MIDI[s_idx] + active_note_fret
                note = pretty_midi.Note(
                    velocity=100, pitch=int(round(pitch_val)),
                    start=start_time, end=end_time
                )
                notes_list.append({'note_obj': note, 'string': s_idx})
    return notes_list


def generate_midi_from_predictions(
        model_to_eval, dataset, device, sample_indices,
        optimal_onset_threshold,
        sr, hop_length, max_frets,
        output_midi_dir,
        mirdata_data_home,
        guitarset_loader_instance=None
):
    """
    Generuje pliki MIDI na podstawie predykcji modelu dla wybranych próbek z datasetu.
    Kopiuje również oryginalne pliki WAV dla porównania.

    Args:
        model_to_eval (torch.nn.Module): Ewaluowany model.
        dataset (torch.utils.data.Dataset): Dataset zawierający cechy i etykiety.
        device (torch.device): Urządzenie (CPU/GPU).
        sample_indices (list of int): Lista indeksów próbek z datasetu do przetworzenia.
        optimal_onset_threshold (float): Optymalny próg dla detekcji onsetów.
        sr (int): Częstotliwość próbkowania audio.
        hop_length (int): Przesunięcie okna analizy.
        max_frets (int): Maksymalny numer progu.
        output_midi_dir (str): Katalog do zapisu wygenerowanych plików MIDI i skopiowanych WAV.
        mirdata_data_home (str): Ścieżka do katalogu głównego danych mirdata (np. GuitarSet).
        guitarset_loader_instance (mirdata.Loader, optional): Wstępnie zainicjalizowana instancja loadera GuitarSet.
                                                              Domyślnie None (zostanie zainicjalizowana wewnątrz).
    Returns:
        None: Funkcja zapisuje pliki na dysku.
    """
    if not PRETTY_MIDI_AVAILABLE:
        print("Biblioteka pretty_midi nie jest dostępna. Pomijam generowanie plików MIDI.")
        return

    if not os.path.exists(output_midi_dir):
        os.makedirs(output_midi_dir, exist_ok=True)
        print(f"Utworzono katalog: {output_midi_dir}")

    # Inicjalizacja loadera GuitarSet, jeśli nie został przekazany
    if guitarset_loader_instance is None:
        try:
            guitarset_loader = mirdata.initialize('guitarset', data_home=mirdata_data_home)
        except Exception as e:
            print(f"Błąd inicjalizacji mirdata dla GuitarSet: {e}. Nie można skopiować plików WAV.")
            guitarset_loader = None # Ustawienie na None, aby uniknąć dalszych błędów
    else:
        guitarset_loader = guitarset_loader_instance

    model_to_eval.eval() # Tryb ewaluacji
    with torch.no_grad(): # Bez gradientów
        for sample_idx in sample_indices:
            if sample_idx >= len(dataset):
                print(f"Indeks {sample_idx} poza zakresem datasetu ({len(dataset)}). Pomijam.")
                continue

            features_sample, _ = dataset[sample_idx] # Etykiety GT nie są tu potrzebne
            # Próba uzyskania bazowego ID utworu z datasetu
            track_id_base = "unknown_track"
            if hasattr(dataset, 'track_ids_base') and dataset.track_ids_base and sample_idx < len(dataset.track_ids_base):
                track_id_base = dataset.track_ids_base[sample_idx]

            print(f"\nPrzetwarzanie próbki {sample_idx} (ID: {track_id_base}) do MIDI...")

            # Przygotowanie danych wejściowych dla modelu
            features_sample_dev = features_sample.unsqueeze(0).to(device) # Dodanie wymiaru batch
            # Predykcja modelu
            onset_pred_logits, fret_pred_logits = model_to_eval(features_sample_dev)

            # Przetwarzanie wyjść modelu
            onset_pred_probs = torch.sigmoid(onset_pred_logits.squeeze(0).cpu()) # Usunięcie wymiaru batch, do CPU
            onset_pred_binary = (onset_pred_probs > optimal_onset_threshold).float()
            fret_pred_indices = torch.argmax(fret_pred_logits.squeeze(0).cpu(), dim=-1)

            # Konwersja predykcji ramkowych na listę nut
            predicted_notes_info = convert_frames_to_notes(
                onset_pred_binary, fret_pred_indices,
                hop_length, sr, max_frets
            )

            if not predicted_notes_info:
                print(f"  Nie wygenerowano żadnych nut dla próbki {track_id_base}.")
            else:
                # Tworzenie obiektu MIDI
                midi_data = pretty_midi.PrettyMIDI(initial_tempo=120.0) # Domyślne tempo
                guitar_instrument = pretty_midi.Instrument(program=25) # 25 = Acoustic Guitar (steel)

                for note_info in predicted_notes_info:
                    guitar_instrument.notes.append(note_info['note_obj'])

                midi_data.instruments.append(guitar_instrument)

                # Zapis pliku MIDI
                midi_filename = f"{track_id_base}_prediction.mid"
                midi_filepath = os.path.join(output_midi_dir, midi_filename)
                try:
                    midi_data.write(midi_filepath)
                    print(f"  Zapisano przewidziane MIDI do: {midi_filepath} ({len(guitar_instrument.notes)} nut)")
                except Exception as e:
                    print(f"  Błąd podczas zapisywania pliku MIDI {midi_filepath}: {e}")

            # Kopiowanie oryginalnego pliku audio (WAV)
            if guitarset_loader:
                try:
                    full_track_id_for_mirdata = None
                    # Próba uzyskania pełnego ID ścieżki na różne sposoby
                    if hasattr(dataset, 'get_full_track_id') and callable(getattr(dataset, 'get_full_track_id')):
                        full_track_id_for_mirdata = dataset.get_full_track_id(sample_idx)
                    elif hasattr(dataset, 'track_ids_full') and dataset.track_ids_full and sample_idx < len(dataset.track_ids_full):
                        full_track_id_for_mirdata = dataset.track_ids_full[sample_idx]
                    else: # Jeśli dataset nie ma tych atrybutów, spróbuj znaleźć ID w loaderze
                        potential_ids = [tid for tid in guitarset_loader.track_ids if track_id_base in tid]
                        if potential_ids:
                            full_track_id_for_mirdata = potential_ids[0] # Weź pierwsze pasujące

                    if full_track_id_for_mirdata:
                        track_obj = guitarset_loader.track(full_track_id_for_mirdata)
                        original_audio_path = None
                        # Sprawdzenie dostępności ścieżek audio (mix lub mic)
                        if hasattr(track_obj, 'audio_mix_path') and track_obj.audio_mix_path and os.path.exists(track_obj.audio_mix_path):
                            original_audio_path = track_obj.audio_mix_path
                        elif hasattr(track_obj, 'audio_mic_path') and track_obj.audio_mic_path and os.path.exists(track_obj.audio_mic_path):
                            original_audio_path = track_obj.audio_mic_path

                        if original_audio_path:
                            wav_filename = f"{track_id_base}_original.wav"
                            wav_filepath = os.path.join(output_midi_dir, wav_filename)
                            shutil.copy2(original_audio_path, wav_filepath) # Kopiowanie pliku
                            print(f"  Skopiowano oryginalny WAV do: {wav_filepath}")
                        else:
                            print(f"  Nie znaleziono oryginalnego pliku audio dla {track_id_base} (pełny ID: {full_track_id_for_mirdata}).")
                    else:
                        print(f"  Nie udało się ustalić pełnego track_id dla {track_id_base}, aby skopiować WAV.")

                except Exception as e:
                    print(f"  Błąd podczas kopiowania oryginalnego pliku WAV dla {track_id_base}: {e}")
    print("\nZakończono generowanie plików MIDI.")


def generate_test_set_visualizations(
        model_to_eval, test_dataset, device, num_samples_to_plot,
        optimal_onset_threshold,
        sr, hop_length, max_frets,
        artifacts_dir
):
    """
    Generuje wizualizacje (wykresy) porównujące predykcje modelu z ground truth
    dla wybranych próbek ze zbioru testowego.

    Args:
        model_to_eval (torch.nn.Module): Ewaluowany model.
        test_dataset (torch.utils.data.Dataset): Dataset testowy.
        device (torch.device): Urządzenie (CPU/GPU).
        num_samples_to_plot (int): Liczba próbek do zwizualizowania.
        optimal_onset_threshold (float): Optymalny próg dla detekcji onsetów.
        sr (int): Częstotliwość próbkowania audio.
        hop_length (int): Przesunięcie okna analizy.
        max_frets (int): Maksymalny numer progu.
        artifacts_dir (str): Katalog do zapisu wygenerowanych plików graficznych.

    Returns:
        None: Funkcja zapisuje pliki na dysku.
    """
    if not (test_dataset and len(test_dataset) > 0):
        print("Brak danych testowych do wizualizacji.")
        return
    try:
        import plotting_utils # Moduł z funkcjami do rysowania (zakładamy jego istnienie)
    except ImportError:
        print("Moduł plotting_utils nie jest dostępny. Pomijam generowanie wizualizacji.")
        return

    print(f"\nGenerowanie wizualizacji dla {num_samples_to_plot} próbek testowych...")
    model_to_eval.eval() # Tryb ewaluacji
    with torch.no_grad(): # Bez gradientów
        for i in range(min(num_samples_to_plot, len(test_dataset))): # Iteruj po N pierwszych próbkach
            features_sample, labels_gt_tuple = test_dataset[i]
            # Ustalenie ID ścieżki dla nazwy pliku
            track_id_for_plot = f"test_sample_{i}"
            if hasattr(test_dataset, 'track_ids_base') and test_dataset.track_ids_base and i < len(test_dataset.track_ids_base):
                track_id_for_plot = test_dataset.track_ids_base[i]

            # Przygotowanie danych i predykcja
            features_sample_dev = features_sample.unsqueeze(0).to(device)
            onset_gt_sample_cpu = labels_gt_tuple[0].cpu()
            fret_gt_sample_cpu = labels_gt_tuple[1].cpu()
            onset_pred_logits, fret_pred_logits = model_to_eval(features_sample_dev)

            # Ścieżka zapisu wykresu
            plot_save_path = os.path.join(artifacts_dir, f"final_test_prediction_{track_id_for_plot.replace('/','_')}.png")
            try:
                # Wywołanie funkcji rysującej
                plotting_utils.plot_predictions_vs_ground_truth(
                    features_sample=features_sample.cpu(),
                    onset_gt_sample=onset_gt_sample_cpu,
                    fret_gt_sample=fret_gt_sample_cpu,
                    onset_pred_logits_sample=onset_pred_logits.squeeze(0).cpu(),
                    fret_pred_logits_sample=fret_pred_logits.squeeze(0).cpu(),
                    sr=sr, hop_length=hop_length, max_frets=max_frets,
                    onset_threshold=optimal_onset_threshold,
                    track_id_base=track_id_for_plot,
                    save_path=plot_save_path
                )
            except Exception as e:
                print(f"    Błąd podczas generowania wizualizacji dla próbki {track_id_for_plot}: {e}")


def _generate_tab_slot_content( # Funkcja pomocnicza
        is_onset_active,
        fret_value,
        is_note_sustained,
        max_frets,
        slot_char_width=3
):
    """
    Generuje reprezentację tekstową pojedynczego "slotu" w tabulaturze.

    Args:
        is_onset_active (bool): Czy w tym slocie (lub jego części) występuje onset.
        fret_value (int): Wartość progu (0-max_frets) lub indeks ciszy.
        is_note_sustained (bool): Czy nuta jest podtrzymywana z poprzedniego slotu.
        max_frets (int): Maksymalny numer progu.
        slot_char_width (int, optional): Szerokość znakowa slotu. Domyślnie 3.

    Returns:
        str: Ciąg znaków reprezentujący slot tabulatury (np. "5--", "12-", "---").
    """
    silence_fret_idx = max_frets + 1

    if is_onset_active:
        if 0 <= fret_value <= max_frets:
            # Wyświetl numer progu, jeśli jest onset i próg jest prawidłowy
            return str(fret_value).ljust(slot_char_width, '-')[:slot_char_width]
        else:  # Onset, ale próg to cisza (lub nieprawidłowy) - traktuj jako myślnik
            return "-" * slot_char_width
    elif is_note_sustained:  # Brak onsetu, ale nuta jest podtrzymywana
        return "-" * slot_char_width
    else:  # Cisza lub brak podtrzymania
        return "-" * slot_char_width


def _generate_tab_matrix_slots( # Funkcja pomocnicza
        onset_data,
        fret_data,
        num_frames,
        num_strings,
        max_frets,
        frames_per_slot=2,
        slot_char_width=3
):
    """
    Generuje macierz (lista list) zawartości slotów tabulatury na podstawie danych ramkowych.

    Args:
        onset_data (torch.Tensor): Tensor binarnych predykcji/GT onsetów [liczba_ramek, liczba_strun].
        fret_data (torch.Tensor): Tensor predykcji/GT indeksów progów [liczba_ramek, liczba_strun].
        num_frames (int): Całkowita liczba ramek.
        num_strings (int): Liczba strun.
        max_frets (int): Maksymalny numer progu.
        frames_per_slot (int, optional): Liczba oryginalnych ramek przypadająca na jeden slot tabulatury. Domyślnie 2.
        slot_char_width (int, optional): Szerokość znakowa pojedynczego slotu. Domyślnie 3.

    Returns:
        list: Macierz 2D (lista list), gdzie tab_matrix_slots[indeks_struny_modelu][indeks_slotu]
              zawiera tekstową reprezentację slotu.
    """
    silence_fret_idx = max_frets + 1
    # Obliczenie liczby slotów w tabulaturze
    num_slots = (num_frames + frames_per_slot - 1) // frames_per_slot

    # Inicjalizacja macierzy slotów
    tab_matrix_slots = [[""] * num_slots for _ in range(num_strings)]

    # Śledzenie, czy nuta na danej strunie jest aktualnie aktywna (brzmi)
    # i jaki próg jest aktywny, aby poprawnie wyświetlać podtrzymanie '-'
    current_active_fret_on_string = [-1] * num_strings  # -1 oznacza brak aktywnego dźwięku

    for slot_idx in range(num_slots): # Iteracja po slotach tabulatury
        start_frame_of_slot = slot_idx * frames_per_slot
        end_frame_of_slot = min(start_frame_of_slot + frames_per_slot, num_frames)

        for s_model_idx in range(num_strings):  # Iteracja po strunach (0=najniższa E, ..., 5=najwyższa e)
            onset_in_this_slot_for_string = False
            fret_at_onset_in_slot = -1 # Próg w miejscu onsetu w tym slocie

            # Sprawdź, czy w ramkach należących do tego slotu jest onset dla tej struny
            for frame_k in range(start_frame_of_slot, end_frame_of_slot):
                if onset_data[frame_k, s_model_idx].item() > 0.5: # Jeśli jest onset
                    fret_val = fret_data[frame_k, s_model_idx].item()
                    if 0 <= fret_val <= max_frets: # Sprawdź czy próg jest "grywalny"
                        fret_at_onset_in_slot = fret_val
                    else: # Onset, ale próg to cisza (lub błąd)
                        fret_at_onset_in_slot = silence_fret_idx # Oznacz jako ciszę
                    onset_in_this_slot_for_string = True
                    break # Pierwszy onset w slocie jest decydujący

            slot_content_str = ""
            if onset_in_this_slot_for_string: # Jeśli w slocie był onset
                if fret_at_onset_in_slot != silence_fret_idx: # Jeśli onset nie był ciszą
                    slot_content_str = str(fret_at_onset_in_slot)
                    current_active_fret_on_string[s_model_idx] = fret_at_onset_in_slot # Zapamiętaj aktywny próg
                else:  # Onset ciszy
                    slot_content_str = "-"
                    current_active_fret_on_string[s_model_idx] = -1 # Zresetuj aktywny próg
            else:  # Brak onsetu w tym slocie
                # Sprawdź, czy nuta jest podtrzymywana z poprzedniej klatki (lub początku tego slotu)
                fret_at_slot_start = fret_data[start_frame_of_slot, s_model_idx].item()
                if 0 <= fret_at_slot_start <= max_frets:
                    # Jeśli próg na początku slotu jest taki sam jak ostatnio aktywny próg, to jest to podtrzymanie
                    if current_active_fret_on_string[s_model_idx] == fret_at_slot_start:
                        slot_content_str = "-"  # Podtrzymanie
                    else:
                        # To może być nowa nuta bez wyraźnego onsetu (np. legato) lub błąd.
                        # Dla uproszczenia, jeśli fret jest aktywny, ale nie było onsetu, a nie jest to
                        # kontynuacja poprzedniego dźwięku, resetujemy.
                        slot_content_str = "-" # Domyślnie cisza, jeśli nie ma jasnego podtrzymania
                        current_active_fret_on_string[s_model_idx] = -1
                else:  # Cisza na początku slotu
                    slot_content_str = "-"
                    current_active_fret_on_string[s_model_idx] = -1 # Zresetuj aktywny próg

            # Formatowanie stringa slotu do odpowiedniej szerokości
            if len(slot_content_str) > slot_char_width: # np. "10" dla slot_char_width=1, lub "12-" dla szer=2
                slot_content_str = slot_content_str[:slot_char_width - 1] + ">" # Oznacz ucięcie
            elif slot_content_str.isdigit(): # Jeśli to cyfra (próg)
                slot_content_str = slot_content_str.ljust(slot_char_width, '-') # Wypełnij myślnikami z prawej
            else:  # Jeśli to już myślnik (cisza/podtrzymanie)
                slot_content_str = slot_content_str * slot_char_width # Powiel myślnik

            tab_matrix_slots[s_model_idx][slot_idx] = slot_content_str
    return tab_matrix_slots


def _format_tab_matrix_into_text_block( # Funkcja pomocnicza
        tab_matrix_slots,
        num_strings,
        line_break_after_slots=20
):
    """
    Formatuje macierz slotów tabulatury w czytelny, wieloliniowy blok tekstowy.

    Args:
        tab_matrix_slots (list of list of str): Macierz slotów wygenerowana przez _generate_tab_matrix_slots.
                                                tab_matrix_slots[indeks_struny_modelu][indeks_slotu].
        num_strings (int): Liczba strun (np. 6).
        line_break_after_slots (int, optional): Liczba slotów w jednej linii wizualizacji. Domyślnie 20.

    Returns:
        str: Sformatowany blok tekstowy tabulatury.
    """
    output_lines = []
    # Nazwy strun do wyświetlania (wysokie 'e' na górze, niskie 'E' na dole)
    string_names_display = ["e", "B", "G", "D", "A", "E"] if num_strings == 6 else [str(i) for i in range(num_strings)]

    if not tab_matrix_slots or not tab_matrix_slots[0]:
        return "Brak danych do sformatowania."

    num_total_slots = len(tab_matrix_slots[0])

    # Iteracja po segmentach slotów (aby łamać linie)
    for slot_start_idx in range(0, num_total_slots, line_break_after_slots):
        slot_end_idx = min(slot_start_idx + line_break_after_slots, num_total_slots)

        # Iteracja po strunach w kolejności wyświetlania (od najwyższej do najniższej)
        for display_s_idx in range(num_strings):
            # Mapowanie indeksu struny wyświetlanej na indeks struny w modelu
            # Model: 0=Low E, ..., 5=High e
            # Wyświetlanie: 0=High e, ..., 5=Low E
            model_s_idx = (num_strings - 1) - display_s_idx

            line = string_names_display[display_s_idx] + "|" # Nazwa struny na początku linii
            current_slot_count_in_line = 0
            for slot_idx in range(slot_start_idx, slot_end_idx): # Iteracja po slotach w bieżącym segmencie
                line += tab_matrix_slots[model_s_idx][slot_idx]
                current_slot_count_in_line += 1
                # Dodaj separator "|" co 4 sloty dla czytelności, ale nie na samym końcu linii
                if current_slot_count_in_line % 4 == 0 and slot_idx < slot_end_idx - 1:
                    line += "|"
            output_lines.append(line)
        output_lines.append("")  # Pusta linia między segmentami tabulatury

    return "\n".join(output_lines)


def generate_text_tablature_comparison(
        model_to_eval,
        dataset,
        device,
        sample_indices,
        optimal_onset_threshold,
        sr,
        hop_length,
        max_frets,
        num_strings=6,
        output_dir=None,
        track_id_prefix="sample", # Zmieniono domyślny prefix
        frames_per_slot=2,
        slot_char_width=3,
        line_break_after_slots=20
):
    """
    Generuje i zapisuje tabulatury tekstowe dla ground truth i predykcji modelu
    dla wybranych próbek.

    Args:
        model_to_eval (torch.nn.Module): Ewaluowany model.
        dataset (torch.utils.data.Dataset): Dataset zawierający cechy i etykiety.
        device (torch.device): Urządzenie (CPU/GPU).
        sample_indices (list of int): Lista indeksów próbek z datasetu do przetworzenia.
        optimal_onset_threshold (float): Optymalny próg dla detekcji onsetów.
        sr (int): Częstotliwość próbkowania (nieużywana bezpośrednio w tej funkcji, ale może być w przyszłości).
        hop_length (int): Przesunięcie okna (nieużywane bezpośrednio w tej funkcji).
        max_frets (int): Maksymalny numer progu.
        num_strings (int, optional): Liczba strun. Domyślnie 6.
        output_dir (str, optional): Katalog do zapisu plików tabulatur. Jeśli None, nic nie jest zapisywane.
        track_id_prefix (str, optional): Prefix dla nazw plików i ID utworów. Domyślnie "sample".
        frames_per_slot (int, optional): Liczba ramek na slot tabulatury. Domyślnie 2.
        slot_char_width (int, optional): Szerokość znakowa slotu. Domyślnie 3.
        line_break_after_slots (int, optional): Liczba slotów na linię. Domyślnie 20.

    Returns:
        None: Funkcja zapisuje pliki na dysku.
    """
    # sr i hop_length nie są bezpośrednio używane w logice generowania tabulatury z już przetworzonych ramek,
    # ale są zachowane dla spójności interfejsu, gdyby były potrzebne np. do konwersji czasu.

    if not (dataset and len(dataset) > 0):
        print("Brak danych do generowania tabulatur tekstowych.")
        return

    if output_dir is None:
        print("Nie podano katalogu wyjściowego, tabulatury tekstowe nie zostaną zapisane.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Utworzono katalog dla tabulatur tekstowych: {output_dir}")

    model_to_eval.eval() # Tryb ewaluacji
    with torch.no_grad(): # Bez gradientów
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= len(dataset):
                print(f"Indeks próbki {sample_idx} poza zakresem ({len(dataset)}). Pomijam.")
                continue

            features_sample, labels_gt_tuple = dataset[sample_idx]
            # Ustalenie ID utworu
            track_id_base = f"{track_id_prefix}_{sample_idx}"
            if hasattr(dataset, 'track_ids_base') and dataset.track_ids_base and sample_idx < len(
                    dataset.track_ids_base):
                track_id_base = dataset.track_ids_base[sample_idx]

            print(f"\nGenerowanie tabulatury tekstowej dla próbki {sample_idx} (ID: {track_id_base})...")

            onset_gt_sample, fret_gt_sample = labels_gt_tuple  # Tensory [liczba_ramek, liczba_strun]
            num_frames = onset_gt_sample.shape[0] # Liczba ramek z ground truth

            # --- Generowanie tabulatury Ground Truth ---
            gt_tab_matrix = _generate_tab_matrix_slots(
                onset_data=onset_gt_sample.cpu(), # Dane GT już powinny być na CPU z DataLoader
                fret_data=fret_gt_sample.cpu(),
                num_frames=num_frames,
                num_strings=num_strings,
                max_frets=max_frets,
                frames_per_slot=frames_per_slot,
                slot_char_width=slot_char_width
            )
            gt_tab_text_block = _format_tab_matrix_into_text_block(
                gt_tab_matrix, num_strings, line_break_after_slots
            )
            gt_output_filename = f"{track_id_base}_tablature_ground_truth.txt"
            gt_output_filepath = os.path.join(output_dir, gt_output_filename)
            try: # Zapis do pliku
                with open(gt_output_filepath, "w", encoding="utf-8") as f:
                    f.write(f"Track ID: {track_id_base}\n")
                    f.write("--- Ground Truth Tablature ---\n\n")
                    f.write(gt_tab_text_block)
                print(f"  Zapisano Ground Truth tabulaturę tekstową do: {gt_output_filepath}")
            except Exception as e:
                print(f"  Błąd podczas zapisywania pliku Ground Truth tabulatury {gt_output_filepath}: {e}")

            # --- Generowanie tabulatury z Predykcji Modelu ---
            features_sample_dev = features_sample.unsqueeze(0).to(device) # Przygotowanie danych dla modelu
            onset_pred_logits, fret_pred_logits = model_to_eval(features_sample_dev) # Predykcja
            # Przetworzenie wyjść modelu
            onset_pred_probs_sample = torch.sigmoid(onset_pred_logits.squeeze(0).cpu())
            onset_pred_binary_sample = (onset_pred_probs_sample > optimal_onset_threshold).float()
            fret_pred_indices_sample = torch.argmax(fret_pred_logits.squeeze(0).cpu(), dim=-1)

            pred_tab_matrix = _generate_tab_matrix_slots(
                onset_data=onset_pred_binary_sample,
                fret_data=fret_pred_indices_sample,
                num_frames=num_frames,  # Użyj tej samej liczby klatek co GT dla spójności długości tabulatur
                num_strings=num_strings,
                max_frets=max_frets,
                frames_per_slot=frames_per_slot,
                slot_char_width=slot_char_width
            )
            pred_tab_text_block = _format_tab_matrix_into_text_block(
                pred_tab_matrix, num_strings, line_break_after_slots
            )
            pred_output_filename = f"{track_id_base}_tablature_prediction_thresh{optimal_onset_threshold:.2f}.txt"
            pred_output_filepath = os.path.join(output_dir, pred_output_filename)
            try: # Zapis do pliku
                with open(pred_output_filepath, "w", encoding="utf-8") as f:
                    f.write(f"Track ID: {track_id_base}\n")
                    f.write(f"--- Predicted Tablature (Onset Threshold: {optimal_onset_threshold:.2f}) ---\n\n")
                    f.write(pred_tab_text_block)
                print(f"  Zapisano Predykowaną tabulaturę tekstową do: {pred_output_filepath}")
            except Exception as e:
                print(f"  Błąd podczas zapisywania pliku Predykowanej tabulatury {pred_output_filepath}: {e}")

    print("\nZakończono generowanie tabulatur tekstowych.")