import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import time
import plotting_utils

class CombinedLoss(nn.Module):
    """
    Połączona funkcja straty dla zadań detekcji onsetów i klasyfikacji progów.

    Używa BCEWithLogitsLoss dla onsetów oraz CrossEntropyLoss dla progów.
    Pozwala na ważenie poszczególnych komponentów straty oraz obsługę
    niezbalansowanych klas (dla onsetów) i ignorowanie indeksów (dla progów).
    """
    def __init__(
        self,
        onset_pos_weight=None,
        fret_ignore_index=None,
        onset_loss_weight=1.0,
        fret_loss_weight=1.0,
    ):
        """
        Inicjalizuje połączoną funkcję straty.

        Args:
            onset_pos_weight (torch.Tensor, optional): Waga dla pozytywnych przykładów w stracie onsetów,
                                                       przydatna przy niezbalansowanych danych. Domyślnie None.
            fret_ignore_index (int, optional): Indeks, który ma być ignorowany w obliczeniach straty dla progów
                                               przez CrossEntropyLoss. Domyślnie None (co oznacza -100 w PyTorch).
            onset_loss_weight (float, optional): Waga dla komponentu straty onsetów w całkowitej stracie.
                                                 Domyślnie 1.0.
            fret_loss_weight (float, optional): Waga dla komponentu straty progów w całkowitej stracie.
                                                Domyślnie 1.0.
        """
        super().__init__()
        self.onset_loss_weight = onset_loss_weight
        self.fret_loss_weight = fret_loss_weight
        # Funkcja straty dla binarnej klasyfikacji onsetów z logitami
        self.criterion_onset = nn.BCEWithLogitsLoss(pos_weight=onset_pos_weight)
        # Funkcja straty dla wieloklasowej klasyfikacji progów
        self.criterion_fret = nn.CrossEntropyLoss(
            ignore_index=fret_ignore_index if fret_ignore_index is not None else -100
        )

    def forward(self, onset_logits, fret_logits, onset_targets, fret_targets):
        """
        Oblicza całkowitą stratę oraz jej poszczególne komponenty (onset i fret).

        Args:
            onset_logits (torch.Tensor): Logity predykcji onsetów z modelu.
                                         Oczekiwany kształt: (batch_size, num_frames, num_strings).
            fret_logits (torch.Tensor): Logity predykcji progów z modelu.
                                        Oczekiwany kształt: (batch_size, num_frames, num_strings, num_fret_classes).
            onset_targets (torch.Tensor): Etykiety (0 lub 1) dla onsetów.
                                          Oczekiwany kształt: (batch_size, num_frames, num_strings).
            fret_targets (torch.Tensor): Etykiety (indeksy klas) dla progów.
                                         Oczekiwany kształt: (batch_size, num_frames, num_strings).

        Returns:
            tuple: Krotka (total_loss, loss_onset, loss_fret) zawierająca:
                   - total_loss (torch.Tensor): Ważona suma strat onsetów i progów.
                   - loss_onset (torch.Tensor): Strata dla detekcji onsetów.
                   - loss_fret (torch.Tensor): Strata dla klasyfikacji progów.
        """
        loss_o = self.criterion_onset(onset_logits, onset_targets)

        # CrossEntropyLoss w PyTorch oczekuje logitów w kształcie (N, C, d1, d2, ...),
        # gdzie C to liczba klas, a etykiet w kształcie (N, d1, d2, ...).
        # Aktualne fret_logits: (batch, frames, strings, classes)
        # Wymagane fret_logits: (batch, classes, frames, strings)
        # Aktualne fret_targets: (batch, frames, strings) - ten kształt jest poprawny dla permutacji logitów
        fret_logits_permuted = fret_logits.permute(0, 3, 1, 2)
        loss_f = self.criterion_fret(fret_logits_permuted, fret_targets)

        # Obliczenie całkowitej, ważonej straty
        total_loss = (self.onset_loss_weight * loss_o) + (
            self.fret_loss_weight * loss_f
        )
        return total_loss, loss_o, loss_f


def train_one_epoch(model, dataloader, optimizer, combined_criterion, device):
    """
    Przeprowadza trening modelu przez jedną epokę.

    Iteruje po danych z `dataloader`, wykonuje propagację w przód i wstecz,
    aktualizuje wagi modelu i agreguje wartości straty.

    Args:
        model (torch.nn.Module): Model PyTorch do trenowania.
        dataloader (torch.utils.data.DataLoader): DataLoader dostarczający dane treningowe.
        optimizer (torch.optim.Optimizer): Optymalizator używany do aktualizacji wag modelu.
        combined_criterion (CombinedLoss): Złożona funkcja straty (np. instancja `CombinedLoss`).
        device (torch.device): Urządzenie ('cuda' lub 'cpu'), na którym przeprowadzany jest trening.

    Returns:
        dict: Słownik zawierający średnie wartości strat dla epoki:
              'train_total_loss', 'train_onset_loss', 'train_fret_loss'.
    """
    model.train() # Ustawienie modelu w tryb treningu (aktywuje np. Dropout, BatchNorm w trybie treningowym)
    total_loss_epoch = 0.0
    onset_loss_epoch = 0.0
    fret_loss_epoch = 0.0

    for features, labels_tuple in dataloader:
        features = features.to(device)
        onset_targets, fret_targets = labels_tuple
        onset_targets = onset_targets.to(device)
        fret_targets = fret_targets.to(device)

        optimizer.zero_grad() # Wyzerowanie gradientów przed obliczeniem nowych

        onset_logits, fret_logits = model(features) # Propagacja w przód

        loss, loss_o, loss_f = combined_criterion( # Obliczenie straty
            onset_logits, fret_logits, onset_targets, fret_targets
        )

        loss.backward() # Propagacja wsteczna - obliczenie gradientów
        optimizer.step() # Aktualizacja wag modelu na podstawie gradientów

        total_loss_epoch += loss.item() # .item() wyciąga wartość skalarną z tensora straty
        onset_loss_epoch += loss_o.item()
        fret_loss_epoch += loss_f.item()

    datalen = len(dataloader) if len(dataloader) > 0 else 1 # Liczba batchy w epoce
    # Zwrócenie średnich wartości strat
    return {
        "train_total_loss": total_loss_epoch / datalen,
        "train_onset_loss": onset_loss_epoch / datalen,
        "train_fret_loss": fret_loss_epoch / datalen,
    }


def calculate_onset_metrics_for_threshold(probs, targets, threshold):
    """
    Oblicza podstawowe metryki klasyfikacji binarnej (precyzja, czułość, F1-score, dokładność)
    dla detekcji onsetów przy zadanym progu decyzyjnym.

    Args:
        probs (torch.Tensor): Tensor zawierający przewidziane prawdopodobieństwa onsetów (wartości [0, 1]).
        targets (torch.Tensor): Tensor zawierający binarne etykiety ground truth dla onsetów (0 lub 1).
        threshold (float): Próg używany do konwersji prawdopodobieństw na binarne predykcje (0 lub 1).

    Returns:
        tuple: Krotka (precision, recall, f1, accuracy) zawierająca obliczone metryki.
    """
    preds_binary = (probs > threshold).float() # Konwersja prawdopodobieństw na predykcje binarne
    # Obliczenie liczby True Positives, False Positives, False Negatives, True Negatives
    tp = ((preds_binary == 1) & (targets == 1)).sum().item()
    fp = ((preds_binary == 1) & (targets == 0)).sum().item()
    fn = ((preds_binary == 0) & (targets == 1)).sum().item()
    tn = ((preds_binary == 0) & (targets == 0)).sum().item()

    # Obliczenie metryk, z zabezpieczeniem przed dzieleniem przez zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return precision, recall, f1, accuracy


def find_optimal_onset_metrics(
    all_onset_probs_aggregated,
    all_onset_targets_aggregated,
    device,
    threshold_step=0.05,
):
    """
    Znajduje optymalny próg dla detekcji onsetów poprzez iteracyjne testowanie
    różnych wartości progów i wybór tej, która maksymalizuje F1-score.

    Args:
        all_onset_probs_aggregated (torch.Tensor): Zagregowany tensor wszystkich przewidzianych
                                                   prawdopodobieństw onsetów z epoki/zbioru danych.
        all_onset_targets_aggregated (torch.Tensor): Zagregowany tensor wszystkich etykiet ground truth
                                                     dla onsetów.
        device (torch.device): Urządzenie, na którym znajdują się tensory (choć obliczenia
                               są przeprowadzane na CPU po konwersji).
        threshold_step (float, optional): Krok, z jakim przeszukiwany jest zakres progów.
                                          Domyślnie 0.05.

    Returns:
        tuple: Krotka (optimal_threshold, best_f1, best_precision, best_recall, best_accuracy)
               zawierająca optymalny próg i metryki osiągnięte przy tym progu.
    """
    best_f1 = -1.0
    optimal_threshold = 0.5  # Wartość domyślna, jeśli żaden próg nie da F1 > 0
    best_precision = 0.0
    best_recall = 0.0
    best_accuracy = 0.0

    # Przeniesienie danych na CPU, jeśli jeszcze tam nie są, do obliczeń z NumPy/CPU tensorami
    all_onset_probs_cpu = all_onset_probs_aggregated.cpu()
    all_onset_targets_cpu = all_onset_targets_aggregated.cpu()

    # Generowanie zakresu progów do przetestowania
    thresholds = np.arange(0.05, 1.0, threshold_step)

    for current_threshold in thresholds:
        precision, recall, f1, accuracy = calculate_onset_metrics_for_threshold(
            all_onset_probs_cpu, all_onset_targets_cpu, current_threshold
        )

        if f1 > best_f1: # Aktualizacja, jeśli znaleziono lepszy F1-score
            best_f1 = f1
            optimal_threshold = current_threshold
            best_precision = precision
            best_recall = recall
            best_accuracy = accuracy

    return optimal_threshold, best_f1, best_precision, best_recall, best_accuracy


def evaluate_one_epoch(
    model,
    dataloader,
    combined_criterion,
    device,
    fret_num_classes,
    fixed_onset_threshold_for_logging=0.5,
):
    """
    Przeprowadza ewaluację modelu przez jedną epokę na danym zbiorze danych.

    Oblicza straty (jeśli `combined_criterion` jest podane) oraz metryki
    wydajności dla detekcji onsetów (przy stałym progu) i klasyfikacji progów.
    Zbiera również wszystkie predykcje prawdopodobieństw onsetów i odpowiadające
    im etykiety w celu późniejszego znalezienia optymalnego progu.

    Args:
        model (torch.nn.Module): Model PyTorch do ewaluacji.
        dataloader (torch.utils.data.DataLoader): DataLoader dla danych ewaluacyjnych.
        combined_criterion (CombinedLoss or None): Złożona funkcja straty. Jeśli None, straty nie są obliczane.
        device (torch.device): Urządzenie ('cuda' lub 'cpu'), na którym przeprowadzana jest ewaluacja.
        fret_num_classes (int): Całkowita liczba klas progów (w tym klasa "ciszy"),
                                używana do identyfikacji ramek aktywnych.
        fixed_onset_threshold_for_logging (float, optional): Stały próg używany do obliczenia
                                                            jednego zestawu metryk onsetów dla celów logowania.
                                                            Domyślnie 0.5.

    Returns:
        tuple: Krotka (metrics_at_fixed_threshold, all_onset_probs_aggregated, all_onset_targets_aggregated), gdzie:
               - metrics_at_fixed_threshold (dict): Słownik metryk (straty, precyzja, czułość, F1, dokładność)
                                                    obliczonych przy stałym progu onsetów.
               - all_onset_probs_aggregated (torch.Tensor): Zagregowany tensor wszystkich przewidzianych
                                                            prawdopodobieństw onsetów z całej epoki.
               - all_onset_targets_aggregated (torch.Tensor): Zagregowany tensor wszystkich etykiet ground truth
                                                              dla onsetów z całej epoki.
    """
    model.eval() # Ustawienie modelu w tryb ewaluacji (dezaktywuje np. Dropout)
    total_loss_epoch = 0.0
    onset_loss_epoch = 0.0
    fret_loss_epoch = 0.0

    # Inicjalizacja liczników dla metryk onsetów przy stałym progu
    onset_tp_fixed_total = 0
    onset_fp_fixed_total = 0
    onset_fn_fixed_total = 0
    onset_tn_fixed_total = 0

    # Inicjalizacja liczników dla metryk progów
    fret_correct_predictions_total = 0  # Całkowita liczba poprawnych predykcji progów
    fret_total_elements = 0             # Całkowita liczba elementów (ramek * strun) dla progów
    fret_correct_predictions_active = 0 # Poprawne predykcje progów tylko w "aktywnych" ramkach
    fret_total_active_frames_total = 0  # Całkowita liczba "aktywnych" ramek

    all_onset_probs_list_epoch = [] # Lista do zbierania prawdopodobieństw onsetów z batchy
    all_onset_targets_list_epoch = [] # Lista do zbierania etykiet onsetów z batchy

    with torch.no_grad(): # Wyłączenie obliczania gradientów podczas ewaluacji
        for features, labels_tuple in dataloader:
            features = features.to(device)
            onset_targets_batch, fret_targets_batch = labels_tuple
            onset_targets_batch = onset_targets_batch.to(device)
            fret_targets_batch = fret_targets_batch.to(device)

            onset_logits, fret_logits = model(features) # Predykcja modelu
            onset_probs_batch = torch.sigmoid(onset_logits) # Konwersja logitów onsetów na prawdopodobieństwa

            # Agregacja prawdopodobieństw i etykiet onsetów z całej epoki (na CPU, aby oszczędzać pamięć GPU)
            all_onset_probs_list_epoch.append(onset_probs_batch.reshape(-1).cpu())
            all_onset_targets_list_epoch.append(onset_targets_batch.reshape(-1).cpu())

            if combined_criterion: # Obliczenie strat, jeśli podano kryterium
                loss, loss_o, loss_f = combined_criterion(
                    onset_logits, fret_logits, onset_targets_batch, fret_targets_batch
                )
                total_loss_epoch += loss.item()
                onset_loss_epoch += loss_o.item()
                fret_loss_epoch += loss_f.item()

            # Obliczenie metryk onsetów przy stałym, predefiniowanym progu
            onset_preds_binary_fixed_batch = (onset_probs_batch > fixed_onset_threshold_for_logging).float()
            onset_tp_fixed_total += (((onset_preds_binary_fixed_batch == 1) & (onset_targets_batch == 1))).sum().item()
            onset_fp_fixed_total += (((onset_preds_binary_fixed_batch == 1) & (onset_targets_batch == 0))).sum().item()
            onset_fn_fixed_total += (((onset_preds_binary_fixed_batch == 0) & (onset_targets_batch == 1))).sum().item()
            onset_tn_fixed_total += (((onset_preds_binary_fixed_batch == 0) & (onset_targets_batch == 0))).sum().item()

            # Obliczenie metryk dla klasyfikacji progów
            fret_pred_indices_batch = torch.argmax(fret_logits, dim=-1) # Wybór klasy progu z najwyższym logitem
            fret_correct_predictions_total += (fret_pred_indices_batch == fret_targets_batch).sum().item()
            fret_total_elements += fret_targets_batch.numel()
            # Maska dla "aktywnych" ramek - tam, gdzie etykieta progu nie jest klasą "ciszy"
            mask_active_batch = fret_targets_batch != (fret_num_classes - 1)
            fret_correct_predictions_active += ((fret_pred_indices_batch == fret_targets_batch) & mask_active_batch).sum().item()
            fret_total_active_frames_total += mask_active_batch.sum().item()

    datalen = len(dataloader) if len(dataloader) > 0 else 1 # Liczba batchy

    # Przygotowanie słownika z metrykami (obliczonymi przy stałym progu onsetów)
    metrics_at_fixed_threshold = {
        "val_total_loss": total_loss_epoch / datalen,
        "val_onset_loss": onset_loss_epoch / datalen,
        "val_fret_loss": fret_loss_epoch / datalen,
        "onset_precision_at_fixed_thresh": (
            (onset_tp_fixed_total / (onset_tp_fixed_total + onset_fp_fixed_total))
            if (onset_tp_fixed_total + onset_fp_fixed_total) > 0 else 0.0
        ),
        "onset_recall_at_fixed_thresh": (
            (onset_tp_fixed_total / (onset_tp_fixed_total + onset_fn_fixed_total))
            if (onset_tp_fixed_total + onset_fn_fixed_total) > 0 else 0.0
        ),
        "fret_accuracy_overall": (
            (fret_correct_predictions_total / fret_total_elements)
            if fret_total_elements > 0 else 0.0
        ),
        "fret_accuracy_active": (
            (fret_correct_predictions_active / fret_total_active_frames_total)
            if fret_total_active_frames_total > 0 else 0.0
        ),
    }
    # Obliczenie F1-score i Accuracy dla onsetów przy stałym progu
    prec_fixed = metrics_at_fixed_threshold["onset_precision_at_fixed_thresh"]
    rec_fixed = metrics_at_fixed_threshold["onset_recall_at_fixed_thresh"]
    metrics_at_fixed_threshold["onset_f1_at_fixed_thresh"] = (
        (2 * prec_fixed * rec_fixed / (prec_fixed + rec_fixed))
        if (prec_fixed + rec_fixed) > 0 else 0.0
    )
    total_elements_fixed_accuracy = (onset_tp_fixed_total + onset_tn_fixed_total + onset_fp_fixed_total + onset_fn_fixed_total)
    metrics_at_fixed_threshold["onset_accuracy_at_fixed_thresh"] = (
        ((onset_tp_fixed_total + onset_tn_fixed_total) / total_elements_fixed_accuracy)
        if total_elements_fixed_accuracy > 0 else 0.0
    )

    # Konkatenacja list prawdopodobieństw i etykiet onsetów w pojedyncze tensory
    all_onset_probs_aggregated = torch.cat(all_onset_probs_list_epoch, dim=0)
    all_onset_targets_aggregated = torch.cat(all_onset_targets_list_epoch, dim=0)

    return (
        metrics_at_fixed_threshold,
        all_onset_probs_aggregated,
        all_onset_targets_aggregated,
    )


def run_training_loop(
    model,
    device,
    train_dataloader,
    validation_dataloader,
    optimizer,
    scheduler,
    combined_criterion,
    training_config_dict,
    sr,
    hop_length,
    validation_dataset_for_plotting=None,
):
    """
    Główna pętla zarządzająca procesem treningu i walidacji modelu przez wiele epok.

    Funkcja obsługuje:
    - Iterację przez zadaną liczbę epok.
    - Trening i walidację w każdej epoce.
    - Logowanie metryk do konsoli i pliku.
    - Zapisywanie historii metryk.
    - Stosowanie harmonogramu współczynnika uczenia (learning rate scheduler).
    - Zapisywanie najlepszej wersji modelu na podstawie wybranej metryki walidacyjnej.
    - Implementację mechanizmu wczesnego zatrzymania (early stopping).
    - Opcjonalne czyszczenie outputu konsoli.
    - Opcjonalne generowanie wizualizacji predykcji dla próbek walidacyjnych.

    Args:
        model (torch.nn.Module): Model PyTorch do trenowania.
        device (torch.device): Urządzenie ('cuda' lub 'cpu'), na którym odbywa się trening.
        train_dataloader (torch.utils.data.DataLoader): DataLoader dla danych treningowych.
        validation_dataloader (torch.utils.data.DataLoader): DataLoader dla danych walidacyjnych.
        optimizer (torch.optim.Optimizer): Optymalizator (np. Adam, SGD).
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Harmonogram współczynnika uczenia. Może być None.
        combined_criterion (CombinedLoss): Złożona funkcja straty.
        training_config_dict (dict): Słownik zawierający parametry konfiguracyjne pętli treningowej, np.:
            - "NUM_EPOCHS" (int): Całkowita liczba epok treningu.
            - "FRET_NUM_CLASSES" (int): Całkowita liczba klas progów (w tym "cisza").
            - "ARTIFACTS_DIR" (str): Katalog do zapisu artefaktów (logi, modele, wykresy).
            - "LOG_FILE_PATH" (str): Pełna ścieżka do pliku logu.
            - "CLEAR_CONSOLE_EVERY_N_EPOCHS" (int): Częstotliwość czyszczenia konsoli (0 = nigdy).
            - "CLEAR_OUTPUT_FUNC" (callable, optional): Funkcja do czyszczenia outputu (np. z IPython).
            - "PLOT_PREDICTIONS_EVERY_N_EPOCHS" (int): Częstotliwość generowania wykresów predykcji (0 = nigdy).
            - "ONSET_PREDICTION_THRESHOLD" (float): Stały próg dla onsetów używany w logach i wykresach.
            - "MAX_FRETS" (int): Maksymalny numer progu (używany w wizualizacjach).
            - "EARLY_STOPPING_PATIENCE" (int): Liczba epok bez poprawy, po której trening jest zatrzymywany.
            - "CHECKPOINT_METRIC" (str): Nazwa metryki walidacyjnej używanej do śledzenia postępów,
                                         zapisu najlepszego modelu i early stopping (np. 'val_onset_f1', 'val_total_loss').
        sr (int): Częstotliwość próbkowania audio (potrzebna do wizualizacji).
        hop_length (int): Długość przesunięcia okna (potrzebna do wizualizacji).
        validation_dataset_for_plotting (torch.utils.data.Dataset, optional): Instancja datasetu walidacyjnego
                                                                             używana do pobierania próbek
                                                                             do wizualizacji predykcji.

    Returns:
        dict: Słownik `history` zawierający zebrane metryki (straty, dokładności, itp.)
              z całego procesu treningu dla każdej epoki.
    """
    # Wczytanie parametrów konfiguracyjnych ze słownika
    num_epochs = training_config_dict.get("NUM_EPOCHS", 50)
    fret_num_classes = training_config_dict.get("FRET_NUM_CLASSES", 22)
    artifacts_dir = training_config_dict.get("ARTIFACTS_DIR", "training_artifacts")
    log_file_path = training_config_dict.get("LOG_FILE_PATH", os.path.join(artifacts_dir, "training_log.txt"))
    clear_output_every_n_epochs = training_config_dict.get("CLEAR_CONSOLE_EVERY_N_EPOCHS", 0)
    clear_output_func = training_config_dict.get("CLEAR_OUTPUT_FUNC", None)
    plot_predictions_every_n_epochs = training_config_dict.get("PLOT_PREDICTIONS_EVERY_N_EPOCHS", 10)
    fixed_onset_threshold_for_plotting_and_logging = training_config_dict.get("ONSET_PREDICTION_THRESHOLD", 0.5)
    max_frets = training_config_dict.get("MAX_FRETS", 20)
    early_stopping_patience = training_config_dict.get("EARLY_STOPPING_PATIENCE", 10)
    checkpoint_metric_name = training_config_dict.get("CHECKPOINT_METRIC", "val_onset_f1")

    # Inicjalizacja zmiennych do śledzenia najlepszej metryki i cierpliwości early stopping
    best_val_metric_value = -float("inf") # Dla metryk, które maksymalizujemy (np. F1, accuracy)
    if checkpoint_metric_name.endswith("loss"): # Dla metryk, które minimalizujemy (np. loss)
        best_val_metric_value = float("inf")
    epochs_no_improve = 0

    # Inicjalizacja słownika historii metryk
    history = {
        "train_total_loss": [], "train_onset_loss": [], "train_fret_loss": [],
        "val_total_loss": [], "val_onset_loss": [], "val_fret_loss": [],
        "onset_f1_at_fixed_thresh": [], "onset_precision_at_fixed_thresh": [],
        "onset_recall_at_fixed_thresh": [], "onset_accuracy_at_fixed_thresh": [],
        "val_onset_f1_optimal_thresh": [], "val_onset_precision_optimal_thresh": [],
        "val_onset_recall_optimal_thresh": [], "val_onset_accuracy_optimal_thresh": [],
        "val_optimal_onset_threshold_epoch": [],
        "fret_accuracy_overall": [], "fret_accuracy_active": [],
        "lr": [],
    }

    if not os.path.exists(artifacts_dir): # Utworzenie katalogu na artefakty, jeśli nie istnieje
        os.makedirs(artifacts_dir, exist_ok=True)

    with open(log_file_path, "a") as log_file: # Otwarcie pliku logu w trybie dopisywania
        log_file.write(f"\n--- Rozpoczęcie Nowej Sesji Treningowej: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.write("--- Konfiguracja Treningu ---\n")
        for key, value in training_config_dict.items():
            if callable(value) and key == "CLEAR_OUTPUT_FUNC":
                log_file.write(f"  {key}: Function provided\n")
            else:
                log_file.write(f"  {key}: {value}\n")
        log_file.write("-" * 30 + "\n\n")

        print(f"\nRozpoczynanie pętli treningowej na {num_epochs} epok...")
        for epoch in range(num_epochs): # Główna pętla po epokach
            epoch_desc = f"Epoka {epoch+1}/{num_epochs}"

            if (clear_output_func and clear_output_every_n_epochs > 0 and epoch > 0 and
                    (epoch) % clear_output_every_n_epochs == 0):
                clear_output_func(wait=True) # Opcjonalne czyszczenie konsoli
                print(f"Output wyczyszczony. Kontynuacja treningu (Epoka {epoch+1}/{num_epochs})...")

            # Faza treningu
            train_progress_bar = tqdm(train_dataloader, desc=f"{epoch_desc} [Trening]", unit="batch", leave=False, dynamic_ncols=True)
            train_metrics_epoch = train_one_epoch(model, train_progress_bar, optimizer, combined_criterion, device)
            train_progress_bar.close()
            history["train_total_loss"].append(train_metrics_epoch["train_total_loss"])
            history["train_onset_loss"].append(train_metrics_epoch["train_onset_loss"])
            history["train_fret_loss"].append(train_metrics_epoch["train_fret_loss"])

            # Faza walidacji
            val_progress_bar = tqdm(validation_dataloader, desc=f"{epoch_desc} [Walidacja]", unit="batch", leave=False, dynamic_ncols=True)
            val_metrics_fixed_thresh, all_onset_probs_val, all_onset_targets_val = evaluate_one_epoch(
                model, val_progress_bar, combined_criterion, device,
                fret_num_classes=fret_num_classes,
                fixed_onset_threshold_for_logging=fixed_onset_threshold_for_plotting_and_logging
            )
            val_progress_bar.close()
            history["val_total_loss"].append(val_metrics_fixed_thresh["val_total_loss"])
            history["val_onset_loss"].append(val_metrics_fixed_thresh["val_onset_loss"])
            history["val_fret_loss"].append(val_metrics_fixed_thresh["val_fret_loss"])
            history["onset_f1_at_fixed_thresh"].append(val_metrics_fixed_thresh["onset_f1_at_fixed_thresh"])
            history["onset_precision_at_fixed_thresh"].append(val_metrics_fixed_thresh["onset_precision_at_fixed_thresh"])
            history["onset_recall_at_fixed_thresh"].append(val_metrics_fixed_thresh["onset_recall_at_fixed_thresh"])
            history["onset_accuracy_at_fixed_thresh"].append(val_metrics_fixed_thresh["onset_accuracy_at_fixed_thresh"])
            history["fret_accuracy_overall"].append(val_metrics_fixed_thresh["fret_accuracy_overall"])
            history["fret_accuracy_active"].append(val_metrics_fixed_thresh["fret_accuracy_active"])

            # Znalezienie optymalnego progu onsetów dla danych walidacyjnych z bieżącej epoki
            (optimal_threshold_epoch, best_f1_epoch, p_at_best_f1, r_at_best_f1, acc_at_best_f1) = find_optimal_onset_metrics(
                all_onset_probs_val, all_onset_targets_val, device
            )
            history["val_onset_f1_optimal_thresh"].append(best_f1_epoch)
            history["val_onset_precision_optimal_thresh"].append(p_at_best_f1)
            history["val_onset_recall_optimal_thresh"].append(r_at_best_f1)
            history["val_onset_accuracy_optimal_thresh"].append(acc_at_best_f1)
            history["val_optimal_onset_threshold_epoch"].append(optimal_threshold_epoch)

            current_lr = optimizer.param_groups[0]["lr"] # Pobranie aktualnego współczynnika uczenia
            history["lr"].append(current_lr)

            # Logowanie wyników epoki
            log_entry_file_lines = [
                f"--- {epoch_desc} ---",
                f"  LR: {current_lr:.2e}",
                f"  Train Loss: {train_metrics_epoch['train_total_loss']:.4f} (Onset: {train_metrics_epoch['train_onset_loss']:.4f}, Fret: {train_metrics_epoch['train_fret_loss']:.4f})",
                f"  Val   Loss: {val_metrics_fixed_thresh.get('val_total_loss',0.0):.4f} (Onset: {val_metrics_fixed_thresh.get('val_onset_loss',0.0):.4f}, Fret: {val_metrics_fixed_thresh.get('val_fret_loss',0.0):.4f})",
                f"  Val Ons (Fixed Th={fixed_onset_threshold_for_plotting_and_logging:.2f}): F1: {val_metrics_fixed_thresh.get('onset_f1_at_fixed_thresh', 0.0):.4f} (P: {val_metrics_fixed_thresh.get('onset_precision_at_fixed_thresh',0.0):.4f} R: {val_metrics_fixed_thresh.get('onset_recall_at_fixed_thresh',0.0):.4f} Acc: {val_metrics_fixed_thresh.get('onset_accuracy_at_fixed_thresh',0.0):.4f})",
                f"  Val Ons (Opt Th={optimal_threshold_epoch:.2f}): F1: {best_f1_epoch:.4f} (P: {p_at_best_f1:.4f} R: {r_at_best_f1:.4f} Acc: {acc_at_best_f1:.4f})",
                f"  Val Fret Acc (Overall): {val_metrics_fixed_thresh.get('fret_accuracy_overall', 0.0):.4f} | Val Fret Acc (Active): {val_metrics_fixed_thresh.get('fret_accuracy_active', 0.0):.4f}",
            ]
            log_file.write("\n".join(log_entry_file_lines) + "\n"); log_file.flush()

            print(f"\n--- {epoch_desc} ---")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Train Loss - Total: {train_metrics_epoch['train_total_loss']:.4f}, Onset: {train_metrics_epoch['train_onset_loss']:.4f}, Fret: {train_metrics_epoch['train_fret_loss']:.4f}")
            print(f"  Val   Loss - Total: {val_metrics_fixed_thresh.get('val_total_loss',0.0):.4f}, Onset: {val_metrics_fixed_thresh.get('val_onset_loss',0.0):.4f}, Fret: {val_metrics_fixed_thresh.get('val_fret_loss',0.0):.4f}")
            print(f"  Val Onsets (Fixed Th={fixed_onset_threshold_for_plotting_and_logging:.2f}) - F1: {val_metrics_fixed_thresh.get('onset_f1_at_fixed_thresh', 0.0):.4f}, P: {val_metrics_fixed_thresh.get('onset_precision_at_fixed_thresh',0.0):.4f}, R: {val_metrics_fixed_thresh.get('onset_recall_at_fixed_thresh',0.0):.4f}, Acc: {val_metrics_fixed_thresh.get('onset_accuracy_at_fixed_thresh',0.0):.4f}")
            print(f"  Val Onsets (Opt Th={optimal_threshold_epoch:.2f}) - F1: {best_f1_epoch:.4f}, P: {p_at_best_f1:.4f}, R: {r_at_best_f1:.4f}, Acc: {acc_at_best_f1:.4f}")
            print(f"  Val Frets  - Acc Overall: {val_metrics_fixed_thresh.get('fret_accuracy_overall', 0.0):.4f}, Acc Active: {val_metrics_fixed_thresh.get('fret_accuracy_active', 0.0):.4f}")

            # Wybór metryki do śledzenia dla schedulera i zapisu modelu
            metric_for_scheduler_and_checkpoint = -float('inf') # Domyślna wartość dla metryk maksymalizowanych
            if checkpoint_metric_name == "val_onset_f1":
                metric_for_scheduler_and_checkpoint = best_f1_epoch
            elif checkpoint_metric_name == "val_onset_f1_fixed":
                metric_for_scheduler_and_checkpoint = val_metrics_fixed_thresh.get('onset_f1_at_fixed_thresh', -float('inf'))
            elif checkpoint_metric_name == "val_total_loss":
                metric_for_scheduler_and_checkpoint = val_metrics_fixed_thresh.get('val_total_loss', float('inf'))
            # Można dodać inne opcje metryk

            if scheduler: # Krok harmonogramu współczynnika uczenia
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metric_for_scheduler_and_checkpoint)
                else:
                    scheduler.step() # Dla schedulerów nie wymagających metryki (np. StepLR)

            # Logika zapisu najlepszego modelu i early stopping
            improved = False
            if checkpoint_metric_name.endswith("loss"): # Jeśli śledzimy stratę (minimalizujemy)
                if metric_for_scheduler_and_checkpoint < best_val_metric_value:
                    best_val_metric_value = metric_for_scheduler_and_checkpoint
                    improved = True
            else: # Jeśli śledzimy metrykę typu F1, accuracy (maksymalizujemy)
                if metric_for_scheduler_and_checkpoint > best_val_metric_value:
                    best_val_metric_value = metric_for_scheduler_and_checkpoint
                    improved = True

            if improved: # Jeśli nastąpiła poprawa śledzonej metryki
                epochs_no_improve = 0 # Reset licznika braku poprawy
                model_save_path = os.path.join(artifacts_dir, "best_model.pth")
                torch.save(model.state_dict(), model_save_path) # Zapis stanu modelu
                improvement_log_base = f"    -> Zapisano nowy najlepszy model ({checkpoint_metric_name}: {best_val_metric_value:.4f}"
                if checkpoint_metric_name == "val_onset_f1":
                     improvement_log = f"{improvement_log_base} przy optymalnym progu {optimal_threshold_epoch:.2f})"
                else:
                    improvement_log = f"{improvement_log_base})"
                print(improvement_log); log_file.write(improvement_log + "\n")
            elif early_stopping_patience is not None and epoch > 0: # Jeśli nie ma poprawy i early stopping jest aktywny
                epochs_no_improve += 1
                no_improvement_log_base = f"({metric_for_scheduler_and_checkpoint:.4f} vs Best: {best_val_metric_value:.4f})"
                if checkpoint_metric_name == "val_onset_f1":
                    no_improvement_log = f"    Brak poprawy {checkpoint_metric_name} (przy opt. progu {optimal_threshold_epoch:.2f}) od {epochs_no_improve} epok {no_improvement_log_base}"
                else:
                    no_improvement_log = f"    Brak poprawy {checkpoint_metric_name} od {epochs_no_improve} epok {no_improvement_log_base}"

                if epochs_no_improve == 1 or epochs_no_improve % max(1, early_stopping_patience // 3) == 0:
                    print(no_improvement_log) # Loguj rzadziej, jeśli brak poprawy trwa
                log_file.write(no_improvement_log + "\n")

                if epochs_no_improve >= early_stopping_patience: # Sprawdzenie warunku early stopping
                    early_stop_log = f"    Wczesne zatrzymanie treningu po {epochs_no_improve} epokach bez poprawy dla metryki '{checkpoint_metric_name}'."
                    print(early_stop_log); log_file.write(early_stop_log + "\n\n")
                    # Opcjonalna wizualizacja przy early stopping
                    if validation_dataset_for_plotting and len(validation_dataset_for_plotting) > 0 and plot_predictions_every_n_epochs > 0:
                        vis_log_entry = f"  Generowanie wizualizacji predykcji dla epoki {epoch+1} (early stop)..."
                        print(vis_log_entry); log_file.write(vis_log_entry + "\n")
                        try:
                            idx_to_plot = min(0, len(validation_dataset_for_plotting)-1)
                            track_id_for_plot = getattr(validation_dataset_for_plotting, 'track_ids_base', [f"sample_{idx_to_plot}"])[idx_to_plot]
                            features_sample, labels_gt_tuple = validation_dataset_for_plotting[idx_to_plot]
                            features_sample_dev = features_sample.unsqueeze(0).to(device)
                            model.eval() # Upewnij się, że model jest w trybie eval
                            with torch.no_grad(): onset_pred_logits, fret_pred_logits = model(features_sample_dev)
                            plot_save_path_es = os.path.join(artifacts_dir, f"predictions_epoch_{epoch+1}_earlystop.png")
                            plotting_utils.plot_predictions_vs_ground_truth(
                                features_sample=features_sample.cpu(), onset_gt_sample=labels_gt_tuple[0].cpu(), fret_gt_sample=labels_gt_tuple[1].cpu(),
                                onset_pred_logits_sample=onset_pred_logits.squeeze(0).cpu(), fret_pred_logits_sample=fret_pred_logits.squeeze(0).cpu(),
                                sr=sr, hop_length=hop_length, max_frets=max_frets, onset_threshold=optimal_threshold_epoch,
                                track_id_base=track_id_for_plot, save_path=plot_save_path_es
                            )
                            log_file.write(f"  Zapisano wykres predykcji (early stop) do: {plot_save_path_es}\n")
                        except Exception as e_vis: print(f"    Błąd wizualizacji (early stop): {e_vis}")
                    break # Przerwanie głównej pętli treningowej

            # Okresowe generowanie wizualizacji predykcji
            plot_now_condition = (plot_predictions_every_n_epochs > 0 and (epoch + 1) % plot_predictions_every_n_epochs == 0) or (epoch == num_epochs - 1)
            if plot_now_condition and not (early_stopping_patience and epochs_no_improve >= early_stopping_patience): # Nie plotuj, jeśli właśnie było early stopping
                if validation_dataset_for_plotting and len(validation_dataset_for_plotting) > 0:
                    vis_log_entry = f"  Generowanie wizualizacji predykcji dla epoki {epoch+1}..."
                    print(vis_log_entry); log_file.write(vis_log_entry + "\n")
                    try:
                        idx_to_plot = min(0, len(validation_dataset_for_plotting)-1)
                        track_id_for_plot = getattr(validation_dataset_for_plotting, 'track_ids_base', [f"sample_{idx_to_plot}"])[idx_to_plot]
                        features_sample, labels_gt_tuple = validation_dataset_for_plotting[idx_to_plot]
                        features_sample_dev = features_sample.unsqueeze(0).to(device)
                        model.eval() # Upewnij się, że model jest w trybie eval
                        with torch.no_grad(): onset_pred_logits, fret_pred_logits = model(features_sample_dev)
                        plot_save_path_epoch = os.path.join(artifacts_dir, f"predictions_epoch_{epoch+1}.png")
                        plotting_utils.plot_predictions_vs_ground_truth(
                            features_sample=features_sample.cpu(), onset_gt_sample=labels_gt_tuple[0].cpu(), fret_gt_sample=labels_gt_tuple[1].cpu(),
                            onset_pred_logits_sample=onset_pred_logits.squeeze(0).cpu(), fret_pred_logits_sample=fret_pred_logits.squeeze(0).cpu(),
                            sr=sr, hop_length=hop_length, max_frets=max_frets, onset_threshold=optimal_threshold_epoch, # Użyj optymalnego progu z tej epoki
                            track_id_base=track_id_for_plot, save_path=plot_save_path_epoch
                        )
                        log_file.write(f"  Zapisano wykres predykcji do: {plot_save_path_epoch}\n")
                    except Exception as e_vis:
                        err_log_vis_epoch = f"    Błąd podczas generowania wizualizacji predykcji (epoch): {e_vis}"
                        print(err_log_vis_epoch); log_file.write(err_log_vis_epoch + "\n")
            log_file.write("-" * 80 + "\n") # Separator w logu po każdej epoce

        # Koniec pętli treningowej
        log_file.write(f"\n--- Koniec Sesji Treningowej: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
    print("\nZakończono pętlę treningową.")
    return history