# plotting_utils.py
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torch


def plot_spectrogram(
    spectrogram_tensor, sr, hop_length, ax=None, title="Mel-Spektrogram"
):
    """
    Rysuje mel-spektrogram.

    Args:
        spectrogram_tensor (torch.Tensor or np.ndarray): Tensor lub tablica NumPy zawierająca spektrogram.
        sr (int): Częstotliwość próbkowania audio.
        hop_length (int): Długość przesunięcia okna (hop size) w próbkach.
        ax (matplotlib.axes.Axes, optional): Oś, na której ma być narysowany wykres.
                                             Jeśli None, tworzona jest nowa figura i oś. Domyślnie None.
        title (str, optional): Tytuł wykresu. Domyślnie "Mel-Spektrogram".

    Returns:
        tuple: Krotka (fig, ax) zawierająca obiekt figury i osi Matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure # Pobranie figury z istniejącej osi

    # Konwersja tensora PyTorch do tablicy NumPy, jeśli to konieczne
    if isinstance(spectrogram_tensor, torch.Tensor):
        spec_np = spectrogram_tensor.cpu().numpy()
    else:
        spec_np = spectrogram_tensor

    # Wyświetlenie spektrogramu
    img = librosa.display.specshow(
        spec_np,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig, ax


def plot_onset_labels(
    onset_targets_tensor,
    sr,
    hop_length,
    num_strings=6,
    ax=None,
    title="Etykiety Onsetów",
):
    """
    Rysuje binarne etykiety onsetów w stylu piano roll.

    Args:
        onset_targets_tensor (torch.Tensor or np.ndarray): Tensor lub tablica NumPy z etykietami onsetów (ramki, struny).
        sr (int): Częstotliwość próbkowania audio.
        hop_length (int): Długość przesunięcia okna w próbkach.
        num_strings (int, optional): Liczba strun. Domyślnie 6.
        ax (matplotlib.axes.Axes, optional): Oś do rysowania. Jeśli None, tworzona jest nowa. Domyślnie None.
        title (str, optional): Tytuł wykresu. Domyślnie "Etykiety Onsetów".

    Returns:
        tuple: Krotka (fig, ax) zawierająca obiekt figury i osi Matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    if isinstance(onset_targets_tensor, torch.Tensor):
        onsets_np = onset_targets_tensor.cpu().numpy()
    else:
        onsets_np = onset_targets_tensor

    # Transpozycja, aby struny były na osi Y, a czas na osi X
    onsets_to_plot = onsets_np.T
    img = librosa.display.specshow(
        onsets_to_plot,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        ax=ax,
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(np.arange(num_strings)) # Ustawienie znaczników osi Y dla strun
    ax.set_yticklabels([f"Str {i}" for i in range(num_strings)])
    ax.set_ylabel("Struna")
    ax.set_title(title)
    return fig, ax


def plot_fret_labels(
    fret_targets_tensor,
    sr,
    hop_length,
    num_strings=6,
    max_fret_display=21, # Maksymalny próg do wyświetlenia na skali kolorów (+1 dla ciszy)
    ax=None,
    title="Etykiety Progów",
):
    """
    Rysuje etykiety progów w stylu piano roll z kolorami reprezentującymi numery progów.

    Args:
        fret_targets_tensor (torch.Tensor or np.ndarray): Tensor lub tablica NumPy z etykietami progów (ramki, struny).
        sr (int): Częstotliwość próbkowania audio.
        hop_length (int): Długość przesunięcia okna w próbkach.
        num_strings (int, optional): Liczba strun. Domyślnie 6.
        max_fret_display (int, optional): Maksymalny numer progu uwzględniany na skali kolorów.
                                          Wartości większe będą wyświetlane jako "cisza". Domyślnie 21.
        ax (matplotlib.axes.Axes, optional): Oś do rysowania. Jeśli None, tworzona jest nowa. Domyślnie None.
        title (str, optional): Tytuł wykresu. Domyślnie "Etykiety Progów".

    Returns:
        tuple: Krotka (fig, ax) zawierająca obiekt figury i osi Matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    if isinstance(fret_targets_tensor, torch.Tensor):
        frets_np = fret_targets_tensor.cpu().numpy()
    else:
        frets_np = fret_targets_tensor

    frets_to_plot = frets_np.T # Transpozycja dla osi czasu i strun
    # Utworzenie dyskretnej mapy kolorów dla numerów progów
    cmap = plt.cm.get_cmap("viridis", max_fret_display + 1) # +1 dla progu "cisza"

    # Użycie pcolormesh do rysowania wartości progów
    img = ax.pcolormesh(
        librosa.frames_to_time( # Konwersja indeksów ramek na czas
            np.arange(frets_to_plot.shape[1] + 1), sr=sr, hop_length=hop_length
        ),
        np.arange(num_strings + 1), # Granice dla strun
        frets_to_plot,
        cmap=cmap,
        vmin=0,
        vmax=max_fret_display, # Ustawienie zakresu dla mapy kolorów
    )
    ax.set_yticks(np.arange(num_strings) + 0.5)
    ax.set_yticklabels([f"Str {i}" for i in range(num_strings)])
    ax.set_ylabel("Struna")
    ax.set_xlabel("Czas (s)")
    ax.set_title(title)
    # Dodanie paska kolorów
    cbar = fig.colorbar(
        img,
        ax=ax,
        ticks=np.arange(0, max_fret_display + 1, max(1, max_fret_display // 10)), # Znaczniki na pasku
    )
    cbar.set_label("Numer progu ('brak dźwięku' >= " + str(max_fret_display) + ")")
    return fig, ax


def plot_batch_sample(
    features,
    labels_tuple,
    sample_idx,
    sr,
    hop_length,
    max_frets_for_display,
    num_strings=6,
    figsize=(12, 10),
):
    """
    Rysuje wizualizację pojedynczej próbki z paczki danych (spektrogram, etykiety onsetów GT, etykiety progów GT).

    Args:
        features (torch.Tensor): Tensor cech (spektrogramów) o kształcie (batch_size, num_mel_bins, num_frames).
        labels_tuple (tuple): Krotka zawierająca tensory etykiet: (onset_labels, fret_labels).
                              onset_labels: (batch_size, num_frames, num_strings)
                              fret_labels: (batch_size, num_frames, num_strings)
        sample_idx (int): Indeks próbki z paczki do zwizualizowania.
        sr (int): Częstotliwość próbkowania.
        hop_length (int): Długość przesunięcia okna.
        max_frets_for_display (int): Maksymalny próg do wyświetlenia na wykresie progów.
        num_strings (int, optional): Liczba strun. Domyślnie 6.
        figsize (tuple, optional): Rozmiar figury. Domyślnie (12, 10).

    Returns:
        None: Funkcja wyświetla wykres i go zamyka.
    """
    # Sprawdzenie poprawności indeksu próbki
    if not (0 <= sample_idx < features.shape[0]):
        print(
            f"Błąd: sample_idx ({sample_idx}) jest poza zakresem paczki (0-{features.shape[0]-1})."
        )
        return

    # Wybranie danych dla konkretnej próbki
    spec_sample = features[sample_idx]
    onset_labels_sample = labels_tuple[0][sample_idx]
    fret_labels_sample = labels_tuple[1][sample_idx]

    # Utworzenie subwykresów (3 wiersze, 1 kolumna)
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True) # Współdzielona oś X (czas)
    if fig is None or axs is None: # Podstawowe sprawdzenie
        print("Błąd: Nie udało się utworzyć figury lub osi dla wykresu.")
        return

    fig.suptitle(
        f"Wizualizacja próbki {sample_idx} z paczki (Ground Truth)", fontsize=16
    )

    # Rysowanie spektrogramu
    plot_spectrogram(spec_sample, sr=sr, hop_length=hop_length, ax=axs[0])
    # Rysowanie etykiet onsetów
    plot_onset_labels(
        onset_labels_sample,
        sr=sr,
        hop_length=hop_length,
        num_strings=num_strings,
        ax=axs[1],
        title="GT Onsety",
    )
    # Rysowanie etykiet progów
    plot_fret_labels(
        fret_labels_sample,
        sr=sr,
        hop_length=hop_length,
        num_strings=num_strings,
        max_fret_display=max_frets_for_display,
        ax=axs[2],
        title="GT Progi",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Dopasowanie układu, zostawiając miejsce na suptitle
    plt.show() # Wyświetlenie wykresu
    plt.close(fig)  # Zamknięcie figury po pokazaniu/zapisaniu, aby zwolnić pamięć


def print_batch_sample_details(
    features, labels_tuple, sample_idx, max_frames_to_show=5, max_strings_to_show=6
):
    """
    Wyświetla szczegółowe informacje tekstowe o pojedynczej próbce z paczki danych.

    Args:
        features (torch.Tensor): Tensor cech (spektrogramów).
        labels_tuple (tuple): Krotka tensorów etykiet (onset_labels, fret_labels).
        sample_idx (int): Indeks próbki z paczki.
        max_frames_to_show (int, optional): Maksymalna liczba ramek do wyświetlenia w fragmentach danych. Domyślnie 5.
        max_strings_to_show (int, optional): Maksymalna liczba strun do wyświetlenia w fragmentach danych. Domyślnie 6.

    Returns:
        None: Funkcja drukuje informacje na konsolę.
    """
    if not (0 <= sample_idx < features.shape[0]):
        print(
            f"Błąd: sample_idx ({sample_idx}) jest poza zakresem paczki (0-{features.shape[0]-1})."
        )
        return

    print(
        f"\n--- Szczegóły tekstowe dla próbki {sample_idx} z paczki (Ground Truth) ---"
    )
    spec_sample = features[sample_idx]
    n_mels, num_frames = spec_sample.shape
    print(f"\n1. Spektrogram (Log-Mel):")
    print(
        f"   Kształt: {spec_sample.shape}, Typ: {spec_sample.dtype}, Min: {spec_sample.min():.2f}, Max: {spec_sample.max():.2f}, Śr: {spec_sample.mean():.2f}"
    )
    # Wyświetlenie fragmentu spektrogramu
    if num_frames > 0 and n_mels > 0:
        print(
            f"   Fragment: {spec_sample[:min(max_frames_to_show, n_mels), :min(max_frames_to_show, num_frames)].cpu().numpy()}"
        )

    onset_labels_sample = labels_tuple[0][sample_idx]
    num_label_frames, num_label_strings = onset_labels_sample.shape
    actual_strings_to_show = min(max_strings_to_show, num_label_strings)
    actual_frames_to_show = min(max_frames_to_show, num_label_frames)
    print(f"\n2. Etykiety Onsetów (GT):")
    print(
        f"   Kształt: {onset_labels_sample.shape}, Typ: {onset_labels_sample.dtype}, Suma (aktywnych onsetów): {onset_labels_sample.sum():.0f}"
    )
    # Wyświetlenie fragmentu etykiet onsetów
    if num_label_frames > 0 and num_label_strings > 0:
        print(
            f"   Początek (transponowany): {onset_labels_sample[:actual_frames_to_show, :actual_strings_to_show].cpu().numpy().T}"
        )
        if num_label_frames > max_frames_to_show: # Jeśli jest więcej ramek, pokaż też koniec
            print(
                f"   Koniec (transponowany): {onset_labels_sample[-actual_frames_to_show:, :actual_strings_to_show].cpu().numpy().T}"
            )

    fret_labels_sample = labels_tuple[1][sample_idx]
    print(f"\n3. Etykiety Progów (GT - indeksy):")
    print(
        f"   Kształt: {fret_labels_sample.shape}, Typ: {fret_labels_sample.dtype}, Unikalne wartości: {torch.unique(fret_labels_sample).cpu().numpy()}"
    )
    # Wyświetlenie fragmentu etykiet progów
    if num_label_frames > 0 and num_label_strings > 0:
        print(
            f"   Początek (transponowany): {fret_labels_sample[:actual_frames_to_show, :actual_strings_to_show].cpu().numpy().T}"
        )
        if num_label_frames > max_frames_to_show:
            print(
                f"   Koniec (transponowany): {fret_labels_sample[-actual_frames_to_show:, :actual_strings_to_show].cpu().numpy().T}"
            )
    print("--- Koniec szczegółów tekstowych (GT) ---")


def plot_predictions_vs_ground_truth(
    features_sample,
    onset_gt_sample,
    fret_gt_sample,
    onset_pred_logits_sample,
    fret_pred_logits_sample,
    sr,
    hop_length,
    max_frets,
    onset_threshold=0.5,
    num_strings=6,
    figsize=(15, 18),
    track_id_base=None,
    save_path=None,
):
    """
    Rysuje kompleksowe porównanie ground truth z predykcjami modelu dla pojedynczej próbki.
    Zawiera spektrogram, GT onsety, predykowane onsety, GT progi, predykowane progi.

    Args:
        features_sample (torch.Tensor): Tensor spektrogramu (num_mel_bins, num_frames).
        onset_gt_sample (torch.Tensor): Tensor GT onsetów (num_frames, num_strings).
        fret_gt_sample (torch.Tensor): Tensor GT progów (num_frames, num_strings).
        onset_pred_logits_sample (torch.Tensor): Tensor predykowanych logitów onsetów (num_frames, num_strings).
        fret_pred_logits_sample (torch.Tensor): Tensor predykowanych logitów progów (num_frames, num_strings, num_fret_classes).
        sr (int): Częstotliwość próbkowania.
        hop_length (int): Długość przesunięcia okna.
        max_frets (int): Maksymalny numer progu (np. 20). `max_fret_display` będzie `max_frets + 1`.
        onset_threshold (float, optional): Próg do binaryzacji predykcji onsetów. Domyślnie 0.5.
        num_strings (int, optional): Liczba strun. Domyślnie 6.
        figsize (tuple, optional): Rozmiar figury. Domyślnie (15, 18).
        track_id_base (str, optional): ID utworu, dodawane do tytułu. Domyślnie None.
        save_path (str, optional): Ścieżka do zapisu wykresu. Jeśli None, wykres nie jest zapisywany. Domyślnie None.

    Returns:
        None: Funkcja zapisuje lub wyświetla wykres i go zamyka.
    """
    # Przetwarzanie predykcji modelu
    onset_pred_probs_sample = torch.sigmoid(onset_pred_logits_sample) # Konwersja logitów na prawdopodobieństwa
    onset_pred_binary_sample = (onset_pred_probs_sample > onset_threshold).float() # Binaryzacja
    fret_pred_indices_sample = torch.argmax(fret_pred_logits_sample, dim=-1) # Wybór progów z największym prawdopodobieństwem

    # Utworzenie subwykresów (5 wierszy)
    fig, axs = plt.subplots(5, 1, figsize=figsize, sharex=True)
    main_title = "Ground Truth vs Predykcje"
    if track_id_base:
        main_title = f"{main_title} (Utwór: {track_id_base})"
    fig.suptitle(main_title, fontsize=16)

    # Maksymalny próg do wyświetlenia na wykresie (uwzględniając klasę "cisza")
    max_fret_display = max_frets + 1 # Klasa `max_frets+1` to często "brak dźwięku"

    # 1. Spektrogram
    plot_spectrogram(
        features_sample,
        sr=sr,
        hop_length=hop_length,
        ax=axs[0],
        title="Mel-Spektrogram",
    )
    # 2. Ground Truth Onsety
    plot_onset_labels(
        onset_gt_sample,
        sr=sr,
        hop_length=hop_length,
        num_strings=num_strings,
        ax=axs[1],
        title="Ground Truth Onsety",
    )
    # 3. Predykowane Onsety
    plot_onset_labels(
        onset_pred_binary_sample,
        sr=sr,
        hop_length=hop_length,
        num_strings=num_strings,
        ax=axs[2],
        title=f"Predykowane Onsety (próg={onset_threshold:.2f})",
    )
    # 4. Ground Truth Progi
    plot_fret_labels(
        fret_gt_sample,
        sr=sr,
        hop_length=hop_length,
        num_strings=num_strings,
        max_fret_display=max_fret_display,
        ax=axs[3],
        title="Ground Truth Progi",
    )
    # 5. Predykowane Progi
    plot_fret_labels(
        fret_pred_indices_sample,
        sr=sr,
        hop_length=hop_length,
        num_strings=num_strings,
        max_fret_display=max_fret_display,
        ax=axs[4],
        title="Predykowane Progi",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Zapis wykresu, jeśli podano ścieżkę
    if save_path:
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"  Nie udało się zapisać wykresu do {save_path}: {e}")
    plt.close(fig)


def plot_training_history(
    history, save_path=None, figsize=(20, 22)
):
    """
    Rysuje wykresy historii treningu modelu na podstawie zebranych metryk.

    Args:
        history (dict): Słownik zawierający listy metryk dla każdej epoki.
                        Oczekiwane klucze to np. 'train_total_loss', 'val_total_loss',
                        'val_onset_f1_optimal_thresh', 'lr', itp.
        save_path (str, optional): Ścieżka do zapisu wykresu. Jeśli None, wykres nie jest zapisywany. Domyślnie None.
        figsize (tuple, optional): Rozmiar figury. Domyślnie (20, 22).

    Returns:
        None: Funkcja zapisuje lub wyświetla wykres i go zamyka.
    """
    # Sprawdzenie, czy historia zawiera dane
    if not history or not history.get("train_total_loss"):
        print("Historia treningu jest pusta lub niekompletna.")
        return

    epochs_ran = len(history["train_total_loss"])
    if epochs_ran == 0:
        print("Brak danych w historii treningu do narysowania.")
        return

    epoch_range = range(1, epochs_ran + 1) # Zakres epok dla osi X

    # Utworzenie figury z subwykresami (4 wiersze, 2 kolumny)
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle("Historia Treningu Modelu", fontsize=18)

    # --- Wykres 1: Strata Całkowita (Treningowa i Walidacyjna) ---
    ax = axs[0, 0]
    plot_lines = [] # Do sprawdzenia czy coś zostało narysowane przed dodaniem legendy
    if "train_total_loss" in history and history["train_total_loss"]:
        plot_lines.append(
            ax.plot(
                epoch_range, history["train_total_loss"], "o-", label="Train Total Loss"
            )
        )
    if "val_total_loss" in history and history["val_total_loss"]:
        plot_lines.append(
            ax.plot(
                epoch_range, history["val_total_loss"], "o-", label="Val Total Loss"
            )
        )
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Strata")
    ax.set_title("Strata Całkowita")
    ax.grid(True)
    if plot_lines: ax.legend()

    # --- Wykres 2: Metryki Walidacyjne Onsetów (przy optymalnym progu) ---
    ax = axs[0, 1]
    plot_lines = []
    if "val_onset_f1_optimal_thresh" in history and history["val_onset_f1_optimal_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["val_onset_f1_optimal_thresh"], "o-", label="Val Onset F1 (Opt Th)"))
    if "val_onset_precision_optimal_thresh" in history and history["val_onset_precision_optimal_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["val_onset_precision_optimal_thresh"], "o-", label="Val Onset Precision (Opt Th)"))
    if "val_onset_recall_optimal_thresh" in history and history["val_onset_recall_optimal_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["val_onset_recall_optimal_thresh"], "o-", label="Val Onset Recall (Opt Th)"))
    if "val_onset_accuracy_optimal_thresh" in history and history["val_onset_accuracy_optimal_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["val_onset_accuracy_optimal_thresh"], "o--", label="Val Onset Acc (Opt Th)"))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Metryka")
    ax.set_title("Metryki Walidacyjne Onsetów (Przy Optymalnym Progu)")
    ax.grid(True)
    ax.set_ylim(0, 1.05) # Ograniczenie osi Y dla metryk (0-1)
    if plot_lines: ax.legend()

    # --- Wykres 3: Strata Onsetów (Treningowa i Walidacyjna) ---
    ax = axs[1, 0]
    plot_lines = []
    if "train_onset_loss" in history and history["train_onset_loss"]:
        plot_lines.append(ax.plot(epoch_range, history["train_onset_loss"], "o-", label="Train Onset Loss"))
    if "val_onset_loss" in history and history["val_onset_loss"]:
        plot_lines.append(ax.plot(epoch_range, history["val_onset_loss"], "o-", label="Val Onset Loss"))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Strata")
    ax.set_title("Strata Onsetów")
    ax.grid(True)
    if plot_lines: ax.legend()

    # --- Wykres 4: Metryki Walidacyjne Progów (Accuracy Overall, Accuracy Active) ---
    ax = axs[1, 1]
    plot_lines = []
    if "fret_accuracy_overall" in history and history["fret_accuracy_overall"]:
        plot_lines.append(ax.plot(epoch_range, history["fret_accuracy_overall"], "o-", label="Val Fret Acc (Overall)"))
    if "fret_accuracy_active" in history and history["fret_accuracy_active"]:
        plot_lines.append(ax.plot(epoch_range, history["fret_accuracy_active"], "o-", label="Val Fret Acc (Active)"))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Dokładność")
    ax.set_title("Metryki Walidacyjne Progów")
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    if plot_lines: ax.legend()

    # --- Wykres 5: Strata Progów (Treningowa i Walidacyjna) ---
    ax = axs[2, 0]
    plot_lines = []
    if "train_fret_loss" in history and history["train_fret_loss"]:
        plot_lines.append(ax.plot(epoch_range, history["train_fret_loss"], "o-", label="Train Fret Loss"))
    if "val_fret_loss" in history and history["val_fret_loss"]:
        plot_lines.append(ax.plot(epoch_range, history["val_fret_loss"], "o-", label="Val Fret Loss"))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Strata")
    ax.set_title("Strata Progów")
    ax.grid(True)
    if plot_lines: ax.legend()

    # --- Wykres 6: Współczynnik Uczenia (Learning Rate) ---
    ax = axs[2, 1]
    plot_lines = []
    if "lr" in history and history["lr"]:
        plot_lines.append(ax.plot(epoch_range, history["lr"], "o-", label="Learning Rate", color="purple"))
        ax.set_yscale("log") # Skala logarytmiczna dla LR
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate")
    ax.grid(True)
    if plot_lines: ax.legend()

    # --- Wykres 7: Optymalny Próg Walidacyjny Onsetów (znaleziony w każdej epoce) ---
    ax = axs[3, 0]
    plot_lines = []
    if "val_optimal_onset_threshold_epoch" in history and history["val_optimal_onset_threshold_epoch"]:
        plot_lines.append(ax.plot(epoch_range, history["val_optimal_onset_threshold_epoch"], "o-", label="Optymalny Próg Onsetów", color="teal"))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Próg")
    ax.set_title("Optymalny Próg Walidacyjny Onsetów")
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    if plot_lines: ax.legend()

    # --- Wykres 8: Metryki Walidacyjne Onsetów (przy stałym progu, np. 0.5) ---
    ax = axs[3, 1]
    plot_lines = []
    if "onset_f1_at_fixed_thresh" in history and history["onset_f1_at_fixed_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["onset_f1_at_fixed_thresh"], "o-", label="Val Onset F1 (Th=0.5)"))
    if "onset_precision_at_fixed_thresh" in history and history["onset_precision_at_fixed_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["onset_precision_at_fixed_thresh"], "o-", label="Val Onset Precision (Th=0.5)"))
    if "onset_recall_at_fixed_thresh" in history and history["onset_recall_at_fixed_thresh"]:
        plot_lines.append(ax.plot(epoch_range, history["onset_recall_at_fixed_thresh"], "o-", label="Val Onset Recall (Th=0.5)"))
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Metryka (Th=0.5)")
    ax.set_title("Metryki Walidacyjne Onsetów (Próg=0.5)")
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    if plot_lines: ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Zapisano wykres historii treningu do: {save_path}")
        except Exception as e:
            print(f"Nie udało się zapisać wykresu historii do {save_path}: {e}")
    plt.close(fig)