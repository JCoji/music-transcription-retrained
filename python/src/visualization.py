import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import (
    AUDIO_SAMPLE_RATE,
    ANNOTATION_HOP,
    FREQ_BINS_NOTES,
    NOTES_BINS_PER_SEMITONE,
    N_FREQ_BINS_NOTES,
    N_FREQ_BINS_CONTOURS,
    GUITAR_BASE_FREQUENCY,
)


def plot_sample_data(
    batch,
    sample_idx=0,
    sr=AUDIO_SAMPLE_RATE,
    hop_length_time=ANNOTATION_HOP,
    freq_bins_cqt_param=FREQ_BINS_NOTES,
    notes_bins_per_semitone_param=NOTES_BINS_PER_SEMITONE,
):
    """
    Tworzy i wyświetla wykresy dla pojedynczej próbki danych z batcha.

    Wizualizuje: spektrogram CQT z onsetami, macierze "ground truth" dla nut,
    onsetów i konturów częstotliwości. Wyświetla również podstawowe informacje o tensorach.

    Args:
        batch (dict): Słownik z danymi ('features', 'notes', 'onsets', 'contours').
        sample_idx (int): Indeks próbki w batchu.
        sr (int): Częstotliwość próbkowania audio (Hz).
        hop_length_time (float): Czas trwania kroku (hop) w sekundach.
        freq_bins_cqt_param (np.ndarray): Częstotliwości binów CQT.
        notes_bins_per_semitone_param (int): Liczba binów na półton.
    """
    features_tensor = batch["features"][sample_idx].cpu()
    notes_target_tensor = batch["notes"][sample_idx].cpu()
    onsets_target_tensor = batch["onsets"][sample_idx].cpu()
    contours_target_tensor = batch["contours"][sample_idx].cpu()
    feature_length = batch["feature_lengths"][sample_idx].item()

    # Przycięcie tensorów do rzeczywistej długości sekwencji
    features_tensor = features_tensor[:feature_length, :]
    notes_target_tensor = notes_target_tensor[:feature_length, :]
    onsets_target_tensor = onsets_target_tensor[:feature_length, :]
    contours_target_tensor = contours_target_tensor[:feature_length, :]

    features_np = features_tensor.numpy()
    notes_target_np = notes_target_tensor.numpy()
    onsets_target_np = onsets_target_tensor.numpy()
    contours_target_np = contours_target_tensor.numpy()

    times = np.arange(feature_length) * hop_length_time
    hop_length_samples = int(hop_length_time * sr)

    # Użycie rzeczywistych wymiarów częstotliwości z danych do rysowania
    actual_n_freq_bins_notes = notes_target_np.shape[1]
    actual_n_freq_bins_contours = contours_target_np.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

    # 1. Wykres CQT (features) z nałożonymi onsetami
    img_cqt = librosa.display.specshow(
        features_np.T,
        sr=sr,
        hop_length=hop_length_samples,
        x_axis="time",
        y_axis="cqt_hz",
        bins_per_octave=12 * notes_bins_per_semitone_param,
        fmin=librosa.midi_to_hz(librosa.note_to_midi("C1")),  # Przykładowe fmin
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title(f"Spektrogram CQT (Próbka {sample_idx})")
    axes[0].set_ylabel("Częstotliwość (Hz)")
    fig.colorbar(img_cqt, ax=axes[0], format="%+2.0f dB")

    # Nakładanie lokalizacji onsetów
    onset_time_indices, onset_freq_indices = torch.where(
        onsets_target_tensor > 0.5
    )  # Próg detekcji onsetu
    onset_times_vis = onset_time_indices.numpy() * hop_length_time

    np_freq_bins_cqt = np.array(freq_bins_cqt_param)
    valid_onset_freq_indices = onset_freq_indices[
        onset_freq_indices < len(np_freq_bins_cqt)
    ]
    onset_freqs_vis = np_freq_bins_cqt[valid_onset_freq_indices.numpy()]

    min_len = min(len(onset_times_vis), len(onset_freqs_vis))
    axes[0].scatter(
        onset_times_vis[:min_len],
        onset_freqs_vis[:min_len],
        c="cyan",
        marker="o",
        s=30,
        label="OnSety (>0.5)",
        alpha=0.7,
        edgecolors="black",
    )
    if len(onset_times_vis) > 0:
        axes[0].legend(loc="upper right")

    # 2. Wykres notes_target
    axes[1].imshow(
        notes_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="gray_r",  # PIERWOWZÓR
        extent=[times.min(), times.max(), -0.5, actual_n_freq_bins_notes - 0.5],
    )
    axes[1].set_title("Referencyjne Nuty (Ground Truth)")
    axes[1].set_ylabel(f"Bin częstotliwości ({actual_n_freq_bins_notes})")
    if actual_n_freq_bins_notes > 20 and notes_bins_per_semitone_param > 0:
        tick_interval = notes_bins_per_semitone_param * 12
        axes[1].set_yticks(np.arange(0, actual_n_freq_bins_notes, tick_interval))
        axes[1].set_yticklabels(
            [
                f"Okt. {i // tick_interval}"
                for i in np.arange(0, actual_n_freq_bins_notes, tick_interval)
            ]
        )

    # 3. Wykres onsets_target
    axes[2].imshow(
        onsets_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="Greens",  # PIERWOWZÓR
        extent=[times.min(), times.max(), -0.5, actual_n_freq_bins_notes - 0.5],
    )
    axes[2].set_title("Referencyjne OnSety - Mapa Ciepła")
    axes[2].set_ylabel(f"Bin częstotliwości ({actual_n_freq_bins_notes})")
    if actual_n_freq_bins_notes > 20 and notes_bins_per_semitone_param > 0:
        tick_interval = notes_bins_per_semitone_param * 12
        axes[2].set_yticks(np.arange(0, actual_n_freq_bins_notes, tick_interval))
        axes[2].set_yticklabels(
            [
                f"Okt. {i // tick_interval}"
                for i in np.arange(0, actual_n_freq_bins_notes, tick_interval)
            ]
        )

    # 4. Wykres contours_target
    img_contours = axes[3].imshow(
        contours_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",  # PIERWOWZÓR
        extent=[times.min(), times.max(), -0.5, actual_n_freq_bins_contours - 0.5],
    )
    axes[3].set_title("Referencyjne Kontury Częstotliwości")
    axes[3].set_ylabel(f"Bin częstotliwości ({actual_n_freq_bins_contours})")
    axes[3].set_xlabel("Czas (s)")
    fig.colorbar(img_contours, ax=axes[3], label="Aktywacja konturu")
    if actual_n_freq_bins_contours > 20 and notes_bins_per_semitone_param > 0:
        tick_interval = notes_bins_per_semitone_param * 12
        axes[3].set_yticks(np.arange(0, actual_n_freq_bins_contours, tick_interval))
        axes[3].set_yticklabels(
            [
                f"Okt. {i // tick_interval}"
                for i in np.arange(0, actual_n_freq_bins_contours, tick_interval)
            ]
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print("\n# Podstawowe sprawdzenia tensorów po przycięciu:")
    print(
        f"  Kształt CQT (features): {features_tensor.shape}, Niezerowe: {torch.count_nonzero(features_tensor)}"
    )
    print(
        f"  Kształt notes_target: {notes_target_tensor.shape}, Min: {notes_target_tensor.min():.2f}, Max: {notes_target_tensor.max():.2f}, Niezerowe: {torch.count_nonzero(notes_target_tensor)}"
    )
    print(
        f"  Kształt onsets_target: {onsets_target_tensor.shape}, Min: {onsets_target_tensor.min():.2f}, Max: {onsets_target_tensor.max():.2f}, Niezerowe: {torch.count_nonzero(onsets_target_tensor)}"
    )
    print(
        f"  Kształt contours_target: {contours_target_tensor.shape}, Min: {contours_target_tensor.min():.2f}, Max: {contours_target_tensor.max():.2f}, Niezerowe: {torch.count_nonzero(contours_target_tensor)}"
    )

    if torch.count_nonzero(features_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'features' jest pusty!")
    if torch.count_nonzero(notes_target_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'notes_target' jest pusty!")
    if torch.count_nonzero(onsets_target_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'onsets_target' jest pusty!")
    if torch.count_nonzero(contours_target_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'contours_target' jest pusty!")


def plot_sample_data_with_predictions(
    batch,
    predictions_probs,
    sample_idx=0,
    sr=AUDIO_SAMPLE_RATE,
    hop_length_time=ANNOTATION_HOP,
    notes_bins_per_semitone_cfg=NOTES_BINS_PER_SEMITONE,
    n_freq_bins_notes_cfg=N_FREQ_BINS_NOTES,
    n_freq_bins_contours_cfg=N_FREQ_BINS_CONTOURS,
    guitar_base_freq_cfg=GUITAR_BASE_FREQUENCY,
    save_path=None,
    threshold=0.5,
):
    """
    Tworzy wykresy porównujące dane referencyjne z predykcjami modelu.

    Wizualizuje: CQT, referencyjne i predykowane nuty, onSety oraz kontury.

    Args:
        batch (dict): Słownik z danymi referencyjnymi (CPU).
        predictions_probs (dict): Słownik z prawdopodobieństwami predykcji (numpy).
        sample_idx (int): Indeks próbki.
        sr (int): Częstotliwość próbkowania (Hz).
        hop_length_time (float): Czas kroku (hop) w sekundach.
        notes_bins_per_semitone_cfg (int): Liczba binów na półton.
        n_freq_bins_notes_cfg (int): Liczba binów dla nut.
        n_freq_bins_contours_cfg (int): Liczba binów dla konturów.
        guitar_base_freq_cfg (float): Bazowa częstotliwość instrumentu (Hz).
        save_path (str, optional): Ścieżka do zapisu wykresu.
        threshold (float): Próg binaryzacji predykcji.
    """
    features_tensor = batch["features"][sample_idx]
    notes_target_tensor = batch["notes"][sample_idx]
    onsets_target_tensor = batch["onsets"][sample_idx]
    contours_target_tensor = batch["contours"][sample_idx]
    feature_length = batch["feature_lengths"][sample_idx].item()

    features_tensor = features_tensor[:feature_length, :]
    notes_target_tensor = notes_target_tensor[:feature_length, :]
    onsets_target_tensor = onsets_target_tensor[:feature_length, :]
    contours_target_tensor = contours_target_tensor[:feature_length, :]

    notes_pred_prob_np = predictions_probs["notes"][sample_idx][:feature_length, :]
    onsets_pred_prob_np = predictions_probs["onsets"][sample_idx][:feature_length, :]
    contours_pred_prob_np = predictions_probs["contours"][sample_idx][
        :feature_length, :
    ]

    notes_pred_binary_np = (notes_pred_prob_np > threshold).astype(float)
    onsets_pred_binary_np = (onsets_pred_prob_np > threshold).astype(float)

    features_np = (
        features_tensor.numpy()
        if isinstance(features_tensor, torch.Tensor)
        else features_tensor
    )
    notes_target_np = (
        notes_target_tensor.numpy()
        if isinstance(notes_target_tensor, torch.Tensor)
        else notes_target_tensor
    )
    onsets_target_np = (
        onsets_target_tensor.numpy()
        if isinstance(onsets_target_tensor, torch.Tensor)
        else onsets_target_tensor
    )
    contours_target_np = (
        contours_target_tensor.numpy()
        if isinstance(contours_target_tensor, torch.Tensor)
        else contours_target_tensor
    )

    times = np.arange(feature_length) * hop_length_time
    hop_length_samples = int(hop_length_time * sr)

    fig, axes = plt.subplots(7, 1, figsize=(15, 28), sharex=True)

    # 1. Spektrogram CQT
    try:
        img_cqt = librosa.display.specshow(
            features_np.T,
            sr=sr,
            hop_length=hop_length_samples,
            x_axis="time",
            y_axis="cqt_hz",
            fmin=guitar_base_freq_cfg,
            bins_per_octave=12 * notes_bins_per_semitone_cfg,
            ax=axes[0],
            cmap="magma",  # ZGODNE z plot_sample_data
        )
    except Exception as e:
        print(
            f"Błąd przy librosa.display.specshow dla CQT: {e}. Rysuję z parametrami awaryjnymi."
        )
        img_cqt = librosa.display.specshow(
            features_np.T,
            sr=sr,
            hop_length=hop_length_samples,
            x_axis="time",
            y_axis="log",
            ax=axes[0],
            cmap="magma",
        )
    axes[0].set_title(f"Spektrogram CQT (Próbka {sample_idx})")
    axes[0].set_ylabel("Częstotliwość (Hz)")
    fig.colorbar(img_cqt, ax=axes[0], format="%+2.0f dB")

    # 2. Referencyjne Nuty
    axes[1].imshow(
        notes_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="gray_r",  # ZGODNE z plot_sample_data
        extent=[times.min(), times.max(), -0.5, n_freq_bins_notes_cfg - 0.5],
    )
    axes[1].set_title("Referencyjne Nuty")
    axes[1].set_ylabel(f"Bin ({n_freq_bins_notes_cfg})")

    # 3. Predykowane Nuty
    axes[2].imshow(
        notes_pred_binary_np.T,
        aspect="auto",
        origin="lower",
        cmap="Blues",  # Odróżniająca mapa dla predykcji
        extent=[times.min(), times.max(), -0.5, n_freq_bins_notes_cfg - 0.5],
    )
    axes[2].set_title(f"Predykowane Nuty (Próg: {threshold})")
    axes[2].set_ylabel(f"Bin ({n_freq_bins_notes_cfg})")

    # 4. Referencyjne OnSety
    axes[3].imshow(
        onsets_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="Greens",  # ZMIENIONE na zgodne z plot_sample_data (było Greens_r)
        extent=[times.min(), times.max(), -0.5, n_freq_bins_notes_cfg - 0.5],
    )
    axes[3].set_title("Referencyjne OnSety")
    axes[3].set_ylabel(f"Bin ({n_freq_bins_notes_cfg})")

    # 5. Predykowane OnSety
    axes[4].imshow(
        onsets_pred_binary_np.T,
        aspect="auto",
        origin="lower",
        cmap="Reds",  # Odróżniająca mapa dla predykcji
        extent=[times.min(), times.max(), -0.5, n_freq_bins_notes_cfg - 0.5],
    )
    axes[4].set_title(f"Predykowane OnSety (Próg: {threshold})")
    axes[4].set_ylabel(f"Bin ({n_freq_bins_notes_cfg})")

    # 6. Referencyjne Kontury
    axes[5].imshow(
        contours_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",  # ZMIENIONE na zgodne z plot_sample_data (było viridis_r)
        extent=[times.min(), times.max(), -0.5, n_freq_bins_contours_cfg - 0.5],
    )
    axes[5].set_title("Referencyjne Kontury")
    axes[5].set_ylabel(f"Bin ({n_freq_bins_contours_cfg})")

    # 7. Predykowane Kontury
    img_contours_pred = axes[6].imshow(
        contours_pred_prob_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",  # ZGODNE z referencyjnymi konturami (i plot_sample_data)
        vmin=0,
        vmax=1,
        extent=[times.min(), times.max(), -0.5, n_freq_bins_contours_cfg - 0.5],
    )
    axes[6].set_title("Predykowane Kontury (Prawdopodobieństwa)")
    axes[6].set_ylabel(f"Bin ({n_freq_bins_contours_cfg})")
    axes[6].set_xlabel("Czas (s)")
    fig.colorbar(img_contours_pred, ax=axes[6], label="Prawdopodobieństwo")

    # Ustawienie etykiet osi Y na nazwy nut
    if notes_bins_per_semitone_cfg > 0:
        tick_interval_semitones = 12
        tick_interval_bins = notes_bins_per_semitone_cfg * tick_interval_semitones
        try:
            base_midi_note = librosa.hz_to_midi(guitar_base_freq_cfg)
        except Exception:
            base_midi_note = librosa.note_to_midi("E2")

        for ax_idx, n_bins_total_for_axis in [
            (1, n_freq_bins_notes_cfg),
            (2, n_freq_bins_notes_cfg),
            (3, n_freq_bins_notes_cfg),
            (4, n_freq_bins_notes_cfg),
            (5, n_freq_bins_contours_cfg),
            (6, n_freq_bins_contours_cfg),
        ]:
            if (
                n_bins_total_for_axis > 0
                and 0
                < tick_interval_bins
                <= n_bins_total_for_axis  # Poprawiony warunek
            ):
                yticks_pos = np.arange(0, n_bins_total_for_axis, tick_interval_bins)
                yticks_labels = []
                for bin_pos_val in yticks_pos:
                    semitone_offset = round(bin_pos_val / notes_bins_per_semitone_cfg)
                    current_midi = base_midi_note + semitone_offset
                    try:
                        note_name = librosa.midi_to_note(
                            int(round(current_midi)), octave=True
                        )
                        yticks_labels.append(note_name)
                    except Exception:
                        yticks_labels.append(f"S{semitone_offset:.0f}")

                if yticks_pos.size > 0:
                    axes[ax_idx].set_yticks(yticks_pos)
                    axes[ax_idx].set_yticklabels(yticks_labels)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Wykres zapisany do: {save_path}")
        except Exception as e:
            print(f"Nie udało się zapisać wykresu do {save_path}: {e}")
        plt.close(fig)
    else:
        plt.show()
