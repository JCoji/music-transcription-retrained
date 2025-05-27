import librosa
import librosa.display
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
    freq_bins_cqt=FREQ_BINS_NOTES,
    notes_bins_per_semitone=NOTES_BINS_PER_SEMITONE,
):
    # Ekstrakcja danych jako tensory PyTorch do obliczeń i sprawdzeń
    features_tensor = batch["features"][sample_idx].cpu()
    notes_target_tensor = batch["notes"][sample_idx].cpu()
    onsets_target_tensor = batch["onsets"][sample_idx].cpu()
    contours_target_tensor = batch["contours"][sample_idx].cpu()
    feature_length = batch["feature_lengths"][sample_idx].item()

    # Przycięcie do rzeczywistej długości
    features_tensor = features_tensor[:feature_length, :]
    notes_target_tensor = notes_target_tensor[:feature_length, :]
    onsets_target_tensor = onsets_target_tensor[:feature_length, :]
    contours_target_tensor = contours_target_tensor[:feature_length, :]

    # Konwersja do NumPy dla celów plottingu (po sprawdzeniach)
    features_np = features_tensor.numpy()
    notes_target_np = notes_target_tensor.numpy()
    onsets_target_np = onsets_target_tensor.numpy()
    contours_target_np = contours_target_tensor.numpy()

    # Przygotowanie osi czasu
    times = np.arange(feature_length) * hop_length_time
    hop_length_samples = int(hop_length_time * sr)

    # Pobranie wymiarów częstotliwości
    N_FREQ_BINS_NOTES = notes_target_np.shape[1]
    N_FREQ_BINS_CONTOURS = contours_target_np.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

    # 1. Wykres CQT (features) z nałożonymi onsetami
    img_cqt = librosa.display.specshow(
        features_np.T,
        sr=sr,
        hop_length=hop_length_samples,
        x_axis="time",
        y_axis="log",
        y_coords=freq_bins_cqt,
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title(f"CQT Spectrogram (Sample {sample_idx})")
    axes[0].set_ylabel("Częstotliwość (Hz)")
    fig.colorbar(img_cqt, ax=axes[0], format="%+2.0f dB")

    # Nakładanie lokalizacji onsetów na wykres CQT
    onset_time_indices, onset_freq_indices = torch.where(onsets_target_tensor > 0.5)
    onset_times_vis = onset_time_indices.numpy() * hop_length_time

    # Upewnij się, że freq_bins_cqt jest tablicą NumPy do poprawnego indeksowania
    np_freq_bins_cqt = np.array(freq_bins_cqt)
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
        label="Onsets (>0.5)",
        alpha=0.7,
        edgecolors="black",
    )
    if len(onset_times_vis) > 0:
        axes[0].legend(loc="upper right")

    # 2. Wykres notes_target jako mapa ciepła
    axes[1].imshow(
        notes_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        extent=[times.min(), times.max(), -0.5, N_FREQ_BINS_NOTES - 0.5],
    )
    axes[1].set_title("Ground Truth Notes")
    axes[1].set_ylabel("Bin częstotliwości (Notes)")
    if N_FREQ_BINS_NOTES > 20 and notes_bins_per_semitone > 0:
        tick_interval = notes_bins_per_semitone * 12
        axes[1].set_yticks(np.arange(0, N_FREQ_BINS_NOTES, tick_interval))
        axes[1].set_yticklabels(
            [
                f"Oct {i // tick_interval}"
                for i in np.arange(0, N_FREQ_BINS_NOTES, tick_interval)
            ]
        )

    # 3. Wykres onsets_target jako mapa ciepła
    axes[2].imshow(
        onsets_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="Greens",
        extent=[times.min(), times.max(), -0.5, N_FREQ_BINS_NOTES - 0.5],
    )
    axes[2].set_title("Ground Truth Onsets - Heatmap")
    axes[2].set_ylabel("Bin częstotliwości (Notes)")
    if N_FREQ_BINS_NOTES > 20 and notes_bins_per_semitone > 0:
        tick_interval = notes_bins_per_semitone * 12
        axes[2].set_yticks(np.arange(0, N_FREQ_BINS_NOTES, tick_interval))
        axes[2].set_yticklabels(
            [
                f"Oct {i // tick_interval}"
                for i in np.arange(0, N_FREQ_BINS_NOTES, tick_interval)
            ]
        )

    # 4. Wykres contours_target jako mapa ciepła
    img_contours = axes[3].imshow(
        contours_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[times.min(), times.max(), -0.5, N_FREQ_BINS_CONTOURS - 0.5],
    )
    axes[3].set_title("Ground Truth Contours")
    axes[3].set_ylabel("Bin częstotliwości (Contours)")
    axes[3].set_xlabel("Czas (s)")
    fig.colorbar(img_contours, ax=axes[3], label="Aktywacja konturu")
    if N_FREQ_BINS_CONTOURS > 20 and notes_bins_per_semitone > 0:
        tick_interval = notes_bins_per_semitone * 12
        axes[3].set_yticks(np.arange(0, N_FREQ_BINS_CONTOURS, tick_interval))
        axes[3].set_yticklabels(
            [
                f"Oct {i // tick_interval}"
                for i in np.arange(0, N_FREQ_BINS_CONTOURS, tick_interval)
            ]
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print("# Podstawowe sprawdzenia tensorów")
    print(f"  Kształt CQT (features): {features_tensor.shape}")
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
        print("  OSTRZEŻENIE: Tensor 'features' jest pusty (same zera)!")
    if torch.count_nonzero(notes_target_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'notes_target' jest pusty (same zera)!")
    if torch.count_nonzero(onsets_target_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'onsets_target' jest pusty (same zera)!")
    if torch.count_nonzero(contours_target_tensor) == 0:
        print("  OSTRZEŻENIE: Tensor 'contours_target' jest pusty (same zera)!")


def plot_sample_data_with_predictions(
    batch,
    predictions_probs,
    sample_idx=0,
    sr=AUDIO_SAMPLE_RATE,
    hop_length_time=ANNOTATION_HOP,
    notes_bins_per_semitone_cfg=NOTES_BINS_PER_SEMITONE,
    save_path=None,
    threshold=0.5,
):
    # batch jest już na CPU z funkcji evaluate_epoch
    features_tensor = batch["features"][sample_idx]
    notes_target_tensor = batch["notes"][sample_idx]
    onsets_target_tensor = batch["onsets"][sample_idx]
    contours_target_tensor = batch["contours"][sample_idx]
    feature_length = batch["feature_lengths"][sample_idx].item()

    # Przycięcie TARGETÓW i CECH do rzeczywistej długości
    features_tensor = features_tensor[:feature_length, :]
    notes_target_tensor = notes_target_tensor[:feature_length, :]
    onsets_target_tensor = onsets_target_tensor[:feature_length, :]
    contours_target_tensor = contours_target_tensor[:feature_length, :]

    # predictions_probs to słownik numpy array
    notes_pred_prob_np = predictions_probs["notes"][:feature_length, :]
    onsets_pred_prob_np = predictions_probs["onsets"][:feature_length, :]
    contours_pred_prob_np = predictions_probs["contours"][:feature_length, :]

    # Konwersja predykcji do binarnych
    notes_pred_binary_np = (notes_pred_prob_np > threshold).astype(float)
    onsets_pred_binary_np = (onsets_pred_prob_np > threshold).astype(float)

    # Konwersja targetów do NumPy
    features_np = features_tensor.numpy()
    notes_target_np = notes_target_tensor.numpy()
    onsets_target_np = onsets_target_tensor.numpy()
    contours_target_np = contours_target_tensor.numpy()

    times = np.arange(feature_length) * hop_length_time
    hop_length_samples = int(hop_length_time * sr)

    _N_FREQ_BINS_NOTES_CFG = N_FREQ_BINS_NOTES
    _N_FREQ_BINS_CONTOURS_CFG = N_FREQ_BINS_CONTOURS

    fig, axes = plt.subplots(7, 1, figsize=(15, 28), sharex=True)

    # 1. CQT Spectrogram
    try:
        img_cqt = librosa.display.specshow(
            features_np.T,
            sr=sr,
            hop_length=hop_length_samples,
            x_axis="time",
            y_axis="cqt_hz",
            fmin=GUITAR_BASE_FREQUENCY,
            bins_per_octave=12 * notes_bins_per_semitone_cfg,
            ax=axes[0],
            cmap="magma",
        )
    except Exception as e:
        print(
            f"Błąd przy specshow dla CQT: {e}. Rysuję bez niektórych parametrów osi Y."
        )
        img_cqt = librosa.display.specshow(
            features_np.T,
            sr=sr,
            hop_length=hop_length_samples,
            x_axis="time",
            y_axis="log",  # Fallback
            ax=axes[0],
            cmap="magma",
        )
    axes[0].set_title(f"CQT Spectrogram (Sample {sample_idx})")
    axes[0].set_ylabel("Częstotliwość (Hz)")
    fig.colorbar(img_cqt, ax=axes[0], format="%+2.0f dB")

    # 2. Ground Truth Notes
    axes[1].imshow(
        notes_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        extent=[times.min(), times.max(), -0.5, _N_FREQ_BINS_NOTES_CFG - 0.5],
    )
    axes[1].set_title("Ground Truth Notes")
    axes[1].set_ylabel(f"Bin ({_N_FREQ_BINS_NOTES_CFG})")

    # 3. Predicted Notes (binarne)
    axes[2].imshow(
        notes_pred_binary_np.T,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        extent=[times.min(), times.max(), -0.5, _N_FREQ_BINS_NOTES_CFG - 0.5],
    )
    axes[2].set_title(f"Predicted Notes (Th: {threshold})")
    axes[2].set_ylabel(f"Bin ({_N_FREQ_BINS_NOTES_CFG})")

    # 4. Ground Truth Onsets
    axes[3].imshow(
        onsets_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="Greens_r",
        extent=[times.min(), times.max(), -0.5, _N_FREQ_BINS_NOTES_CFG - 0.5],
    )
    axes[3].set_title("Ground Truth Onsets")
    axes[3].set_ylabel(f"Bin ({_N_FREQ_BINS_NOTES_CFG})")

    # 5. Predicted Onsets (binarne)
    axes[4].imshow(
        onsets_pred_binary_np.T,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        extent=[times.min(), times.max(), -0.5, _N_FREQ_BINS_NOTES_CFG - 0.5],
    )
    axes[4].set_title(f"Predicted Onsets (Th: {threshold})")
    axes[4].set_ylabel(f"Bin ({_N_FREQ_BINS_NOTES_CFG})")

    # 6. Ground Truth Contours
    axes[5].imshow(
        contours_target_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis_r",
        extent=[times.min(), times.max(), -0.5, _N_FREQ_BINS_CONTOURS_CFG - 0.5],
    )
    axes[5].set_title("Ground Truth Contours")
    axes[5].set_ylabel(f"Bin ({_N_FREQ_BINS_CONTOURS_CFG})")

    # 7. Predicted Contours (prawdopodobieństwa)
    img_contours_pred = axes[6].imshow(
        contours_pred_prob_np.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[times.min(), times.max(), -0.5, _N_FREQ_BINS_CONTOURS_CFG - 0.5],
    )
    axes[6].set_title("Predicted Contours (Probabilities)")
    axes[6].set_ylabel(f"Bin ({_N_FREQ_BINS_CONTOURS_CFG})")
    axes[6].set_xlabel("Czas (s)")
    fig.colorbar(img_contours_pred, ax=axes[6], label="Prawdopodobieństwo")

    if notes_bins_per_semitone_cfg > 0:
        tick_interval_semitones = 12
        tick_interval_bins = notes_bins_per_semitone_cfg * tick_interval_semitones
        base_midi_note = librosa.note_to_midi("E2")

        for ax_idx, n_bins_total in [
            (1, _N_FREQ_BINS_NOTES_CFG),
            (2, _N_FREQ_BINS_NOTES_CFG),
            (3, _N_FREQ_BINS_NOTES_CFG),
            (4, _N_FREQ_BINS_NOTES_CFG),
            (5, _N_FREQ_BINS_CONTOURS_CFG),
            (6, _N_FREQ_BINS_CONTOURS_CFG),
        ]:
            if n_bins_total > 0 and tick_interval_bins > 0:
                yticks_pos = np.arange(0, n_bins_total, tick_interval_bins)
                yticks_labels = []
                for i, bin_pos_val in enumerate(yticks_pos):  # Use the value of bin_pos
                    semitone_offset = round(bin_pos_val / notes_bins_per_semitone_cfg)
                    current_midi = base_midi_note + semitone_offset
                    try:
                        note_name = librosa.midi_to_note(
                            int(round(current_midi)), octave=True
                        )  # Ensure midi is int
                        yticks_labels.append(note_name)
                    except Exception:
                        yticks_labels.append(f"S{semitone_offset}")

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
