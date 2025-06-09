import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
import config

def plot_spectrogram(spectrogram_tensor, sr, hop_length, ax=None, title="Mel-Spektrogram"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    spec_np = spectrogram_tensor.cpu().numpy()
    img = librosa.display.specshow(spec_np, sr=sr, hop_length=hop_length, x_axis="time", y_axis="cqt_hz", ax=ax, cmap="magma")
    ax.set_title(title)
    ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig, ax

def plot_onset_labels(onset_targets_tensor, sr, hop_length, num_strings=config.DEFAULT_NUM_STRINGS, ax=None, title="Etykiety Onsetów"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))
    onsets_np = onset_targets_tensor.cpu().numpy().T
    librosa.display.specshow(onsets_np, sr=sr, hop_length=hop_length, x_axis="time", ax=ax, cmap="gray_r", vmin=0, vmax=1)
    ax.set_yticks(np.arange(num_strings))
    ax.set_yticklabels([f"Str {i}" for i in range(num_strings)])
    ax.set_ylabel("Struna")
    ax.set_title(title)
    return fig, ax

def plot_fret_labels(fret_targets_tensor, sr, hop_length, num_strings=config.DEFAULT_NUM_STRINGS, max_fret_display=(config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET), ax=None, title="Etykiety Progów"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))
    frets_np = fret_targets_tensor.cpu().numpy().T
    custom_cmap = plt.cm.get_cmap("viridis", max_fret_display + 1)
    img = ax.pcolormesh(
        librosa.frames_to_time(np.arange(frets_np.shape[1] + 1), sr=sr, hop_length=hop_length),
        np.arange(num_strings + 1),
        frets_np, cmap=custom_cmap, vmin=0, vmax=max_fret_display, shading='auto'
    )
    ax.set_yticks(np.arange(num_strings) + 0.5)
    ax.set_yticklabels([f"Str {i}" for i in range(num_strings)])
    ax.set_ylabel("Struna")
    ax.set_xlabel("Czas (s)")
    ax.set_title(title)
    cbar = ax.figure.colorbar(img, ax=ax, ticks=np.arange(0, max_fret_display + 1, max(1, (max_fret_display + 1) // 10)))
    cbar.set_label(f"Numer progu (cisza >= {max_fret_display})")
    return fig, ax

def plot_sample_data_summary(features_tensor, onset_labels_tensor, fret_labels_tensor, sampling_rate, hop_len, track_id_display=None):
    fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    main_title = "Wizualizacja Próbki Danych (Ground Truth)"
    if track_id_display:
        main_title = f"{main_title}\nUtwór: {track_id_display}"
    fig.suptitle(main_title, fontsize=16)
    plot_spectrogram(features_tensor, sr=sampling_rate, hop_length=hop_len, ax=axs[0])
    plot_onset_labels(onset_labels_tensor, sr=sampling_rate, hop_length=hop_len, ax=axs[1])
    plot_fret_labels(fret_labels_tensor, sr=sampling_rate, hop_length=hop_len, max_fret_display=(config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET), ax=axs[2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)

def plot_training_history(history_data, output_save_path=None, figure_size=(16, 10)):
    if not history_data or not history_data.get("train_total_loss"):
        return

    epochs = range(1, len(history_data["train_total_loss"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=figure_size)
    fig.suptitle("Podsumowanie Historii Treningu", fontsize=18)

    ax = axs[0, 0]
    ax.plot(epochs, history_data.get("train_total_loss", []), 'o-', label="Strata Treningowa")
    ax.plot(epochs, history_data.get("val_total_loss", []), 'o-', label="Strata Walidacyjna")
    ax.set_title("Strata (Loss)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Strata")
    ax.grid(True)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(epochs, history_data.get("lr", []), 'o-', label="Learning Rate", color="purple")
    ax.set_title("Współczynnik Uczenia (LR)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(epochs, history_data.get("val_tdr_f1_at_0.5", []), 'o-', label="TDR F1 (nuty)")
    ax.plot(epochs, history_data.get("val_tdr_precision_at_0.5", []), 's--', label="TDR Precision", alpha=0.6)
    ax.plot(epochs, history_data.get("val_tdr_recall_at_0.5", []), '^-', label="TDR Recall", alpha=0.6)
    ax.set_title("Metryki Nutowe (TDR @ thr=0.5)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Wynik")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axs[1, 1]
    ax.plot(epochs, history_data.get("val_mpe_f1", []), 'o-', label="MPE F1 (ramki)")
    ax.plot(epochs, history_data.get("val_onset_f1_event_at_0.5", []), 's-', label="Onset F1 (event @ thr=0.5)")
    ax.set_title("Metryki Ramek i Zdarzeń")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Wynik")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_save_path:
        try:
            plt.savefig(output_save_path)
        except Exception as e:
            print(f"Błąd podczas zapisu wykresu historii: {e}")
    plt.show()
    plt.close(fig)