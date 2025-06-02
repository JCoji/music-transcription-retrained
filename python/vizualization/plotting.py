import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
import config


def plot_spectrogram(
        spectrogram_tensor, sr, hop_length, ax=None, title="Mel-Spektrogram"
):
    if ax is None:
        fig_obj, ax_obj = plt.subplots(figsize=(12, 4))
    else:
        fig_obj = ax.figure
        ax_obj = ax

    spec_np = spectrogram_tensor.cpu().numpy() if isinstance(spectrogram_tensor, torch.Tensor) else spectrogram_tensor

    img = librosa.display.specshow(
        spec_np,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax_obj,
        cmap="magma",
    )
    ax_obj.set_title(title)
    fig_obj.colorbar(img, ax=ax_obj, format="%+2.0f dB")
    return fig_obj, ax_obj


def plot_onset_labels(
        onset_targets_tensor, sr, hop_length, num_strings=config.DEFAULT_NUM_STRINGS,
        ax=None, title="Etykiety Onsetów (GT)"
):
    if ax is None:
        fig_obj, ax_obj = plt.subplots(figsize=(12, 2))
    else:
        fig_obj = ax.figure
        ax_obj = ax

    onsets_np = onset_targets_tensor.cpu().numpy() if isinstance(onset_targets_tensor,
                                                                 torch.Tensor) else onset_targets_tensor

    onsets_to_plot = onsets_np.T
    librosa.display.specshow(
        onsets_to_plot,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        ax=ax_obj,
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    ax_obj.set_yticks(np.arange(num_strings))
    ax_obj.set_yticklabels([f"Str {i}" for i in range(num_strings)])
    ax_obj.set_ylabel("Struna")
    ax_obj.set_title(title)
    return fig_obj, ax_obj


def plot_fret_labels(
        fret_targets_tensor, sr, hop_length,
        num_strings=config.DEFAULT_NUM_STRINGS,
        max_fret_display=(config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET),
        ax=None, title="Etykiety Progów (GT)"
):
    if ax is None:
        fig_obj, ax_obj = plt.subplots(figsize=(12, 2))
    else:
        fig_obj = ax.figure
        ax_obj = ax

    frets_np = fret_targets_tensor.cpu().numpy() if isinstance(fret_targets_tensor,
                                                               torch.Tensor) else fret_targets_tensor

    frets_to_plot = frets_np.T
    custom_cmap = plt.cm.get_cmap("viridis", max_fret_display + 1)

    img = ax_obj.pcolormesh(
        librosa.frames_to_time(
            np.arange(frets_to_plot.shape[1] + 1), sr=sr, hop_length=hop_length
        ),
        np.arange(num_strings + 1),
        frets_to_plot,
        cmap=custom_cmap,
        vmin=0,
        vmax=max_fret_display,
        shading='auto'
    )
    ax_obj.set_yticks(np.arange(num_strings) + 0.5)
    ax_obj.set_yticklabels([f"Str {i}" for i in range(num_strings)])
    ax_obj.set_ylabel("Struna")
    ax_obj.set_xlabel("Czas (s)")
    ax_obj.set_title(title)

    cbar = fig_obj.colorbar(
        img,
        ax=ax_obj,
        ticks=np.arange(0, max_fret_display + 1, max(1, (max_fret_display + 1) // 10)),
    )
    cbar.set_label(f"Numer progu (cisza >= {max_fret_display})")
    return fig_obj, ax_obj


def plot_sample_data_summary(
        features_tensor,
        onset_labels_tensor,
        fret_labels_tensor,
        sampling_rate,
        hop_len,
        track_id_display=None,
        max_frets_visual=config.MAX_FRETS,
        num_strings_visual=config.DEFAULT_NUM_STRINGS
):
    fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    main_title = "Wizualizacja Próbki Danych (Ground Truth)"
    if track_id_display:
        main_title = f"{main_title}\nUtwór: {track_id_display}"
    fig.suptitle(main_title, fontsize=16)

    plot_spectrogram(
        features_tensor,
        sr=sampling_rate,
        hop_length=hop_len,
        ax=axs[0]
    )
    plot_onset_labels(
        onset_labels_tensor,
        sr=sampling_rate,
        hop_length=hop_len,
        num_strings=num_strings_visual,
        ax=axs[1]
    )
    plot_fret_labels(
        fret_labels_tensor,
        sr=sampling_rate,
        hop_length=hop_len,
        num_strings=num_strings_visual,
        max_fret_display=(max_frets_visual + config.FRET_SILENCE_CLASS_OFFSET),
        ax=axs[2]
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)


def plot_predictions_vs_ground_truth(
        features_sample,
        onset_gt_sample,
        fret_gt_sample,
        onset_pred_logits_sample,
        fret_pred_logits_sample,
        sr,
        hop_length,
        max_frets_val,
        onset_threshold_val=config.DEFAULT_ONSET_THRESHOLD,
        num_strings_val=config.DEFAULT_NUM_STRINGS,
        figure_size=(15, 12),
        base_track_id=None,
        output_save_path=None,
):
    onset_pred_probs = torch.sigmoid(onset_pred_logits_sample)
    onset_pred_binary = (onset_pred_probs > onset_threshold_val).float()
    fret_pred_indices = torch.argmax(fret_pred_logits_sample, dim=-1)

    fig_obj, axs_array = plt.subplots(5, 1, figsize=figure_size, sharex=True)
    title_str = "Ground Truth vs Predykcje"
    if base_track_id:
        title_str = f"{title_str} (Utwór: {base_track_id})"
    fig_obj.suptitle(title_str, fontsize=16)

    max_fret_to_display = max_frets_val + config.FRET_SILENCE_CLASS_OFFSET

    plot_spectrogram(features_sample, sr=sr, hop_length=hop_length, ax=axs_array[0], title="Mel-Spektrogram")
    plot_onset_labels(onset_gt_sample, sr=sr, hop_length=hop_length, num_strings=num_strings_val, ax=axs_array[1],
                      title="GT Onsety")
    plot_onset_labels(onset_pred_binary, sr=sr, hop_length=hop_length, num_strings=num_strings_val, ax=axs_array[2],
                      title=f"Predykowane Onsety (próg={onset_threshold_val:.2f})")
    plot_fret_labels(fret_gt_sample, sr=sr, hop_length=hop_length, num_strings=num_strings_val,
                     max_fret_display=max_fret_to_display, ax=axs_array[3], title="GT Progi")
    plot_fret_labels(fret_pred_indices, sr=sr, hop_length=hop_length, num_strings=num_strings_val,
                     max_fret_display=max_fret_to_display, ax=axs_array[4], title="Predykowane Progi")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_save_path:
        try:
            plt.savefig(output_save_path)
            print(f"Zapisano wykres predykcji do: {output_save_path}")
        except Exception as e:
            print(f"  Nie udało się zapisać wykresu do {output_save_path}: {e}")
    plt.close(fig_obj)


def plot_training_history(
        history_data, output_save_path=None, figure_size=(20, 28)
):
    if not history_data or not history_data.get("train_total_loss"):
        print("Historia treningu jest pusta lub niekompletna. Wykres nie zostanie wygenerowany.")
        return

    num_epochs_completed = len(history_data["train_total_loss"])
    if num_epochs_completed == 0:
        print("Brak danych w historii treningu do narysowania.")
        return

    epochs_range_list = range(1, num_epochs_completed + 1)

    fig_obj, axs_array = plt.subplots(6, 2, figsize=figure_size)
    fig_obj.suptitle("Historia Treningu Modelu", fontsize=18)

    ax = axs_array[0, 0]
    if history_data.get("train_total_loss"):
        ax.plot(epochs_range_list, history_data["train_total_loss"], "o-", label="Train Total Loss")
    if history_data.get("val_total_loss"):
        ax.plot(epochs_range_list, history_data["val_total_loss"], "o-", label="Val Total Loss")
    ax.set_title("Strata Całkowita")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Strata")
    ax.grid(True)
    ax.legend()

    ax = axs_array[0, 1]
    if history_data.get("train_onset_loss"):
        ax.plot(epochs_range_list, history_data["train_onset_loss"], "o-", label="Train Onset Loss")
    if history_data.get("val_onset_loss"):
        ax.plot(epochs_range_list, history_data["val_onset_loss"], "o-", label="Val Onset Loss")
    if history_data.get("train_fret_loss"):
        ax.plot(epochs_range_list, history_data["train_fret_loss"], "s-", label="Train Fret Loss", alpha=0.7)
    if history_data.get("val_fret_loss"):
        ax.plot(epochs_range_list, history_data["val_fret_loss"], "s-", label="Val Fret Loss", alpha=0.7)
    ax.set_title("Straty Komponentów")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Strata")
    ax.grid(True)
    ax.legend()

    ax = axs_array[1, 0]
    if history_data.get("val_onset_f1_mir_eval"):
        ax.plot(epochs_range_list, history_data["val_onset_f1_mir_eval"], "o-", label="Val Onset F1 (mir_eval)")
    if history_data.get("val_onset_precision_mir_eval"):
        ax.plot(epochs_range_list, history_data["val_onset_precision_mir_eval"], "s-", label="Val Onset P (mir_eval)",
                alpha=0.7)
    if history_data.get("val_onset_recall_mir_eval"):
        ax.plot(epochs_range_list, history_data["val_onset_recall_mir_eval"], "^-", label="Val Onset R (mir_eval)",
                alpha=0.7)
    ax.set_title("Metryki Onsetów Walidacyjne (mir_eval, 50ms)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Metryka")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axs_array[1, 1]
    if history_data.get("val_onset_f1_optimal_thresh_frame"):
        ax.plot(epochs_range_list, history_data["val_onset_f1_optimal_thresh_frame"], "o-",
                label="Val Onset F1 (Frame, OptTh)")
    if history_data.get("val_optimal_threshold_epoch_frame"):
        ax_twin = ax.twinx()
        ax_twin.plot(epochs_range_list, history_data["val_optimal_threshold_epoch_frame"], "x--",
                     label="Opt. Th (Frame)", color='gray', alpha=0.6)
        ax_twin.set_ylabel("Optymalny Próg Ramkowy")
        ax_twin.legend(loc='lower right')
    ax.set_title("Metryki Onsetów Walidacyjne (Ramkowe, OptTh)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Metryka")
    ax.grid(True);
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.05)

    ax = axs_array[2, 0]
    if history_data.get("val_ftab"):
        ax.plot(epochs_range_list, history_data["val_ftab"], "o-", label="Val FTab")
    ax.set_title("FTab Walidacyjny")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("FTab")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axs_array[2, 1]
    if history_data.get("val_tdr_recall"):
        ax.plot(epochs_range_list, history_data["val_tdr_recall"], "o-", label="Val TDR (Recall)")
    if history_data.get("val_tdr_precision"):
        ax.plot(epochs_range_list, history_data["val_tdr_precision"], "s-", label="Val TDR P", alpha=0.7)
    if history_data.get("val_tdr_f1"):
        ax.plot(epochs_range_list, history_data["val_tdr_f1"], "^-", label="Val TDR F1", alpha=0.7)
    ax.set_title("Metryki TDR Walidacyjne")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Metryka")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axs_array[3, 0]
    if history_data.get("fret_accuracy_active"):
        ax.plot(epochs_range_list, history_data["fret_accuracy_active"], "o-", label="Val Fret Acc (Active Frames)")
    if history_data.get("fret_accuracy_overall"):
        ax.plot(epochs_range_list, history_data["fret_accuracy_overall"], "s-", label="Val Fret Acc (Overall)",
                alpha=0.7)
    ax.set_title("Dokładność Progów Walidacyjna (Ramkowa)")
    ax.set_xlabel("Epoka");
    ax.set_ylabel("Dokładność");
    ax.grid(True);
    ax.legend();
    ax.set_ylim(0, 1.05)

    ax = axs_array[3, 1]
    if history_data.get("lr"):
        ax.plot(epochs_range_list, history_data["lr"], "o-", label="Learning Rate", color="purple")
        ax.set_yscale("log")
    ax.set_title("Współczynnik Uczenia (Learning Rate)")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("LR")
    ax.grid(True)
    ax.legend()

    axs_array[4, 0].axis('off')
    axs_array[4, 1].axis('off')
    axs_array[5, 0].axis('off')
    axs_array[5, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_save_path:
        try:
            plt.savefig(output_save_path)
            print(f"Zapisano wykres historii treningu do: {output_save_path}")
        except Exception as e:
            print(f"Nie udało się zapisać wykresu historii do {output_save_path}: {e}")
    plt.show()
    plt.close(fig_obj)