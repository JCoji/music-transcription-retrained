import numpy as np
import torch
import os
import time
import json  # Dodano import json

from torch import optim
from tqdm import tqdm
import config  # Zakładamy, że config zawiera DATASET_TRAIN_AUGMENTATION_PARAMS
from evaluation import performance_metrics
from model import architecture, utils
from . import epoch_processing, loss_functions
from vizualization import plotting


def run_training_loop(
    model_instance,
    device_to_use,
    train_loader,
    validation_loader,
    optimizer_instance,
    scheduler_instance,
    criterion_instance_combined,
    run_training_config,  # Ten słownik powinien zawierać też parametry augmentacji
    audio_sr_val,
    audio_hop_len_val,
):
    num_epochs_total = run_training_config.get("NUM_EPOCHS", config.NUM_EPOCHS_DEFAULT)
    fret_classes_count = run_training_config.get(
        "FRET_NUM_CLASSES", config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET + 1
    )
    artifacts_output_dir = run_training_config.get(
        "ARTIFACTS_DIR", "training_artifacts"
    )
    training_log_filepath = run_training_config.get(
        "LOG_FILE_PATH", os.path.join(artifacts_output_dir, "training_log.txt")
    )
    current_batch_size = run_training_config.get(
        "BATCH_SIZE", config.BATCH_SIZE_DEFAULT
    )
    clear_console_output_every_n = run_training_config.get(
        "CLEAR_CONSOLE_EVERY_N_RUNS", 0
    )
    clear_output_function_ref = run_training_config.get("CLEAR_OUTPUT_FUNC", None)

    fixed_thresh_for_log = run_training_config.get(
        "ONSET_PREDICTION_THRESHOLD", config.DEFAULT_ONSET_THRESHOLD
    )
    early_stop_patience_val = run_training_config.get(
        "EARLY_STOPPING_PATIENCE", config.EARLY_STOPPING_PATIENCE_DEFAULT
    )
    checkpoint_metric_to_track = run_training_config.get(
        "CHECKPOINT_METRIC", config.CHECKPOINT_METRIC_DEFAULT
    )

    best_tracked_metric_val = -float("inf")
    if "loss" in checkpoint_metric_to_track.lower():
        best_tracked_metric_val = float("inf")
    epochs_without_improvement = 0

    training_history = {
        "train_total_loss": [],
        "train_onset_loss": [],
        "train_fret_loss": [],
        "val_total_loss": [],
        "val_onset_loss": [],
        "val_fret_loss": [],
        "onset_f1_at_fixed_thresh_frame": [],
        "onset_precision_at_fixed_thresh_frame": [],
        "onset_recall_at_fixed_thresh_frame": [],
        "onset_accuracy_at_fixed_thresh_frame": [],
        "val_onset_f1_optimal_thresh_frame": [],
        "val_onset_precision_optimal_thresh_frame": [],
        "val_onset_recall_optimal_thresh_frame": [],
        "val_onset_accuracy_optimal_thresh_frame": [],
        "val_optimal_threshold_epoch_frame": [],
        "val_ftab": [],
        "val_onset_precision_mir_eval": [],
        "val_onset_recall_mir_eval": [],
        "val_onset_f1_mir_eval": [],
        "val_tdr_precision": [],
        "val_tdr_recall": [],
        "val_tdr_f1": [],
        "lr": [],
    }

    os.makedirs(artifacts_output_dir, exist_ok=True)

    with open(training_log_filepath, "a", encoding="utf-8") as log_file_handle:
        log_file_handle.write(
            f"\n--- Rozpoczęcie Nowej Sesji Treningowej: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        log_file_handle.write(
            f"--- Konfiguracja Treningu (tracking: {checkpoint_metric_to_track}, Batch Size: {current_batch_size}) ---\n"
        )

        # Zapisywanie parametrów modelu i treningu (z hyperparams_combo)
        log_file_handle.write("  --- Parametry Modelu/Treningu ---\n")
        for conf_key, conf_value in run_training_config.items():
            # Filtrujemy klucze, aby nie zapisywać wewnętrznych parametrów konfiguracyjnych pętli treningowej
            # ani parametrów augmentacji (zostaną zapisane osobno)
            if (
                conf_key
                not in [
                    "NUM_EPOCHS",
                    "FRET_NUM_CLASSES",
                    "ARTIFACTS_DIR",
                    "LOG_FILE_PATH",
                    "CLEAR_CONSOLE_EVERY_N_RUNS",
                    "CLEAR_OUTPUT_FUNC",
                    "ONSET_PREDICTION_THRESHOLD",
                    "MAX_FRETS",
                    "EARLY_STOPPING_PATIENCE",
                    "CHECKPOINT_METRIC",
                    "RUN_DESCRIPTION",
                ]
                and not conf_key.startswith("aug_")
                and not conf_key.startswith("specaug_")
                and conf_key != "enable_audio_augmentations"
                and conf_key != "enable_specaugment"
            ):
                log_line = f"    {conf_key}: {conf_value}\n"
                log_file_handle.write(log_line)

        # Zapisywanie parametrów augmentacji (tych przekazanych w run_training_config)
        log_file_handle.write("  --- Parametry Augmentacji ---\n")
        augmentation_params_to_log = {
            k: v
            for k, v in run_training_config.items()
            if k.startswith("aug_")
            or k.startswith("specaug_")
            or k == "enable_audio_augmentations"
            or k == "enable_specaugment"
        }
        if not augmentation_params_to_log and hasattr(
            config, "DATASET_TRAIN_AUGMENTATION_PARAMS"
        ):
            # Jeśli nie przekazano w run_training_config, użyj domyślnych z config.py
            augmentation_params_to_log = config.DATASET_TRAIN_AUGMENTATION_PARAMS

        for aug_key, aug_value in augmentation_params_to_log.items():
            log_file_handle.write(f"    {aug_key}: {aug_value}\n")

        log_file_handle.write("-" * 30 + "\n\n")

        print(
            f"\nRozpoczynanie pętli treningowej na {num_epochs_total} epok. Śledzona metryka: {checkpoint_metric_to_track}. Batch Size: {current_batch_size}"
        )

        # ... (reszta pętli treningowej bez zmian) ...
        for current_epoch_num in range(num_epochs_total):
            epoch_description_str = f"Epoka {current_epoch_num + 1}/{num_epochs_total}"

            if (
                clear_output_function_ref
                and clear_console_output_every_n > 0
                and current_epoch_num > 0
                and (current_epoch_num) % clear_console_output_every_n == 0
            ):
                clear_output_function_ref(wait=True)
                print(
                    f"Output wyczyszczony. Kontynuacja treningu ({epoch_description_str})..."
                )

            train_pbar = tqdm(
                train_loader,
                desc=f"{epoch_description_str} [Trening]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            train_epoch_metrics = epoch_processing.train_one_epoch(
                model_instance,
                train_pbar,
                optimizer_instance,
                criterion_instance_combined,
                device_to_use,
            )
            train_pbar.close()
            training_history["train_total_loss"].append(
                train_epoch_metrics["train_total_loss"]
            )
            training_history["train_onset_loss"].append(
                train_epoch_metrics["train_onset_loss"]
            )
            training_history["train_fret_loss"].append(
                train_epoch_metrics["train_fret_loss"]
            )

            val_pbar = tqdm(
                validation_loader,
                desc=f"{epoch_description_str} [Walidacja]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            val_epoch_all_metrics, _, _ = epoch_processing.evaluate_one_epoch(
                model_instance,
                val_pbar,
                criterion_instance_combined,
                device_to_use,
                fret_classes_count,
                audio_sr_val=audio_sr_val,
                audio_hop_len_val=audio_hop_len_val,
                log_fixed_onset_threshold=fixed_thresh_for_log,
            )
            val_pbar.close()

            for key, value in val_epoch_all_metrics.items():
                if key in training_history:
                    training_history[key].append(value)
                else:
                    training_history[key] = [value]

            if (
                "val_optimal_threshold_epoch_frame" not in val_epoch_all_metrics
                and "val_optimal_threshold_epoch_frame" in training_history
            ):
                training_history["val_optimal_threshold_epoch_frame"].append(
                    config.DEFAULT_ONSET_THRESHOLD
                )

            current_learning_rate = optimizer_instance.param_groups[0]["lr"]
            training_history["lr"].append(current_learning_rate)

            log_lines_for_file = [
                f"--- {epoch_description_str} ---",
                f"  LR: {current_learning_rate:.2e}",
                f"  Train Loss: {train_epoch_metrics['train_total_loss']:.4f} (O: {train_epoch_metrics['train_onset_loss']:.4f}, F: {train_epoch_metrics['train_fret_loss']:.4f})",
                f"  Val Loss: {val_epoch_all_metrics.get('val_total_loss', 0.0):.4f} (O: {val_epoch_all_metrics.get('val_onset_loss', 0.0):.4f}, F: {val_epoch_all_metrics.get('val_fret_loss', 0.0):.4f})",
                f"  Val Onset (mir_eval, Th={fixed_thresh_for_log:.2f}): P: {val_epoch_all_metrics.get('val_onset_precision_mir_eval', 0.0):.4f} R: {val_epoch_all_metrics.get('val_onset_recall_mir_eval', 0.0):.4f} F1: {val_epoch_all_metrics.get('val_onset_f1_mir_eval', 0.0):.4f}",
                f"  Val FTab (Th={fixed_thresh_for_log:.2f}): {val_epoch_all_metrics.get('val_ftab', 0.0):.4f}",
                f"  Val TDR (Th={fixed_thresh_for_log:.2f}): P: {val_epoch_all_metrics.get('val_tdr_precision', 0.0):.4f} R (TDR): {val_epoch_all_metrics.get('val_tdr_recall', 0.0):.4f} F1: {val_epoch_all_metrics.get('val_tdr_f1', 0.0):.4f}",
                f"  Val Onset (Frame, OptTh={val_epoch_all_metrics.get('val_optimal_threshold_epoch_frame', 0.0):.2f}): F1: {val_epoch_all_metrics.get('val_onset_f1_optimal_thresh_frame', 0.0):.4f} (P: {val_epoch_all_metrics.get('val_onset_precision_optimal_thresh_frame', 0.0):.4f} R: {val_epoch_all_metrics.get('val_onset_recall_optimal_thresh_frame', 0.0):.4f})",
            ]
            log_file_handle.write("\n".join(log_lines_for_file) + "\n")
            log_file_handle.flush()

            print(f"\n--- {epoch_description_str} ---")
            for line in log_lines_for_file[1:]:
                print(line)

            metric_value_for_checkpoint = val_epoch_all_metrics.get(
                checkpoint_metric_to_track,
                (
                    -float("inf")
                    if "loss" not in checkpoint_metric_to_track
                    else float("inf")
                ),
            )

            if scheduler_instance:
                if isinstance(
                    scheduler_instance, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler_instance.step(metric_value_for_checkpoint)
                else:
                    scheduler_instance.step()

            is_improved = False
            if "loss" in checkpoint_metric_to_track.lower():
                if metric_value_for_checkpoint < best_tracked_metric_val:
                    best_tracked_metric_val = metric_value_for_checkpoint
                    is_improved = True
            else:
                if metric_value_for_checkpoint > best_tracked_metric_val:
                    best_tracked_metric_val = metric_value_for_checkpoint
                    is_improved = True

            if is_improved:
                epochs_without_improvement = 0
                model_checkpoint_path = os.path.join(
                    artifacts_output_dir, "best_model.pth"
                )
                torch.save(model_instance.state_dict(), model_checkpoint_path)
                improvement_log_msg = f"    -> Zapisano nowy najlepszy model ({checkpoint_metric_to_track}: {best_tracked_metric_val:.4f})"
                print(improvement_log_msg)
                log_file_handle.write(improvement_log_msg + "\n")
            elif (
                early_stop_patience_val is not None and current_epoch_num > 0
            ):  # Dodano warunek current_epoch_num > 0
                epochs_without_improvement += 1
                no_improvement_log_msg = f"    Brak poprawy {checkpoint_metric_to_track} od {epochs_without_improvement} epok ({metric_value_for_checkpoint:.4f} vs Best: {best_tracked_metric_val:.4f})"
                if (
                    epochs_without_improvement == 1
                    or epochs_without_improvement % max(1, early_stop_patience_val // 3)
                    == 0
                ):
                    print(no_improvement_log_msg)
                log_file_handle.write(no_improvement_log_msg + "\n")

                if epochs_without_improvement >= early_stop_patience_val:
                    early_stop_msg = f"    Wczesne zatrzymanie treningu po {epochs_without_improvement} epokach bez poprawy dla '{checkpoint_metric_to_track}'."
                    print(early_stop_msg)
                    log_file_handle.write(early_stop_msg + "\n\n")
                    break
            log_file_handle.write("-" * 80 + "\n")

        log_file_handle.write(
            f"\n--- Koniec Sesji Treningowej: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n"
        )

    print("\nZakończono pętlę treningową.")

    if training_history:
        history_plot_path = os.path.join(
            artifacts_output_dir, "training_history_summary.png"
        )
        if hasattr(plotting, "plot_training_history"):
            print(f"Generowanie podsumowania historii treningu do: {history_plot_path}")
            plotting.plot_training_history(
                training_history, output_save_path=history_plot_path
            )
        else:
            print(
                "Funkcja plot_training_history nie jest dostępna w vizualization.plotting."
            )

    return training_history


def process_single_hyperparameter_run(
    run_id,
    hyperparams_combo,  # Ten słownik zawiera parametry modelu/treningu
    current_augmentation_params,  # NOWY PARAMETR: słownik z parametrami augmentacji
    config_obj,
    main_artifacts_dir,
    train_loader,
    validation_loader,
    test_loader,
    jupyter_notebook_clear_output_func,
):
    run_start_time = time.time()

    run_description_str = hyperparams_combo.get("run_description", f"run_{run_id}")
    # Dodajemy informację o augmentacji do opisu folderu, jeśli chcemy rozróżniać
    # Można to zrobić bardziej elegancko, np. tworząc hash z parametrów augmentacji
    # Tutaj proste sprawdzenie, czy augmentacje są w ogóle włączone
    aug_suffix = (
        "_augEnabled"
        if current_augmentation_params.get("enable_audio_augmentations", False)
        or current_augmentation_params.get("enable_specaugment", False)
        else "_augDisabled"
    )

    # Sprawdzamy, czy run_description_str już zawiera informacje o BS
    # Jeśli nie, a current_augmentation_params ma BATCH_SIZE, dodajemy go.
    # To jest bardziej dla nazwy folderu, bo BATCH_SIZE jest też w hyperparams_combo
    batch_size_for_desc = hyperparams_combo.get("BATCH_SIZE", "BS_unknown")
    if (
        f"_BS{batch_size_for_desc}" not in run_description_str
        and f"BATCH{batch_size_for_desc}" not in run_description_str
    ):
        run_description_str_with_bs = f"{run_description_str}_BS{batch_size_for_desc}"
    else:
        run_description_str_with_bs = run_description_str

    current_run_folder_name_sanitized = f"run_{run_id}_{run_description_str_with_bs.replace(' ', '_').replace('/', '-').replace(':', '').replace(',', '')[:50]}{aug_suffix}"

    current_run_artifacts_dir = os.path.join(
        main_artifacts_dir, current_run_folder_name_sanitized
    )
    current_run_log_file = os.path.join(current_run_artifacts_dir, "training_log.txt")

    os.makedirs(current_run_artifacts_dir, exist_ok=True)

    print(
        f"--- Konfiguracja dla przebiegu {run_id} ({current_run_folder_name_sanitized}) ---"
    )
    print(f"  Katalog artefaktów: {current_run_artifacts_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calculated_cnn_out_dim = None
    try:
        temp_cnn_model = architecture.TabCNN(
            input_channels=config_obj.CNN_INPUT_CHANNELS,
            output_channels_list=config_obj.CNN_OUTPUT_CHANNELS_LIST_DEFAULT,
            kernel_sizes=config_obj.CNN_KERNEL_SIZES_DEFAULT,
            strides=config_obj.CNN_STRIDES_DEFAULT,
            paddings=config_obj.CNN_PADDINGS_DEFAULT,
            pooling_kernels=config_obj.CNN_POOLING_KERNELS_DEFAULT,
            pooling_strides=config_obj.CNN_POOLING_STRIDES_DEFAULT,
        )
        with torch.no_grad():
            dummy_cnn_input = torch.randn(
                1, config_obj.CNN_INPUT_CHANNELS, config_obj.N_MELS, 32
            )
            dummy_cnn_output = temp_cnn_model(dummy_cnn_input)
            calculated_cnn_out_dim = (
                temp_cnn_model.output_channels * dummy_cnn_output.shape[2]
            )
        del temp_cnn_model, dummy_cnn_input, dummy_cnn_output
    except Exception as e_cnn:
        print(f"Błąd podczas obliczania wymiaru z CNN dla przebiegu {run_id}: {e_cnn}")
        error_summary = {
            "run_index": run_id,
            "params_combo": hyperparams_combo,
            "augmentation_params": current_augmentation_params,  # Dodano
            "status": "CNN_DIM_ERROR",
            "error": str(e_cnn),
            "run_folder_name": current_run_folder_name_sanitized,
        }
        return error_summary

    if calculated_cnn_out_dim is None:
        return {
            "run_index": run_id,
            "params_combo": hyperparams_combo,
            "augmentation_params": current_augmentation_params,  # Dodano
            "status": "CNN_DIM_CALC_FAILED",
            "error": "calculated_cnn_out_dim is None",
            "run_folder_name": current_run_folder_name_sanitized,
        }

    current_model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=calculated_cnn_out_dim,
        rnn_hidden_size=hyperparams_combo["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams_combo["RNN_LAYERS"],
        rnn_dropout=hyperparams_combo["RNN_DROPOUT"],
        num_strings=config_obj.DEFAULT_NUM_STRINGS,
        max_frets_val=config_obj.MAX_FRETS,
        cnn_input_channels=config_obj.CNN_INPUT_CHANNELS,
        cnn_output_channels_list=config_obj.CNN_OUTPUT_CHANNELS_LIST_DEFAULT,
        cnn_kernel_sizes=config_obj.CNN_KERNEL_SIZES_DEFAULT,
        cnn_strides=config_obj.CNN_STRIDES_DEFAULT,
        cnn_paddings=config_obj.CNN_PADDINGS_DEFAULT,
        cnn_pooling_kernels=config_obj.CNN_POOLING_KERNELS_DEFAULT,
        cnn_pooling_strides=config_obj.CNN_POOLING_STRIDES_DEFAULT,
    )
    current_model.to(device)

    onset_pos_weight_tensor = (
        torch.tensor(
            [hyperparams_combo["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device
        )
        if hyperparams_combo.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0
        else None
    )  # Dodano .get()

    combined_loss_criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pos_weight_tensor,
        fret_ignore_index=config_obj.FRET_PADDING_VALUE,
        onset_loss_weight=hyperparams_combo["ONSET_LOSS_WEIGHT"],
        fret_loss_weight=config_obj.FRET_LOSS_WEIGHT_DEFAULT,
    ).to(device)

    optimizer = optim.AdamW(
        current_model.parameters(),
        lr=hyperparams_combo["LEARNING_RATE_INIT"],
        weight_decay=hyperparams_combo["WEIGHT_DECAY"],
    )

    scheduler_mode = (
        "min" if "loss" in config_obj.CHECKPOINT_METRIC_DEFAULT.lower() else "max"
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=hyperparams_combo["SCHEDULER_FACTOR"],
        patience=hyperparams_combo["SCHEDULER_PATIENCE"],
    )

    # Połączenie hiperparametrów modelu/treningu z parametrami augmentacji
    training_run_configuration = {
        "NUM_EPOCHS": config_obj.NUM_EPOCHS_DEFAULT,
        "FRET_NUM_CLASSES": config_obj.MAX_FRETS
        + config_obj.FRET_SILENCE_CLASS_OFFSET
        + 1,
        "ARTIFACTS_DIR": current_run_artifacts_dir,
        "LOG_FILE_PATH": current_run_log_file,
        "CLEAR_CONSOLE_EVERY_N_RUNS": config_obj.CLEAR_CONSOLE_EVERY_N_RUNS,
        "CLEAR_OUTPUT_FUNC": jupyter_notebook_clear_output_func,
        "ONSET_PREDICTION_THRESHOLD": config_obj.DEFAULT_ONSET_THRESHOLD,
        "MAX_FRETS": config_obj.MAX_FRETS,
        "EARLY_STOPPING_PATIENCE": config_obj.EARLY_STOPPING_PATIENCE_DEFAULT,
        "CHECKPOINT_METRIC": config_obj.CHECKPOINT_METRIC_DEFAULT,
        "RUN_DESCRIPTION": run_description_str,
        **hyperparams_combo,  # Parametry modelu/treningu
        **current_augmentation_params,  # Dodane parametry augmentacji
    }
    # Usunięcie duplikatu klucza BATCH_SIZE, jeśli current_augmentation_params go zawiera,
    # a hyperparams_combo również (co jest bardziej prawdopodobne)
    if (
        "BATCH_SIZE" in hyperparams_combo
        and "BATCH_SIZE" in current_augmentation_params
    ):
        # Dajemy priorytet wartości z hyperparams_combo (zwykle bardziej specyficzne dla przebiegu)
        # lub usuwamy z current_augmentation_params przed rozpakowaniem, aby uniknąć konfliktu
        # W praktyce, BATCH_SIZE powinien być tylko w jednym miejscu (np. hyperparams_combo)
        # lub kontrolowany globalnie.
        pass  # Zakładamy, że BATCH_SIZE jest w hyperparams_combo lub globalnie, a nie w current_augmentation_params

    print(
        f"Rozpoczynanie treningu dla przebiegu {run_id}. Logi w: {current_run_log_file}"
    )
    # Zapisanie pełnej konfiguracji przebiegu (w tym augmentacji) do pliku JSON w folderze przebiegu
    # dla łatwiejszej reprodukcji i analizy
    run_config_save_path = os.path.join(
        current_run_artifacts_dir, "run_configuration.json"
    )
    try:
        with open(run_config_save_path, "w", encoding="utf-8") as f_conf:
            # Konwertujemy tensory na listy, jeśli są w current_augmentation_params (np. limity)
            serializable_aug_params = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in current_augmentation_params.items()
            }
            full_config_to_save = {
                **hyperparams_combo,
                "augmentation_params": serializable_aug_params,
            }
            json.dump(full_config_to_save, f_conf, indent=4)
        print(f"Zapisano pełną konfigurację przebiegu do: {run_config_save_path}")
    except Exception as e_json_save:
        print(
            f"Ostrzeżenie: Nie udało się zapisać pliku run_configuration.json: {e_json_save}"
        )

    try:
        training_run_history = run_training_loop(
            model_instance=current_model,
            device_to_use=device,
            train_loader=train_loader,  # train_loader powinien być już skonfigurowany z odpowiednimi aug params
            validation_loader=validation_loader,
            optimizer_instance=optimizer,
            scheduler_instance=scheduler,
            criterion_instance_combined=combined_loss_criterion,
            run_training_config=training_run_configuration,  # Przekazujemy pełną konfigurację
            audio_sr_val=config_obj.SAMPLE_RATE,
            audio_hop_len_val=config_obj.HOP_LENGTH,
        )
    except Exception as e_train:
        print(f"BŁĄD KRYTYCZNY podczas treningu przebiegu {run_id}: {e_train}")
        import traceback

        traceback.print_exc()
        error_summary = {
            "run_index": run_id,
            "params_combo": hyperparams_combo,
            "augmentation_params": current_augmentation_params,  # Dodano
            "status": "TRAINING_ERROR",
            "error": str(e_train),
            "run_folder_name": current_run_folder_name_sanitized,
        }
        del current_model, optimizer, scheduler, combined_loss_criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return error_summary

    run_end_time_training = time.time()
    run_duration_training_minutes = (run_end_time_training - run_start_time) / 60
    print(
        f"Zakończono trening dla przebiegu {run_id}. Czas (trening): {run_duration_training_minutes:.2f} min."
    )

    best_val_metric_final = 0.0
    stopped_at_epoch = 0
    optimal_threshold_at_best_metric_frame = config_obj.DEFAULT_ONSET_THRESHOLD

    if training_run_history:
        stopped_at_epoch = len(training_run_history.get("train_total_loss", []))
        tracked_metric_history = training_run_history.get(
            config_obj.CHECKPOINT_METRIC_DEFAULT, []
        )
        if tracked_metric_history:  # Sprawdzenie, czy lista nie jest pusta
            # Upewnij się, że tracked_metric_history zawiera wartości liczbowe
            valid_metric_history = [
                m for m in tracked_metric_history if isinstance(m, (int, float))
            ]
            if valid_metric_history:
                if "loss" in config_obj.CHECKPOINT_METRIC_DEFAULT.lower():
                    best_val_metric_final = min(valid_metric_history)
                    best_metric_epoch_idx = valid_metric_history.index(
                        best_val_metric_final
                    )  # Użyj .index na odfiltrowanej liście
                else:
                    best_val_metric_final = max(valid_metric_history)
                    best_metric_epoch_idx = valid_metric_history.index(
                        best_val_metric_final
                    )

                # Sprawdź, czy val_optimal_threshold_epoch_frame ma wystarczająco dużo elementów
                if (
                    "val_optimal_threshold_epoch_frame" in training_run_history
                    and len(training_run_history["val_optimal_threshold_epoch_frame"])
                    > best_metric_epoch_idx
                ):
                    optimal_threshold_at_best_metric_frame = training_run_history[
                        "val_optimal_threshold_epoch_frame"
                    ][best_metric_epoch_idx]
                else:
                    print(
                        f"Ostrzeżenie: Brak wpisu dla optymalnego progu w epoce {best_metric_epoch_idx+1}. Używam domyślnego."
                    )
            else:
                print(
                    "Ostrzeżenie: Brak poprawnych wartości metryk w historii do wyznaczenia najlepszej."
                )

    final_test_metrics = "N/A"
    final_loaded_model = None

    if test_loader:
        best_model_path_for_run = os.path.join(
            current_run_artifacts_dir, "best_model.pth"
        )
        if os.path.exists(best_model_path_for_run):
            model_init_parameters = {
                "num_frames_rnn_input_dim": calculated_cnn_out_dim,
                "rnn_hidden_size": hyperparams_combo["RNN_HIDDEN_SIZE"],
                "rnn_layers": hyperparams_combo["RNN_LAYERS"],
                "rnn_dropout": hyperparams_combo["RNN_DROPOUT"],
                "num_strings": config_obj.DEFAULT_NUM_STRINGS,
                "max_frets_val": config_obj.MAX_FRETS,
                "cnn_input_channels": config_obj.CNN_INPUT_CHANNELS,
                "cnn_output_channels_list": config_obj.CNN_OUTPUT_CHANNELS_LIST_DEFAULT,
                "cnn_kernel_sizes": config_obj.CNN_KERNEL_SIZES_DEFAULT,
                "cnn_strides": config_obj.CNN_STRIDES_DEFAULT,
                "cnn_paddings": config_obj.CNN_PADDINGS_DEFAULT,
                "cnn_pooling_kernels": config_obj.CNN_POOLING_KERNELS_DEFAULT,
                "cnn_pooling_strides": config_obj.CNN_POOLING_STRIDES_DEFAULT,
            }
            final_loaded_model = utils.load_best_model(
                architecture.GuitarTabCRNN,
                model_init_parameters,
                best_model_path_for_run,
                device,
            )
            if final_loaded_model:
                print(
                    f"Ewaluacja na zbiorze testowym z progiem onsetów (ramkowym): {optimal_threshold_at_best_metric_frame:.2f}"
                )
                final_test_metrics = performance_metrics.evaluate_model_on_test_set(
                    model_to_eval=final_loaded_model,
                    test_dataloader=test_loader,
                    device_to_use=device,
                    fret_num_classes_val=config_obj.MAX_FRETS
                    + config_obj.FRET_SILENCE_CLASS_OFFSET
                    + 1,
                    optimal_onset_threshold=optimal_threshold_at_best_metric_frame,
                    audio_sr=config_obj.SAMPLE_RATE,
                    audio_hop_length=config_obj.HOP_LENGTH,
                )
            else:
                print(
                    f"Nie udało się załadować najlepszego modelu dla przebiegu {run_id}."
                )
        else:
            print(f"Nie znaleziono pliku best_model.pth dla przebiegu {run_id}.")
    else:
        print(
            "Test DataLoader nie jest dostępny. Pomijam ewaluację na zbiorze testowym."
        )

    run_end_time_total = time.time()
    run_duration_total_minutes = (run_end_time_total - run_start_time) / 60

    current_run_summary = {
        "run_index": run_id,
        "run_folder_name": current_run_folder_name_sanitized,
        "params_combo": hyperparams_combo,
        "augmentation_params": current_augmentation_params,  # Dodano parametry augmentacji
        f"best_{config_obj.CHECKPOINT_METRIC_DEFAULT}": best_val_metric_final,
        "optimal_threshold_at_best_val_metric_frame": optimal_threshold_at_best_metric_frame,
        "test_metrics": final_test_metrics,
        "run_duration_minutes": run_duration_total_minutes,
        "stopped_epoch": stopped_at_epoch,
        "status": "COMPLETED",
    }

    del current_model, optimizer, scheduler, combined_loss_criterion
    if training_run_history:
        del training_run_history
    if final_loaded_model:
        del final_loaded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return current_run_summary
