import os
import mirdata
import torch
from evaluation import tablature_export, midi_export
from model.architecture import GuitarTabCRNN, TabCNN
from model.utils import load_best_model


def visualize_best_model_outputs(
    sorted_completed_runs_list,
    main_search_artifacts_dir_path,
    config_obj,
    test_dataset_instance,
    current_device,
    data_home_path
):
    print("\n\n--- Rozpoczynanie Wizualizacji Wyników dla Najlepszego Modelu ---")

    if not test_dataset_instance or len(test_dataset_instance) == 0:
        print("Zbiór testowy (test_dataset) nie jest dostępny lub jest pusty. Pomijam wizualizację.")
        return

    if not sorted_completed_runs_list:
        print("Brak ukończonych przebiegów w `sorted_completed_runs_list`. Nie można wybrać najlepszego modelu.")
        return

    best_run_summary = sorted_completed_runs_list[0]
    best_run_folder_name = best_run_summary.get("run_folder_name")
    best_run_params_combo = best_run_summary.get("params_combo")
    best_run_optimal_threshold = best_run_summary.get(
        "optimal_threshold_at_best_val_metric_frame", config_obj.DEFAULT_ONSET_THRESHOLD
    )

    if not best_run_folder_name or not best_run_params_combo:
        print("Nie można odczytać informacji o folderze lub parametrach najlepszego przebiegu. Pomijam wizualizację.")
        return

    print(f"Wybrany najlepszy przebieg: {best_run_folder_name}")
    print(f"  Używane hiperparametry: {best_run_params_combo}")
    print(f"  Optymalny próg onsetów (ramkowy) z walidacji: {best_run_optimal_threshold:.2f}")

    try:
        temp_cnn_model = TabCNN(
            input_channels=config_obj.CNN_INPUT_CHANNELS,
            output_channels_list=config_obj.CNN_OUTPUT_CHANNELS_LIST_DEFAULT,
            kernel_sizes=config_obj.CNN_KERNEL_SIZES_DEFAULT,
            strides=config_obj.CNN_STRIDES_DEFAULT,
            paddings=config_obj.CNN_PADDINGS_DEFAULT,
            pooling_kernels=config_obj.CNN_POOLING_KERNELS_DEFAULT,
            pooling_strides=config_obj.CNN_POOLING_STRIDES_DEFAULT
        )
        with torch.no_grad():
            dummy_cnn_input = torch.randn(1, config_obj.CNN_INPUT_CHANNELS, config_obj.N_MELS, 32)
            dummy_cnn_output = temp_cnn_model(dummy_cnn_input)
            calculated_cnn_out_dim_best_model = temp_cnn_model.output_channels * dummy_cnn_output.shape[2]
        del temp_cnn_model, dummy_cnn_input, dummy_cnn_output
    except Exception as e_cnn_viz:
        print(f"Błąd podczas obliczania wymiaru CNN dla najlepszego modelu: {e_cnn_viz}. Pomijam wizualizację.")
        return

    model_init_params_for_load = {
        'num_frames_rnn_input_dim': calculated_cnn_out_dim_best_model,
        'rnn_hidden_size': best_run_params_combo['RNN_HIDDEN_SIZE'],
        'rnn_layers': best_run_params_combo['RNN_LAYERS'],
        'rnn_dropout': best_run_params_combo['RNN_DROPOUT'],
        'num_strings': config_obj.DEFAULT_NUM_STRINGS,
        'max_frets_val': config_obj.MAX_FRETS,
        'cnn_input_channels': config_obj.CNN_INPUT_CHANNELS,
        'cnn_output_channels_list': config_obj.CNN_OUTPUT_CHANNELS_LIST_DEFAULT,
        'cnn_kernel_sizes': config_obj.CNN_KERNEL_SIZES_DEFAULT,
        'cnn_strides': config_obj.CNN_STRIDES_DEFAULT,
        'cnn_paddings': config_obj.CNN_PADDINGS_DEFAULT,
        'cnn_pooling_kernels': config_obj.CNN_POOLING_KERNELS_DEFAULT,
        'cnn_pooling_strides': config_obj.CNN_POOLING_STRIDES_DEFAULT
    }
    path_to_best_model_weights = os.path.join(main_search_artifacts_dir_path, best_run_folder_name, "best_model.pth")

    loaded_model = load_best_model(
        model_class=GuitarTabCRNN,
        model_init_params=model_init_params_for_load,
        model_path=path_to_best_model_weights,
        device=current_device
    )

    if not loaded_model:
        print(f"Nie udało się załadować modelu z pliku: {path_to_best_model_weights}. Przerywam wizualizację.")
        return

    print(f"Pomyślnie załadowano wagi najlepszego modelu z: {path_to_best_model_weights}")

    visualization_output_dir = os.path.join(main_search_artifacts_dir_path, best_run_folder_name, "post_search_visualizations")
    os.makedirs(visualization_output_dir, exist_ok=True)
    print(f"Wizualizacje zostaną zapisane w: {visualization_output_dir}")

    num_samples_to_visualize = min(config_obj.NUM_SAMPLES_TO_VISUALIZE_FROM_TEST, len(test_dataset_instance))
    sample_indices_for_viz = list(range(num_samples_to_visualize))

    if not sample_indices_for_viz:
        print("Brak próbek w zbiorze testowym do wizualizacji.")
        return

    print(f"Generowanie wizualizacji dla {len(sample_indices_for_viz)} próbek: {sample_indices_for_viz}")

    guitarset_mirdata_loader = None
    if data_home_path and os.path.exists(data_home_path):
        try:
            guitarset_mirdata_loader = mirdata.initialize('guitarset', data_home=data_home_path)
        except Exception as e_mir:
            print(f"Ostrzeżenie: Nie udało się zainicjalizować mirdata dla GuitarSet: {e_mir}. Pliki WAV mogą nie zostać skopiowane.")
    else:
        print(f"Ostrzeżenie: Katalog DATA_HOME ('{data_home_path}') nie istnieje lub nie został podany. Pliki WAV nie zostaną skopiowane.")

    midi_export.generate_midi_from_predictions(
        model_to_eval=loaded_model,
        dataset_instance=test_dataset_instance,
        device_to_use=current_device,
        sample_indices_list=sample_indices_for_viz,
        onset_threshold_optimal=best_run_optimal_threshold,
        sampling_rate=config_obj.SAMPLE_RATE,
        hop_len=config_obj.HOP_LENGTH,
        max_fret_value=config_obj.MAX_FRETS,
        midi_output_directory=visualization_output_dir,
        guitarset_data_home=data_home_path,
        guitarset_loader=guitarset_mirdata_loader
    )
    print(f"Zakończono generowanie plików MIDI.")

    tablature_export.generate_text_tablature_comparison(
        model_to_eval=loaded_model,
        dataset_instance=test_dataset_instance,
        device_to_use=current_device,
        sample_indices_list=sample_indices_for_viz,
        onset_threshold_optimal=best_run_optimal_threshold,
        max_fret_val=config_obj.MAX_FRETS,
        output_directory_path=visualization_output_dir
    )
    print(f"Zakończono generowanie plików tabulatur tekstowych i podsumowania dopasowania.")

    print("\n--- Wizualizacje (pliki MIDI i TXT) zostały zapisane. Podsumowanie dopasowania tabulatur wyświetlono powyżej. ---")
    print(f"--- Sprawdź katalog: {visualization_output_dir} ---")
    print("\n--- Zakończono Wizualizację Wyników ---")