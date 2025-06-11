import os
import torch
import librosa
import numpy as np
import noisereduce as nr
from evaluation import tablature_export
from model import utils as model_utils


def predict_tablature_from_audio_file(model, audio_file_path, device, config_obj):
    try:
        audio, sr = librosa.load(audio_file_path, sr=config_obj.SAMPLE_RATE, mono=True)

        reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)

        cqt_spec = librosa.cqt(
            y=reduced_noise_audio,
            sr=config_obj.SAMPLE_RATE,
            hop_length=config_obj.HOP_LENGTH,
            fmin=config_obj.FMIN_CQT,
            n_bins=config_obj.N_BINS_CQT,
            bins_per_octave=config_obj.BINS_PER_OCTAVE_CQT,
        )
        log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features_tensor = (
            torch.tensor(log_cqt_spec, dtype=torch.float32).unsqueeze(0).to(device)
        )

        model.eval()
        with torch.no_grad():
            onset_logits, fret_logits = model(features_tensor)

        onset_probs = torch.sigmoid(onset_logits).squeeze(0).cpu()
        fret_indices = torch.argmax(fret_logits, dim=-1).squeeze(0).cpu()

        optimal_threshold = config_obj.DEFAULT_TDR_THRESHOLD

        tab_matrix = tablature_export._generate_tablature_matrix_slots(
            onset_data_frames=onset_probs,
            fret_data_frames=fret_indices,
            num_total_frames=onset_probs.shape[0],
            num_total_strings=config_obj.DEFAULT_NUM_STRINGS,
            max_fret_val=config_obj.MAX_FRETS,
            onset_threshold=optimal_threshold,
        )
        tab_text = tablature_export._format_tablature_matrix_to_text(
            tab_matrix_data_slots=tab_matrix,
            num_total_strings=config_obj.DEFAULT_NUM_STRINGS,
        )
        return tab_text

    except Exception as e:
        return f"Wystąpił błąd podczas przetwarzania pliku {os.path.basename(audio_file_path)}: {e}"


def run_inference_on_directory(
    model_class,
    model_init_params,
    model_path,
    audio_dir,
    output_dir,
    device,
    config_obj,
):
    loaded_model = model_utils.load_best_model(
        model_class, model_init_params, model_path, device
    )
    if loaded_model is None:
        print("Nie udało się załadować modelu. Przerwanie działania.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Rozpoczynanie predykcji dla plików w: {audio_dir}")
    print(f"Wynikowe tabulatury zostaną zapisane w: {output_dir}")

    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("Nie znaleziono żadnych plików .wav w podanym folderze.")
        return

    for wav_file in wav_files:
        audio_path = os.path.join(audio_dir, wav_file)
        print(f"\n--- Przetwarzanie: {wav_file} ---")

        predicted_tablature = predict_tablature_from_audio_file(
            loaded_model, audio_path, device, config_obj
        )

        output_filename = f"{os.path.splitext(wav_file)[0]}_predicted_tab.txt"
        output_filepath = os.path.join(output_dir, output_filename)

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"Tabulatura przewidziana przez model dla pliku: {wav_file}\n")
            f.write("=" * 50 + "\n\n")
            f.write(predicted_tablature)

        print(f"Zapisano tabulaturę do: {output_filepath}")

    print("\nZakończono przetwarzanie wszystkich plików.")
