import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import argparse

# NOWE IMPORTY na potrzeby pre-processingu
import scipy.signal
import noisereduce as nr

# Import z plików projektu
import config
from model.utils import load_best_model
from model.architecture import GuitarTabCRNN
from training.note_conversion_utils import frames_to_notes_for_eval
from evaluation.tablature_export import save_notes_to_ascii_tab


# NOWA FUNKCJA: Potok pre-processingu audio
def preprocess_audio(y, sr, high_pass_cutoff=80.0, noise_reduce_duration=0.5):
    """
    Stosuje potok pre-processingu na surowym sygnale audio.
    1. Filtr górnoprzepustowy w celu usunięcia niskoczęstotliwościowego "błota".
    2. Redukcja szumu na podstawie próbki z początku nagrania.
    """
    try:
        # Krok 1: Filtr górnoprzepustowy (usuwa dudnienie i hałas sieciowy)
        # Projektujemy filtr Butterwortha 4. rzędu
        nyquist = 0.5 * sr
        cutoff_normalized = high_pass_cutoff / nyquist
        b, a = scipy.signal.butter(4, cutoff_normalized, btype="high", analog=False)
        y_filtered = scipy.signal.lfilter(b, a, y)

        # Krok 2: Redukcja szumu
        # Zakładamy, że pierwsza część nagrania to cisza/szum
        # i używamy jej jako profilu szumu.
        noise_clip = y_filtered[: int(sr * noise_reduce_duration)]
        y_reduced = nr.reduce_noise(
            y=y_filtered, y_noise=noise_clip, sr=sr, stationary=True
        )

        # Normalizacja głośności po przetworzeniu
        y_processed = librosa.util.normalize(y_reduced)

        return y_processed
    except Exception as e:
        print(f"\nBłąd podczas pre-processingu: {e}. Zwracanie oryginalnego audio.")
        return y  # W razie błędu, zwróć oryginalny sygnał


# ZMODYFIKOWANA FUNKCJA: Dodano flagę `apply_preprocessing`
def predict_on_single_file(
    model, audio_path, config_obj, device, onset_threshold, apply_preprocessing=False
):
    """
    Przetwarza pojedynczy plik audio i zwraca listę przewidzianych nut.
    """
    try:
        # 1. Wczytanie audio
        y, sr = librosa.load(audio_path, sr=config_obj.SAMPLE_RATE, mono=True)

        # 2. ZASTOSOWANIE PRE-PROCESSNGU (jeśli flaga jest aktywna)
        if apply_preprocessing:
            print(
                f"  -> Stosowanie pre-processingu dla {os.path.basename(audio_path)}..."
            )
            y = preprocess_audio(y, sr)

        # 3. Obliczenie spektrogramu CQT (teraz na przetworzonym audio)
        cqt_spec = librosa.cqt(
            y=y,
            sr=config_obj.SAMPLE_RATE,
            hop_length=config_obj.HOP_LENGTH,
            fmin=config_obj.FMIN_CQT,
            n_bins=config_obj.N_BINS_CQT,
            bins_per_octave=config_obj.BINS_PER_OCTAVE_CQT,
        )
        log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features = (
            torch.tensor(log_cqt_spec, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # ... reszta funkcji bez zmian ...
        model.eval()
        with torch.no_grad():
            onset_logits, fret_logits = model(features)

        onset_probs = torch.sigmoid(onset_logits).squeeze(0)
        fret_pred_indices = torch.argmax(fret_logits, dim=-1).squeeze(0)
        onset_preds_binary = (onset_probs > onset_threshold).float()

        predicted_notes = frames_to_notes_for_eval(
            onset_preds_binary_frames=onset_preds_binary.cpu(),
            fret_pred_indices_frames=fret_pred_indices.cpu(),
            frame_hop_length=config_obj.HOP_LENGTH,
            audio_sample_rate=config_obj.SAMPLE_RATE,
            max_fret_value=config_obj.MAX_FRETS,
            min_note_duration_frames=config_obj.MIN_NOTE_DURATION_FRAMES,
        )
        return predicted_notes

    except Exception as e:
        print(f"\nBłąd podczas przetwarzania pliku {os.path.basename(audio_path)}: {e}")
        return None


# ZMODYFIKOWANA FUNKCJA: Dodano pytanie o pre-processing
def run_custom_prediction(artifacts_dir, audio_dir, device):
    """
    Główna funkcja uruchamiająca potok predykcji na niestandardowych nagraniach.
    """
    print("--- Uruchamianie predykcji na nagraniach użytkownika ---")
    # ... (logika wyboru modelu bez zmian) ...
    if not os.path.exists(artifacts_dir):
        print(f"BŁĄD: Katalog z artefaktami '{artifacts_dir}' nie istnieje.")
        return

    available_runs = [
        d
        for d in os.listdir(artifacts_dir)
        if d.startswith("run_")
        and os.path.exists(os.path.join(artifacts_dir, d, "best_model.pth"))
    ]
    if not available_runs:
        print(f"Brak folderów z zapisanymi modelami w '{artifacts_dir}'")
        return

    print("\nDostępne wytrenowane modele:")
    for idx, name in enumerate(available_runs):
        print(f"[{idx + 1}] {name}")

    selected_run_folder_name = None
    while True:
        try:
            choice_str = input(
                f"\nWybierz numer modelu (1-{len(available_runs)}) lub '0' aby pominąć: "
            )
            choice = int(choice_str) if choice_str else 0
            if 0 <= choice <= len(available_runs):
                if choice > 0:
                    selected_run_folder_name = available_runs[choice - 1]
                break
            else:
                print("Nieprawidłowy wybór.")
        except ValueError:
            print("Nieprawidłowe dane, podaj liczbę.")

    if not selected_run_folder_name:
        print("\nPominięto predykcję.")
        return

    run_dir = os.path.join(artifacts_dir, selected_run_folder_name)
    model_path = os.path.join(run_dir, "best_model.pth")
    config_path = os.path.join(run_dir, "run_configuration.json")
    print(f"\nŁadowanie modelu z przebiegu: {selected_run_folder_name}...")
    loaded_model = load_best_model(GuitarTabCRNN, model_path, config_path, device)

    if not loaded_model:
        print("Błąd ładowania modelu. Zamykanie.")
        return

    if not os.path.exists(audio_dir):
        print(f"BŁĄD: Katalog z nagraniami '{audio_dir}' nie istnieje.")
        return
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"Nie znaleziono plików .wav w katalogu '{audio_dir}'.")
        return

    # Pytanie o próg onsetu (bez zmian)
    onset_thresh = 0.5
    try:
        user_thresh_str = input(
            f"Podaj próg detekcji onsetu (0.0-1.0, Enter dla domyślnej 0.5): "
        )
        if user_thresh_str:
            onset_thresh = float(user_thresh_str)
    except ValueError:
        print("Nieprawidłowe dane, używam domyślnej 0.5.")

    # NOWE PYTANIE: Włączenie/wyłączenie pre-processingu
    apply_proc_input = input(
        "Czy chcesz zastosować automatyczny pre-processing audio (filtr + redukcja szumu)? (t/n, domyślnie 'n'): "
    ).lower()
    apply_preprocessing = apply_proc_input == "t"

    if apply_preprocessing:
        print("Pre-processing WŁĄCZONY.")
    else:
        print("Pre-processing WYŁĄCZONY.")

    output_dir = os.path.join(run_dir, "custom_predictions")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nWynikowe tabulatury zostaną zapisane w: {output_dir}")

    for wav_file in tqdm(wav_files, desc="Przetwarzanie nagrań"):
        # Przekazanie flagi `apply_preprocessing` do funkcji
        notes = predict_on_single_file(
            model=loaded_model,
            audio_path=os.path.join(audio_dir, wav_file),
            config_obj=config,
            device=device,
            onset_threshold=onset_thresh,
            apply_preprocessing=apply_preprocessing,
        )
        if notes is not None:
            base_name = os.path.splitext(wav_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}_tab.txt")
            save_notes_to_ascii_tab(notes, output_path, base_name, config)

    print("\nZakończono predykcję.")


# ... (główna część uruchamiająca skrypt __main__ bez zmian) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transkrypcja plików audio .wav na tabulatury."
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default=r"C:\Users\lukig\Documents\Programming\music-transcription\python\results\hyperparam_search",
        help="Katalog z modelami.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=r"C:\Users\lukig\Documents\Programming\music-transcription\python\test",
        help="Katalog z plikami .wav.",
    )
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_custom_prediction(args.artifacts, args.audio, dev)
