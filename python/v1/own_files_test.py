import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

from v1.config import (
    AUDIO_SAMPLE_RATE,
    ANNOTATION_HOP,
    NOTES_BINS_PER_SEMITONE,
    GUITAR_BASE_FREQUENCY,
)
from v1.features import compute_cqt
from v1.midi_conversion import predictions_to_midi


def transcribe_single_audio_file(
    audio_path: Path,
    model,
    device,
    thresholds: dict,
    output_dir: Path,
    copy_wav: bool = True,
):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    track_id = audio_path.stem

    if copy_wav:
        output_wav_path = output_dir / f"{track_id}.wav"
        try:
            shutil.copy(audio_path, output_wav_path)
            print(f"  Skopiowano WAV: {audio_path.name} -> {output_wav_path.name}")
        except Exception as e:
            print(f"  Błąd podczas kopiowania {audio_path.name}: {e}")

    try:
        audio_waveform, sr = librosa.load(
            str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True
        )
        if np.max(np.abs(audio_waveform)) > 1e-6:
            audio_waveform = audio_waveform / np.max(np.abs(audio_waveform))
        else:
            print(f"  OSTRZEŻENIE: Plik {audio_path.name} jest bardzo cichy lub pusty.")
            audio_waveform = np.zeros_like(audio_waveform)

    except Exception as e:
        print(f"  Błąd podczas ładowania audio {audio_path.name}: {e}")
        return

    features_cqt = compute_cqt(audio_waveform)
    features_tensor_batch = features_cqt.unsqueeze(0).to(device)
    feature_length_sample = features_cqt.shape[0]

    with torch.no_grad():
        output_logits_sample = model(features_tensor_batch)

    notes_probs_sample = (
        torch.sigmoid(output_logits_sample["notes"][0, :feature_length_sample, :])
        .cpu()
        .numpy()
    )
    onsets_probs_sample = (
        torch.sigmoid(output_logits_sample["onsets"][0, :feature_length_sample, :])
        .cpu()
        .numpy()
    )

    notes_binary_sample = (notes_probs_sample >= thresholds.get("notes", 0.5)).astype(
        int
    )
    onsets_binary_sample = (
        onsets_probs_sample >= thresholds.get("onsets", 0.5)
    ).astype(int)

    output_mid_path = output_dir / f"{track_id}.mid"
    predictions_to_midi(
        notes_binary=notes_binary_sample,
        onsets_binary=onsets_binary_sample,
        output_midi_path=str(output_mid_path),
        annotation_hop=ANNOTATION_HOP,
        notes_bins_per_semitone=NOTES_BINS_PER_SEMITONE,
        guitar_base_frequency=GUITAR_BASE_FREQUENCY,
    )


def transcribe_custom_audio_directory(
    custom_audio_dir: str,
    model,
    device,
    optimal_thresholds_dict: dict,
    output_base_dir: str,
    num_files_to_process: int = None,  # Ile plików przetworzyć (None = wszystkie)
    copy_wav: bool = True,
):
    custom_audio_path = Path(custom_audio_dir)
    output_midi_dir = Path(output_base_dir) / "custom_transcriptions"

    if not custom_audio_path.exists() or not custom_audio_path.is_dir():
        print(
            f"OSTRZEŻENIE: Katalog z własnymi nagraniami '{custom_audio_dir}' nie istnieje."
        )
        return

    wav_files = sorted(list(custom_audio_path.glob("*.wav")))
    if not wav_files:
        print(f"OSTRZEŻENIE: Brak plików .wav w katalogu '{custom_audio_dir}'.")
        return

    if num_files_to_process is not None:
        wav_files_to_process = wav_files[:num_files_to_process]
    else:
        wav_files_to_process = wav_files
        num_files_to_process = len(wav_files)

    print(
        f"\nTranskrypcja {len(wav_files_to_process)} własnych plików .wav z '{custom_audio_dir}'..."
    )

    for audio_file_path in tqdm(
        wav_files_to_process, desc="Transkrypcja własnych nagrań"
    ):
        transcribe_single_audio_file(
            audio_path=audio_file_path,
            model=model,
            device=device,
            thresholds=optimal_thresholds_dict,
            output_dir=output_midi_dir,
            copy_wav=copy_wav,
        )

    print(
        f"\nZakończono transkrypcję własnych nagrań. Pliki znajdują się w: {output_midi_dir}"
    )
