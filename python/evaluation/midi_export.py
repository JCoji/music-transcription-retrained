import os
import torch
import shutil  # Do kopiowania plików WAV
import mirdata  # Do ładowania GuitarSet dla ścieżek audio
import config
from training import note_conversion_utils  # Import naszej funkcji

try:
    import pretty_midi

    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False


def generate_midi_from_predictions(
        model_to_eval,
        dataset_instance,
        device_to_use,
        sample_indices_list,
        onset_threshold_optimal,
        sampling_rate,
        hop_len,
        max_fret_value,
        midi_output_directory,
        guitarset_data_home,
        guitarset_loader=None
):
    if not PRETTY_MIDI_AVAILABLE:
        print("Biblioteka pretty_midi nie jest dostępna. Pomijam generowanie plików MIDI.")
        return

    os.makedirs(midi_output_directory, exist_ok=True)

    if guitarset_loader is None and guitarset_data_home:
        try:
            guitarset_loader = mirdata.initialize('guitarset', data_home=guitarset_data_home)
        except Exception as e:
            print(f"Błąd inicjalizacji mirdata dla GuitarSet: {e}. Nie można skopiować plików WAV.")

    model_to_eval.eval()
    with torch.no_grad():
        for sample_idx in sample_indices_list:
            if sample_idx >= len(dataset_instance):
                print(
                    f"Ostrzeżenie: Indeks próbki {sample_idx} poza zakresem datasetu ({len(dataset_instance)}). Pomijam.")
                continue

            # Zakładamy, że dataset __getitem__ zwraca: features, (onset_targets, fret_targets), raw_labels
            # Dla generowania MIDI potrzebujemy tylko features.
            # Jeśli dataset zwraca inaczej, trzeba to dostosować.
            try:
                features_sample, _, _ = dataset_instance[sample_idx]  # Pobieramy tylko cechy
            except TypeError:  # Jeśli dataset zwraca tylko (features, labels_tuple)
                features_sample, _ = dataset_instance[sample_idx]
            except Exception as e_get:
                print(f"Błąd podczas pobierania próbki {sample_idx} z datasetu: {e_get}. Pomijam.")
                continue

            base_track_id_str = config.DEFAULT_TRACK_ID_BASE
            if hasattr(dataset_instance, 'base_track_ids') and dataset_instance.base_track_ids and sample_idx < len(
                    dataset_instance.base_track_ids):
                base_track_id_str = dataset_instance.base_track_ids[sample_idx]
            elif hasattr(dataset_instance, 'get_full_track_id_for_item'):  # Alternatywa, jeśli dataset ma tę metodę
                try:
                    full_id = dataset_instance.get_full_track_id_for_item(sample_idx)
                    base_track_id_str = os.path.splitext(os.path.basename(full_id))[0]
                except:  # Leniwa obsługa błędu
                    pass

            features_sample_dev = features_sample.unsqueeze(0).to(device_to_use)
            onset_pred_logits, fret_pred_logits = model_to_eval(features_sample_dev)

            onset_pred_probs = torch.sigmoid(onset_pred_logits.squeeze(0).cpu())
            onset_pred_binary_frames = (onset_pred_probs > onset_threshold_optimal).float()
            fret_pred_indices_frames = torch.argmax(fret_pred_logits.squeeze(0).cpu(), dim=-1)

            predicted_notes_info_list = note_conversion_utils.frames_to_notes_for_eval(
                onset_preds_binary_frames=onset_pred_binary_frames,
                fret_pred_indices_frames=fret_pred_indices_frames,
                frame_hop_length=hop_len,
                audio_sample_rate=sampling_rate,
                max_fret_value=max_fret_value,
                min_note_duration_frames=config.MIN_NOTE_DURATION_FRAMES,
                midi_velocity=config.DEFAULT_MIDI_VELOCITY,
                open_string_pitches=config.OPEN_STRING_PITCHES_MIDI
            )

            if predicted_notes_info_list:
                pm_object = pretty_midi.PrettyMIDI(initial_tempo=config.DEFAULT_MIDI_INITIAL_TEMPO)
                guitar_instrument_obj = pretty_midi.Instrument(program=config.ACOUSTIC_GUITAR_STEEL_PROGRAM)

                for note_data in predicted_notes_info_list:
                    # frames_to_notes_for_eval zwraca słownik, tworzymy z niego obiekt Note
                    midi_note = pretty_midi.Note(
                        velocity=config.DEFAULT_MIDI_VELOCITY,  # Można też pobrać z note_data, jeśli tam jest
                        pitch=note_data['pitch_midi'],
                        start=note_data['start_time'],
                        end=note_data['end_time']
                    )
                    guitar_instrument_obj.notes.append(midi_note)

                pm_object.instruments.append(guitar_instrument_obj)

                midi_filename_str = f"{base_track_id_str}{config.OUTPUT_MIDI_FILENAME_SUFFIX}"
                midi_file_path_str = os.path.join(midi_output_directory, midi_filename_str)
                try:
                    pm_object.write(midi_file_path_str)
                except Exception as e_write:
                    print(f"Błąd podczas zapisywania pliku MIDI {midi_file_path_str}: {e_write}")

            if guitarset_loader:
                try:
                    full_track_id_for_mirdata = None
                    if hasattr(dataset_instance, 'get_full_track_id_for_item') and callable(
                            getattr(dataset_instance, 'get_full_track_id_for_item')):
                        full_track_id_for_mirdata = dataset_instance.get_full_track_id_for_item(sample_idx)
                    elif hasattr(dataset_instance,
                                 'full_track_ids') and dataset_instance.full_track_ids and sample_idx < len(
                            dataset_instance.full_track_ids):
                        full_track_id_for_mirdata = dataset_instance.full_track_ids[sample_idx]

                    if full_track_id_for_mirdata:
                        track_obj_mirdata = guitarset_loader.track(full_track_id_for_mirdata)
                        original_audio_source_path = None
                        if hasattr(track_obj_mirdata,
                                   'audio_mix_path') and track_obj_mirdata.audio_mix_path and os.path.exists(
                                track_obj_mirdata.audio_mix_path):
                            original_audio_source_path = track_obj_mirdata.audio_mix_path
                        elif hasattr(track_obj_mirdata,
                                     'audio_mic_path') and track_obj_mirdata.audio_mic_path and os.path.exists(
                                track_obj_mirdata.audio_mic_path):
                            original_audio_source_path = track_obj_mirdata.audio_mic_path

                        if original_audio_source_path:
                            wav_filename_str = f"{base_track_id_str}{config.ORIGINAL_WAV_FILENAME_SUFFIX}"
                            wav_file_path_str = os.path.join(midi_output_directory, wav_filename_str)
                            shutil.copy2(original_audio_source_path, wav_file_path_str)
                    else:
                        print(f"  Nie udało się ustalić pełnego track_id dla {base_track_id_str}, aby skopiować WAV.")
                except mirdata.core.errors.TrackIdError:
                    print(
                        f"  Track ID {full_track_id_for_mirdata if full_track_id_for_mirdata else base_track_id_str} nie znaleziony w mirdata. Pomijam kopiowanie WAV.")
                except Exception as e_wav:
                    print(f"Błąd podczas kopiowania oryginalnego pliku WAV dla {base_track_id_str}: {e_wav}")