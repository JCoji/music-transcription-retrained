import os
import torch
import config


def _generate_tab_slot_content_str(
        is_onset_present,
        fret_number,
        is_note_currently_sustained,  # Ten argument nie jest używany w oryginalnej logice, ale może być przydatny
        max_fret_val  # Zmieniona nazwa dla spójności
):
    silence_fret_val = max_fret_val + config.FRET_SILENCE_CLASS_OFFSET

    if is_onset_present:
        if 0 <= fret_number <= max_fret_val:
            return str(fret_number).ljust(config.TAB_SLOT_CHAR_WIDTH, config.TAB_SUSTAIN_CHAR)[
                   :config.TAB_SLOT_CHAR_WIDTH]
        else:  # Onset, ale próg jest ciszą lub poza zakresem - traktuj jak sustain/puste
            return config.TAB_SUSTAIN_CHAR * config.TAB_SLOT_CHAR_WIDTH
    # Jeśli nie ma onsetu, zawsze sustain/puste, niezależnie od is_note_currently_sustained
    # Oryginalna logika upraszcza to do wyświetlania sustain jeśli nie ma onsetu.
    # Aby było bardziej precyzyjne, potrzebowalibyśmy śledzić stan każdej nuty (czy jest aktywna).
    # Na razie trzymamy się uproszczenia: jeśli nie ma onsetu, to sustain.
    return config.TAB_SUSTAIN_CHAR * config.TAB_SLOT_CHAR_WIDTH


def _generate_tablature_matrix_slots(
        onset_data_frames,  # (T, S)
        fret_data_frames,  # (T, S)
        num_total_frames,
        num_total_strings,
        max_fret_val
):
    silence_fret_val = max_fret_val + config.FRET_SILENCE_CLASS_OFFSET
    num_tab_slots = (num_total_frames + config.TAB_FRAMES_PER_SLOT - 1) // config.TAB_FRAMES_PER_SLOT

    tab_matrix = [[""] * num_tab_slots for _ in range(num_total_strings)]

    # Uproszczona logika - bazuje tylko na pierwszym frame w slocie dla onsetu,
    # i ostatnim znanym progu jeśli nie ma onsetu (co jest niedokładne dla sustainu)
    # Dla lepszego sustainu, trzeba by śledzić aktywność nuty.

    current_played_fret_on_string = [-1] * num_total_strings  # Ostatnio zagrany próg (nie cisza)

    for slot_idx in range(num_tab_slots):
        start_frame_for_slot = slot_idx * config.TAB_FRAMES_PER_SLOT
        end_frame_for_slot = min(start_frame_for_slot + config.TAB_FRAMES_PER_SLOT, num_total_frames)

        for string_model_idx in range(num_total_strings):
            onset_detected_in_slot_for_string = False
            fret_at_onset_time_in_slot = -1  # Domyślnie nieznany/cisza

            # Sprawdź, czy jest onset w tym slocie dla danej struny
            for frame_k_idx in range(start_frame_for_slot, end_frame_for_slot):
                if onset_data_frames[frame_k_idx, string_model_idx].item() > 0.5:  # Próg onsetu
                    fret_val_at_onset = fret_data_frames[frame_k_idx, string_model_idx].item()
                    if 0 <= fret_val_at_onset <= max_fret_val:
                        fret_at_onset_time_in_slot = fret_val_at_onset
                    else:  # Onset, ale próg jest ciszą
                        fret_at_onset_time_in_slot = silence_fret_val
                    onset_detected_in_slot_for_string = True
                    break

            slot_content_char_list = ""
            if onset_detected_in_slot_for_string:
                if fret_at_onset_time_in_slot != silence_fret_val:
                    slot_content_char_list = str(fret_at_onset_time_in_slot)
                    current_played_fret_on_string[string_model_idx] = fret_at_onset_time_in_slot
                else:  # Onset na ciszy
                    slot_content_char_list = config.TAB_SUSTAIN_CHAR
                    current_played_fret_on_string[string_model_idx] = -1  # Resetujemy ostatnio grany próg
            else:  # Brak onsetu w slocie - sustain poprzedniego lub cisza
                # W oryginalnej logice, jeśli nie ma onsetu, to zawsze sustain.
                # Lepsze byłoby sprawdzenie, czy poprzednia nuta jest nadal aktywna.
                # Dla uproszczenia (jak w oryginale):
                slot_content_char_list = config.TAB_SUSTAIN_CHAR
                # Jeśli chcemy resetować sustain przy braku onsetu:
                # current_played_fret_on_string[string_model_idx] = -1

            # Formatowanie długości stringa dla slotu
            if len(slot_content_char_list) > config.TAB_SLOT_CHAR_WIDTH:
                slot_content_char_list = slot_content_char_list[
                                         :config.TAB_SLOT_CHAR_WIDTH - 1] + config.TAB_OVERFLOW_CHAR
            elif slot_content_char_list.isdigit():
                slot_content_char_list = slot_content_char_list.ljust(config.TAB_SLOT_CHAR_WIDTH,
                                                                      config.TAB_SUSTAIN_CHAR)
            else:  # config.TAB_SUSTAIN_CHAR
                slot_content_char_list = slot_content_char_list * config.TAB_SLOT_CHAR_WIDTH

            tab_matrix[string_model_idx][slot_idx] = slot_content_char_list

    return tab_matrix


def _format_tablature_matrix_to_text(
        tab_matrix_data_slots,  # Lista list, gdzie zewnętrzna lista to struny
        num_total_strings
):
    output_text_lines = []
    string_display_names = config.TAB_STRING_NAMES_DISPLAY_6_STRING if num_total_strings == 6 else [str(i) for i in
                                                                                                    range(
                                                                                                        num_total_strings)]

    if not tab_matrix_data_slots or not tab_matrix_data_slots[0]:
        return "Brak danych do sformatowania."

    total_num_slots = len(tab_matrix_data_slots[0])

    for slot_start_batch_idx in range(0, total_num_slots, config.TAB_LINE_BREAK_AFTER_SLOTS):
        slot_end_batch_idx = min(slot_start_batch_idx + config.TAB_LINE_BREAK_AFTER_SLOTS, total_num_slots)

        for display_string_idx in range(num_total_strings):
            # Tabulatury są zwykle rysowane od najwyższej struny (e) na górze
            model_string_idx_for_display = (num_total_strings - 1) - display_string_idx
            line_str = string_display_names[display_string_idx] + config.TAB_SEPARATOR_CHAR

            current_slot_count_in_line = 0
            for current_slot_idx in range(slot_start_batch_idx, slot_end_batch_idx):
                line_str += tab_matrix_data_slots[model_string_idx_for_display][current_slot_idx]
                current_slot_count_in_line += 1
                if current_slot_count_in_line % config.TAB_GROUP_SEPARATOR_EVERY_N_SLOTS == 0 and \
                        current_slot_idx < slot_end_batch_idx - 1:
                    line_str += config.TAB_SEPARATOR_CHAR
            output_text_lines.append(line_str)
        output_text_lines.append("")
    return "\n".join(output_text_lines)


def generate_text_tablature_comparison(
        model_to_eval,
        dataset_instance,
        device_to_use,
        sample_indices_list,
        onset_threshold_optimal,
        max_fret_val,  # Zmieniona nazwa
        output_directory_path=None
        # sr i hop_length nie są bezpośrednio używane do generowania tabulatur tekstowych,
        # ale mogą być potrzebne, jeśli logika generowania slotów by się zmieniła.
):
    if not (dataset_instance and len(dataset_instance) > 0):
        print("Dataset jest pusty, pomijam generowanie tabulatur.")
        return

    if output_directory_path is None:
        print("Katalog wyjściowy nie jest zdefiniowany, pomijam generowanie tabulatur.")
        return

    os.makedirs(output_directory_path, exist_ok=True)

    model_to_eval.eval()
    with torch.no_grad():
        for sample_idx in sample_indices_list:
            if sample_idx >= len(dataset_instance):
                print(f"Ostrzeżenie: Indeks próbki {sample_idx} poza zakresem datasetu. Pomijam.")
                continue

            try:
                features_sample, (onset_gt_sample_cpu, fret_gt_sample_cpu), _ = dataset_instance[sample_idx]
            except TypeError:  # Starsza wersja datasetu
                features_sample, (onset_gt_sample_cpu, fret_gt_sample_cpu) = dataset_instance[sample_idx]
            except Exception as e_get:
                print(f"Błąd podczas pobierania próbki {sample_idx} dla tabulatur: {e_get}. Pomijam.")
                continue

            base_track_id_str = f"{config.TAB_TRACK_ID_PREFIX}_{sample_idx}"
            if hasattr(dataset_instance, 'base_track_ids') and dataset_instance.base_track_ids and sample_idx < len(
                    dataset_instance.base_track_ids):
                base_track_id_str = dataset_instance.base_track_ids[sample_idx]

            num_frames_gt = onset_gt_sample_cpu.shape[0]

            gt_tab_matrix_data = _generate_tablature_matrix_slots(
                onset_data_frames=onset_gt_sample_cpu.cpu(),  # Upewniamy się, że na CPU
                fret_data_frames=fret_gt_sample_cpu.cpu(),
                num_total_frames=num_frames_gt,
                num_total_strings=config.DEFAULT_NUM_STRINGS,
                max_fret_val=max_fret_val
            )
            gt_tab_text_output = _format_tablature_matrix_to_text(
                gt_tab_matrix_data, config.DEFAULT_NUM_STRINGS
            )
            gt_output_filename_str = f"{base_track_id_str}{config.TAB_GT_FILENAME_SUFFIX}"
            gt_output_filepath_str = os.path.join(output_directory_path, gt_output_filename_str)
            try:
                with open(gt_output_filepath_str, "w", encoding="utf-8") as f_gt:
                    f_gt.write(f"Track ID: {base_track_id_str}\n")
                    f_gt.write("--- Ground Truth Tablature ---\n\n")
                    f_gt.write(gt_tab_text_output)
            except Exception as e_write_gt:
                print(f"Błąd podczas zapisywania pliku Ground Truth tabulatury {gt_output_filepath_str}: {e_write_gt}")

            features_sample_dev = features_sample.unsqueeze(0).to(device_to_use)
            onset_pred_logits, fret_pred_logits = model_to_eval(features_sample_dev)

            onset_pred_probs_sample = torch.sigmoid(onset_pred_logits.squeeze(0).cpu())
            onset_pred_binary_sample = (onset_pred_probs_sample > onset_threshold_optimal).float()
            fret_pred_indices_sample = torch.argmax(fret_pred_logits.squeeze(0).cpu(), dim=-1)

            num_frames_pred = onset_pred_binary_sample.shape[0]  # Użyj długości predykcji

            pred_tab_matrix_data = _generate_tablature_matrix_slots(
                onset_data_frames=onset_pred_binary_sample,
                fret_data_frames=fret_pred_indices_sample,
                num_total_frames=num_frames_pred,  # Użyj tej samej liczby ramek co predykcja
                num_total_strings=config.DEFAULT_NUM_STRINGS,
                max_fret_val=max_fret_val
            )
            pred_tab_text_output = _format_tablature_matrix_to_text(
                pred_tab_matrix_data, config.DEFAULT_NUM_STRINGS
            )

            # Poprawka nazwy pliku, aby poprawnie formatować track_id i threshold
            pred_output_filename_str = f"{base_track_id_str}{config.TAB_PRED_FILENAME_SUFFIX_TEMPLATE.format(threshold=onset_threshold_optimal).replace('{threshold:.2f}', f'_{onset_threshold_optimal:.2f}')}"
            # Bardziej bezpośrednie formatowanie, jeśli szablon jest prosty:
            # pred_output_filename_str = f"{base_track_id_str}_tablature_prediction_thresh{onset_threshold_optimal:.2f}.txt"

            pred_output_filepath_str = os.path.join(output_directory_path, pred_output_filename_str)
            try:
                with open(pred_output_filepath_str, "w", encoding="utf-8") as f_pred:
                    f_pred.write(f"Track ID: {base_track_id_str}\n")
                    f_pred.write(f"--- Predicted Tablature (Onset Threshold: {onset_threshold_optimal:.2f}) ---\n\n")
                    f_pred.write(pred_tab_text_output)
            except Exception as e_write_pred:
                print(
                    f"Błąd podczas zapisywania pliku Predykowanej tabulatury {pred_output_filepath_str}: {e_write_pred}")