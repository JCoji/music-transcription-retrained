import os
import torch
import config


def _generate_tab_slot_content_str(is_onset_present, fret_number, max_fret_val):
    if is_onset_present:
        if 0 <= fret_number <= max_fret_val:
            return str(fret_number).ljust(
                config.TAB_SLOT_CHAR_WIDTH, config.TAB_SUSTAIN_CHAR
            )[: config.TAB_SLOT_CHAR_WIDTH]
        else:
            return config.TAB_SUSTAIN_CHAR * config.TAB_SLOT_CHAR_WIDTH
    return config.TAB_SUSTAIN_CHAR * config.TAB_SLOT_CHAR_WIDTH


def _generate_tablature_matrix_slots(
    onset_data_frames,
    fret_data_frames,
    num_total_frames,
    num_total_strings,
    max_fret_val,
):
    silence_fret_val = max_fret_val + config.FRET_SILENCE_CLASS_OFFSET
    num_tab_slots = (
        num_total_frames + config.TAB_FRAMES_PER_SLOT - 1
    ) // config.TAB_FRAMES_PER_SLOT

    tab_matrix = [[""] * num_tab_slots for _ in range(num_total_strings)]

    current_played_fret_on_string = [-1] * num_total_strings

    for slot_idx in range(num_tab_slots):
        start_frame_for_slot = slot_idx * config.TAB_FRAMES_PER_SLOT
        end_frame_for_slot = min(
            start_frame_for_slot + config.TAB_FRAMES_PER_SLOT, num_total_frames
        )

        for string_model_idx in range(num_total_strings):
            onset_detected_in_slot_for_string = False
            fret_at_onset_time_in_slot = -1

            for frame_k_idx in range(start_frame_for_slot, end_frame_for_slot):
                if onset_data_frames[frame_k_idx, string_model_idx].item() > 0.5:
                    fret_val_at_onset = fret_data_frames[
                        frame_k_idx, string_model_idx
                    ].item()
                    if 0 <= fret_val_at_onset <= max_fret_val:
                        fret_at_onset_time_in_slot = fret_val_at_onset
                    else:
                        fret_at_onset_time_in_slot = silence_fret_val
                    onset_detected_in_slot_for_string = True
                    break

            slot_content_char_list = ""
            if onset_detected_in_slot_for_string:
                if fret_at_onset_time_in_slot != silence_fret_val:
                    slot_content_char_list = str(fret_at_onset_time_in_slot)
                    current_played_fret_on_string[string_model_idx] = (
                        fret_at_onset_time_in_slot
                    )
                else:
                    slot_content_char_list = config.TAB_SUSTAIN_CHAR
                    current_played_fret_on_string[string_model_idx] = -1
            else:
                slot_content_char_list = config.TAB_SUSTAIN_CHAR

            if len(slot_content_char_list) > config.TAB_SLOT_CHAR_WIDTH:
                slot_content_char_list = (
                    slot_content_char_list[: config.TAB_SLOT_CHAR_WIDTH - 1]
                    + config.TAB_OVERFLOW_CHAR
                )
            elif slot_content_char_list.isdigit():
                slot_content_char_list = slot_content_char_list.ljust(
                    config.TAB_SLOT_CHAR_WIDTH, config.TAB_SUSTAIN_CHAR
                )
            else:
                slot_content_char_list = (
                    slot_content_char_list * config.TAB_SLOT_CHAR_WIDTH
                )

            tab_matrix[string_model_idx][slot_idx] = slot_content_char_list

    return tab_matrix


def _format_tablature_matrix_to_text(tab_matrix_data_slots, num_total_strings):
    output_text_lines = []
    string_display_names = (
        config.TAB_STRING_NAMES_DISPLAY_6_STRING
        if num_total_strings == 6
        else [str(i) for i in range(num_total_strings)]
    )

    if not tab_matrix_data_slots or not tab_matrix_data_slots[0]:
        return "Brak danych do sformatowania."

    total_num_slots = len(tab_matrix_data_slots[0])

    for slot_start_batch_idx in range(
        0, total_num_slots, config.TAB_LINE_BREAK_AFTER_SLOTS
    ):
        slot_end_batch_idx = min(
            slot_start_batch_idx + config.TAB_LINE_BREAK_AFTER_SLOTS, total_num_slots
        )

        for display_string_idx in range(num_total_strings):
            model_string_idx_for_display = (num_total_strings - 1) - display_string_idx
            line_str = (
                string_display_names[display_string_idx] + config.TAB_SEPARATOR_CHAR
            )

            current_slot_count_in_line = 0
            for current_slot_idx in range(slot_start_batch_idx, slot_end_batch_idx):
                line_str += tab_matrix_data_slots[model_string_idx_for_display][
                    current_slot_idx
                ]
                current_slot_count_in_line += 1
                if (
                    current_slot_count_in_line
                    % config.TAB_GROUP_SEPARATOR_EVERY_N_SLOTS
                    == 0
                    and current_slot_idx < slot_end_batch_idx - 1
                ):
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
    max_fret_val,
    output_directory_path=None,
):
    if not (dataset_instance and len(dataset_instance) > 0):
        print("Dataset jest pusty, pomijam generowanie tabulatur.")
        return

    if output_directory_path is None:
        print("Katalog wyjściowy nie jest zdefiniowany, pomijam generowanie tabulatur.")
        return

    os.makedirs(output_directory_path, exist_ok=True)

    model_to_eval.eval()
    all_sample_match_percentages = []

    with torch.no_grad():
        for sample_idx in sample_indices_list:
            if sample_idx >= len(dataset_instance):
                print(
                    f"Ostrzeżenie: Indeks próbki {sample_idx} poza zakresem datasetu. Pomijam."
                )
                continue

            try:
                features_sample, (onset_gt_sample_cpu, fret_gt_sample_cpu), _ = (
                    dataset_instance[sample_idx]
                )
            except TypeError:
                features_sample, (onset_gt_sample_cpu, fret_gt_sample_cpu) = (
                    dataset_instance[sample_idx]
                )
            except Exception as e_get:
                print(
                    f"Błąd podczas pobierania próbki {sample_idx} dla tabulatur: {e_get}. Pomijam."
                )
                continue

            base_track_id_str = f"{config.TAB_TRACK_ID_PREFIX}_{sample_idx}"
            if (
                hasattr(dataset_instance, "base_track_ids")
                and dataset_instance.base_track_ids
                and sample_idx < len(dataset_instance.base_track_ids)
            ):
                base_track_id_str = dataset_instance.base_track_ids[sample_idx]

            num_frames_gt = onset_gt_sample_cpu.shape[0]
            gt_tab_matrix_data = _generate_tablature_matrix_slots(
                onset_data_frames=onset_gt_sample_cpu.cpu(),
                fret_data_frames=fret_gt_sample_cpu.cpu(),
                num_total_frames=num_frames_gt,
                num_total_strings=config.DEFAULT_NUM_STRINGS,
                max_fret_val=max_fret_val,
            )
            gt_tab_text_output = _format_tablature_matrix_to_text(
                gt_tab_matrix_data, config.DEFAULT_NUM_STRINGS
            )
            gt_output_filename_str = (
                f"{base_track_id_str}{config.TAB_GT_FILENAME_SUFFIX}"
            )
            gt_output_filepath_str = os.path.join(
                output_directory_path, gt_output_filename_str
            )
            try:
                with open(gt_output_filepath_str, "w", encoding="utf-8") as f_gt:
                    f_gt.write(f"Track ID: {base_track_id_str}\n")
                    f_gt.write("--- Ground Truth Tablature ---\n\n")
                    f_gt.write(gt_tab_text_output)
            except Exception as e_write_gt:
                print(
                    f"Błąd podczas zapisywania pliku Ground Truth tabulatury {gt_output_filepath_str}: {e_write_gt}"
                )

            features_sample_dev = features_sample.unsqueeze(0).to(device_to_use)
            onset_pred_logits, fret_pred_logits = model_to_eval(features_sample_dev)
            onset_pred_probs_sample = torch.sigmoid(onset_pred_logits.squeeze(0).cpu())
            onset_pred_binary_sample = (
                onset_pred_probs_sample > onset_threshold_optimal
            ).float()
            fret_pred_indices_sample = torch.argmax(
                fret_pred_logits.squeeze(0).cpu(), dim=-1
            )
            num_frames_pred = onset_pred_binary_sample.shape[0]

            pred_tab_matrix_data = _generate_tablature_matrix_slots(
                onset_data_frames=onset_pred_binary_sample,
                fret_data_frames=fret_pred_indices_sample,
                num_total_frames=num_frames_pred,
                num_total_strings=config.DEFAULT_NUM_STRINGS,
                max_fret_val=max_fret_val,
            )
            pred_tab_text_output = _format_tablature_matrix_to_text(
                pred_tab_matrix_data, config.DEFAULT_NUM_STRINGS
            )
            pred_output_filename_str = f"{base_track_id_str}{config.TAB_PRED_FILENAME_SUFFIX_TEMPLATE.format(threshold=onset_threshold_optimal).replace('{threshold:.2f}', f'_{onset_threshold_optimal:.2f}')}"
            pred_output_filepath_str = os.path.join(
                output_directory_path, pred_output_filename_str
            )
            try:
                with open(pred_output_filepath_str, "w", encoding="utf-8") as f_pred:
                    f_pred.write(f"Track ID: {base_track_id_str}\n")
                    f_pred.write(
                        f"--- Predicted Tablature (Onset Threshold: {onset_threshold_optimal:.2f}) ---\n\n"
                    )
                    f_pred.write(pred_tab_text_output)
            except Exception as e_write_pred:
                print(
                    f"Błąd podczas zapisywania pliku Predykowanej tabulatury {pred_output_filepath_str}: {e_write_pred}"
                )

            total_active_gt_frames_for_matching = 0
            matched_frames = 0
            num_common_frames = min(num_frames_gt, num_frames_pred)
            num_strings = config.DEFAULT_NUM_STRINGS

            for frame_k in range(num_common_frames):
                for string_s in range(num_strings):
                    gt_onset_active = (
                        onset_gt_sample_cpu[frame_k, string_s].item() > 0.5
                    )
                    gt_fret_val = fret_gt_sample_cpu[frame_k, string_s].item()

                    is_gt_relevant_for_match = gt_onset_active and (
                        0 <= gt_fret_val <= max_fret_val
                    )

                    if is_gt_relevant_for_match:
                        total_active_gt_frames_for_matching += 1

                        pred_onset_active = (
                            onset_pred_binary_sample[frame_k, string_s].item() > 0.5
                        )
                        pred_fret_val = fret_pred_indices_sample[
                            frame_k, string_s
                        ].item()

                        if pred_onset_active and gt_fret_val == pred_fret_val:
                            matched_frames += 1

            current_sample_match_percentage = 0.0
            if total_active_gt_frames_for_matching > 0:
                current_sample_match_percentage = (
                    matched_frames / total_active_gt_frames_for_matching
                ) * 100.0
            elif total_active_gt_frames_for_matching == 0:
                current_sample_match_percentage = 100.0

            print(
                f"\n  Podsumowanie dopasowania dla próbki '{base_track_id_str}' (Próg onsetu: {onset_threshold_optimal:.2f}):"
            )
            print(
                f"    Liczba aktywnych ramek w Ground Truth (do porównania): {total_active_gt_frames_for_matching}"
            )
            print(
                f"    Liczba poprawnie przewidzianych aktywnych ramek (onset+próg): {matched_frames}"
            )
            print(
                f"    Procentowe dopasowanie tabulatury dla próbki: {current_sample_match_percentage:.2f}%"
            )
            all_sample_match_percentages.append(current_sample_match_percentage)

    if all_sample_match_percentages:
        average_match_overall = sum(all_sample_match_percentages) / len(
            all_sample_match_percentages
        )
        print(f"\n--- Ogólne Podsumowanie Dopasowania Tabulatur ---")
        print(
            f"  Średnie procentowe dopasowanie dla {len(all_sample_match_percentages)} próbek: {average_match_overall:.2f}%"
        )
    else:
        print(
            "\nBrak przetworzonych próbek do obliczenia średniego dopasowania tabulatur."
        )
