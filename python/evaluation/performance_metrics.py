import torch
import numpy as np
import mir_eval
import config
from training import note_conversion_utils  # Dla frames_to_notes_for_eval


def evaluate_model_on_test_set(
        model_to_eval,
        test_dataloader,  # Oczekuje, że zwraca (features, (onsets, frets), lengths, raw_labels_list)
        device_to_use,
        fret_num_classes_val,  # Całkowita liczba klas progów (np. MAX_FRETS + FRET_SILENCE_CLASS_OFFSET + 1)
        optimal_onset_threshold,
        audio_sr,  # Potrzebne dla mir_eval i konwersji na nuty
        audio_hop_length  # Potrzebne dla mir_eval i konwersji na nuty
):
    model_to_eval.eval()

    # Sumy dla metryk ramkowych onsetów
    onset_tp_total_frame = 0
    onset_fp_total_frame = 0
    onset_fn_total_frame = 0
    onset_tn_total_frame = 0

    # Sumy dla metryk ramkowych progów
    fret_correct_predictions_total_frame = 0
    fret_total_elements_frame = 0
    fret_correct_predictions_active_frame = 0
    fret_total_active_frames_total_frame = 0

    # Sumy dla ftab
    ftab_correct_frames_sum = 0
    ftab_gt_active_frames_sum = 0

    # Listy dla mir_eval onsets
    all_test_mir_eval_onset_p = []
    all_test_mir_eval_onset_r = []
    all_test_mir_eval_onset_f1 = []

    # Listy dla TDR
    all_test_tdr_p = []
    all_test_tdr_r = []
    all_test_tdr_f1 = []

    onset_eval_window_mir = 0.05  # 50 ms

    with torch.no_grad():
        for features_batch, labels_tuple_batch, lengths_batch, raw_labels_batch in test_dataloader:
            features_batch = features_batch.to(device_to_use)
            onset_targets_b, fret_targets_b = labels_tuple_batch
            onset_targets_b = onset_targets_b.to(device_to_use)
            fret_targets_b = fret_targets_b.to(device_to_use)
            # lengths_batch jest już na CPU z DataLoader'a, ale jeśli nie, to .to(device_to_use)

            onset_logits_b, fret_logits_b = model_to_eval(features_batch)
            onset_probs_b = torch.sigmoid(onset_logits_b)
            onset_preds_binary_b = (onset_probs_b > optimal_onset_threshold).float()
            fret_pred_indices_b = torch.argmax(fret_logits_b, dim=-1)

            # --- Metryki ramkowe ---
            onset_tp_total_frame += ((onset_preds_binary_b == 1) & (onset_targets_b == 1)).sum().item()
            onset_fp_total_frame += ((onset_preds_binary_b == 1) & (onset_targets_b == 0)).sum().item()
            onset_fn_total_frame += ((onset_preds_binary_b == 0) & (onset_targets_b == 1)).sum().item()
            onset_tn_total_frame += ((onset_preds_binary_b == 0) & (onset_targets_b == 0)).sum().item()

            fret_correct_predictions_total_frame += (fret_pred_indices_b == fret_targets_b).sum().item()
            fret_total_elements_frame += fret_targets_b.numel()

            silence_fret_idx_val = config.MAX_FRETS + config.FRET_SILENCE_CLASS_OFFSET
            mask_active_b = (fret_targets_b != silence_fret_idx_val) & (fret_targets_b != config.FRET_PADDING_VALUE)

            fret_correct_predictions_active_frame += (
                        (fret_pred_indices_b == fret_targets_b) & mask_active_b).sum().item()
            fret_total_active_frames_total_frame += mask_active_b.sum().item()

            # --- Metryki ftab, mir_eval onsets, TDR (per próbka w batchu) ---
            time_per_frame_val = audio_hop_length / audio_sr

            for i in range(features_batch.size(0)):
                true_len_sample = lengths_batch[i].item()

                gt_onsets_s = onset_targets_b[i, :true_len_sample, :]
                gt_frets_s = fret_targets_b[i, :true_len_sample, :]
                pred_onsets_s_binary = onset_preds_binary_b[i, :true_len_sample, :]
                pred_frets_s_indices = fret_pred_indices_b[i, :true_len_sample, :]

                # ftab
                gt_active_mask_s = (gt_frets_s != silence_fret_idx_val) & (gt_frets_s != config.FRET_PADDING_VALUE)
                ftab_gt_active_frames_sum += gt_active_mask_s.sum().item()
                pred_active_mask_s = (pred_onsets_s_binary == 1) & (pred_frets_s_indices != silence_fret_idx_val)
                correct_and_active_s = gt_active_mask_s & pred_active_mask_s & (gt_frets_s == pred_frets_s_indices)
                ftab_correct_frames_sum += correct_and_active_s.sum().item()

                # mir_eval onsets
                gt_onsets_times_s = []
                for str_idx in range(config.DEFAULT_NUM_STRINGS):
                    str_onsets_f = gt_onsets_s[:, str_idx].nonzero(as_tuple=False).squeeze().cpu().numpy()
                    gt_onsets_times_s.extend(str_onsets_f * time_per_frame_val)
                gt_onsets_times_s = np.unique(np.array(gt_onsets_times_s))

                pred_onsets_times_s = []
                for str_idx in range(config.DEFAULT_NUM_STRINGS):
                    str_onsets_f = pred_onsets_s_binary[:, str_idx].nonzero(as_tuple=False).squeeze().cpu().numpy()
                    pred_onsets_times_s.extend(str_onsets_f * time_per_frame_val)
                pred_onsets_times_s = np.unique(np.array(pred_onsets_times_s))

                if len(gt_onsets_times_s) > 0 or len(pred_onsets_times_s) > 0:
                    ons_p, ons_r, ons_f1, _ = mir_eval.onset.f_measure(gt_onsets_times_s, pred_onsets_times_s,
                                                                       window=onset_eval_window_mir)
                    all_test_mir_eval_onset_p.append(ons_p)
                    all_test_mir_eval_onset_r.append(ons_r)
                    all_test_mir_eval_onset_f1.append(ons_f1)

                # TDR
                gt_notes_raw_s = raw_labels_batch[i].cpu().numpy()
                gt_notes_eval_s = []
                if gt_notes_raw_s.ndim == 2 and gt_notes_raw_s.shape[0] > 0:
                    for note_idx_s in range(gt_notes_raw_s.shape[0]):
                        str_val, fret_val = int(gt_notes_raw_s[note_idx_s, 2]), int(gt_notes_raw_s[note_idx_s, 3])
                        if 0 <= str_val < config.DEFAULT_NUM_STRINGS and 0 <= fret_val <= config.MAX_FRETS:
                            gt_notes_eval_s.append({
                                'start_time': gt_notes_raw_s[note_idx_s, 0], 'end_time': gt_notes_raw_s[note_idx_s, 1],
                                'string': str_val, 'fret': fret_val
                            })

                pred_notes_s = note_conversion_utils.frames_to_notes_for_eval(
                    onset_preds_binary_frames=pred_onsets_s_binary.cpu(),
                    fret_pred_indices_frames=pred_frets_s_indices.cpu(),
                    frame_hop_length=audio_hop_length, audio_sample_rate=audio_sr,
                    max_fret_value=config.MAX_FRETS
                )

                tp_tdr_s = 0
                matched_pred_indices_s = set()
                for gt_n_s in gt_notes_eval_s:
                    for pred_idx_s, pred_n_s in enumerate(pred_notes_s):
                        if pred_idx_s in matched_pred_indices_s: continue
                        onset_match_s = abs(gt_n_s['start_time'] - pred_n_s['start_time']) <= onset_eval_window_mir
                        string_match_s = gt_n_s['string'] == pred_n_s['string']
                        fret_match_s = gt_n_s['fret'] == pred_n_s['fret']
                        if onset_match_s and string_match_s and fret_match_s:
                            tp_tdr_s += 1
                            matched_pred_indices_s.add(pred_idx_s)
                            break

                p_tdr_s = tp_tdr_s / len(pred_notes_s) if len(pred_notes_s) > 0 else (
                    1.0 if len(gt_notes_eval_s) == 0 else 0.0)
                r_tdr_s = tp_tdr_s / len(gt_notes_eval_s) if len(gt_notes_eval_s) > 0 else (
                    1.0 if len(pred_notes_s) == 0 else 0.0)
                f1_tdr_s = 2 * p_tdr_s * r_tdr_s / (p_tdr_s + r_tdr_s) if (p_tdr_s + r_tdr_s) > 0 else 0.0

                if len(gt_notes_eval_s) == 0 and len(pred_notes_s) == 0:  # Ideal for silence
                    p_tdr_s, r_tdr_s, f1_tdr_s = 1.0, 1.0, 1.0

                all_test_tdr_p.append(p_tdr_s)
                all_test_tdr_r.append(r_tdr_s)
                all_test_tdr_f1.append(f1_tdr_s)

    # Obliczanie finalnych metryk ramkowych
    onset_precision_frame = onset_tp_total_frame / (onset_tp_total_frame + onset_fp_total_frame) if (
                                                                                                                onset_tp_total_frame + onset_fp_total_frame) > 0 else 0.0
    onset_recall_frame = onset_tp_total_frame / (onset_tp_total_frame + onset_fn_total_frame) if (
                                                                                                             onset_tp_total_frame + onset_fn_total_frame) > 0 else 0.0
    onset_f1_frame = 2 * (onset_precision_frame * onset_recall_frame) / (
                onset_precision_frame + onset_recall_frame) if (onset_precision_frame + onset_recall_frame) > 0 else 0.0
    total_onset_elements_for_acc = onset_tp_total_frame + onset_tn_total_frame + onset_fp_total_frame + onset_fn_total_frame
    onset_accuracy_frame = (
                                       onset_tp_total_frame + onset_tn_total_frame) / total_onset_elements_for_acc if total_onset_elements_for_acc > 0 else 0.0

    fret_accuracy_overall_val = fret_correct_predictions_total_frame / fret_total_elements_frame if fret_total_elements_frame > 0 else 0.0
    fret_accuracy_active_val = fret_correct_predictions_active_frame / fret_total_active_frames_total_frame if fret_total_active_frames_total_frame > 0 else 0.0

    # Obliczanie finalnych metryk ftab, mir_eval, TDR
    final_ftab = ftab_correct_frames_sum / ftab_gt_active_frames_sum if ftab_gt_active_frames_sum > 0 else 0.0

    final_onset_p_mir_eval = np.mean(all_test_mir_eval_onset_p) if all_test_mir_eval_onset_p else 0.0
    final_onset_r_mir_eval = np.mean(all_test_mir_eval_onset_r) if all_test_mir_eval_onset_r else 0.0
    final_onset_f1_mir_eval = np.mean(all_test_mir_eval_onset_f1) if all_test_mir_eval_onset_f1 else 0.0

    final_tdr_p = np.mean(all_test_tdr_p) if all_test_tdr_p else 0.0
    final_tdr_r = np.mean(all_test_tdr_r) if all_test_tdr_r else 0.0  # TDR
    final_tdr_f1 = np.mean(all_test_tdr_f1) if all_test_tdr_f1 else 0.0

    test_set_metrics_results = {
        "test_onset_precision_frame": onset_precision_frame,
        "test_onset_recall_frame": onset_recall_frame,
        "test_onset_f1_frame": onset_f1_frame,
        "test_onset_accuracy_frame": onset_accuracy_frame,
        "test_fret_accuracy_overall": fret_accuracy_overall_val,
        "test_fret_accuracy_active": fret_accuracy_active_val,
        "test_ftab": final_ftab,
        "test_onset_precision_mir_eval": final_onset_p_mir_eval,
        "test_onset_recall_mir_eval": final_onset_r_mir_eval,
        "test_onset_f1_mir_eval": final_onset_f1_mir_eval,
        "test_tdr_precision": final_tdr_p,
        "test_tdr_recall": final_tdr_r,  # TDR
        "test_tdr_f1": final_tdr_f1
    }

    print("\n--- Metryki na Zbiorze Testowym ---")
    for metric_name, metric_value in test_set_metrics_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    return test_set_metrics_results