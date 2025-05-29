import torch
import numpy as np
import mir_eval
import config
from . import optimization_metrics
from . import note_conversion_utils


def train_one_epoch(model, dataloader, optimizer, criterion_combined, device_to_use):
    model.train()
    epoch_total_loss = 0.0
    epoch_onset_loss = 0.0
    epoch_fret_loss = 0.0

    for features_data, labels_data_tuple, sequence_lengths, _raw_labels_batch in dataloader:
        features_data = features_data.to(device_to_use)
        onset_targets_data, fret_targets_data = labels_data_tuple
        onset_targets_data = onset_targets_data.to(device_to_use)
        fret_targets_data = fret_targets_data.to(device_to_use)
        sequence_lengths = sequence_lengths.to(device_to_use)

        optimizer.zero_grad()
        onset_predictions_logits, fret_predictions_logits = model(features_data)

        loss_val, loss_onset_val, loss_fret_val = criterion_combined(
            onset_predictions_logits, fret_predictions_logits,
            onset_targets_data, fret_targets_data, sequence_lengths
        )

        loss_val.backward()
        optimizer.step()

        epoch_total_loss += loss_val.item()
        epoch_onset_loss += loss_onset_val.item()
        epoch_fret_loss += loss_fret_val.item()

    num_batches = len(dataloader) if len(dataloader) > 0 else 1
    return {
        "train_total_loss": epoch_total_loss / num_batches,
        "train_onset_loss": epoch_onset_loss / num_batches,
        "train_fret_loss": epoch_fret_loss / num_batches,
    }


def evaluate_one_epoch(
        model_to_eval,
        eval_dataloader,
        eval_criterion_combined,
        device_to_use,
        num_fret_classes,
        audio_sr_val,
        audio_hop_len_val,
        log_fixed_onset_threshold=config.DEFAULT_ONSET_THRESHOLD,
):
    model_to_eval.eval()
    epoch_total_loss = 0.0
    epoch_onset_loss = 0.0
    epoch_fret_loss = 0.0

    # Metryki klatkowe dla onsetów (stały próg)
    fixed_thresh_tp_total = 0
    fixed_thresh_fp_total = 0
    fixed_thresh_fn_total = 0
    fixed_thresh_tn_total = 0

    # NOWE: Zmienne do poprawnego obliczania FTab
    ftab_tp_total = 0  # True Positives dla FTab (poprawnie przewidziane aktywne progi na strunach)
    ftab_gt_active_total = 0  # Ground Truth Active (TP_ftab + FN_ftab)
    ftab_pred_active_total = 0  # Predicted Active (TP_ftab + FP_ftab)

    # Listy do agregacji predykcji i etykiet onsetów dla optymalnego progu
    epoch_all_onset_probs_list = []
    epoch_all_onset_targets_list = []

    # Listy do przechowywania wyników mir_eval dla każdej próbki
    all_mir_eval_onset_scores_p = []
    all_mir_eval_onset_scores_r = []
    all_mir_eval_onset_scores_f1 = []

    # Listy do przechowywania wyników TDR (Twoja implementacja) dla każdej próbki
    all_tdr_scores_p = []
    all_tdr_scores_r = []
    all_tdr_scores_f1 = []

    onset_eval_window = 0.05  # 50 ms okno dla mir_eval

    with torch.no_grad():
        for features_data, labels_data_tuple, sequence_lengths, raw_labels_batch_from_loader in eval_dataloader:
            features_data = features_data.to(device_to_use)
            onset_targets_batch_data, fret_targets_batch_data = labels_data_tuple
            onset_targets_batch_data = onset_targets_batch_data.to(device_to_use)
            fret_targets_batch_data = fret_targets_batch_data.to(device_to_use)
            sequence_lengths = sequence_lengths.to(device_to_use)

            onset_pred_logits, fret_pred_logits = model_to_eval(features_data)
            onset_pred_probs_batch = torch.sigmoid(onset_pred_logits)
            fret_pred_indices_batch = torch.argmax(fret_pred_logits, dim=-1)

            # Agregacja dla optymalnego progu onsetów
            epoch_all_onset_probs_list.append(onset_pred_probs_batch.cpu().reshape(-1))
            epoch_all_onset_targets_list.append(onset_targets_batch_data.cpu().reshape(-1))

            # Obliczanie strat
            if eval_criterion_combined:
                loss_val, loss_onset_val, loss_fret_val = eval_criterion_combined(
                    onset_pred_logits, fret_pred_logits,
                    onset_targets_batch_data, fret_targets_batch_data, sequence_lengths
                )
                epoch_total_loss += loss_val.item()
                epoch_onset_loss += loss_onset_val.item()
                epoch_fret_loss += loss_fret_val.item()

            # Metryki klatkowe dla onsetów (stały próg)
            onset_preds_binary_fixed_batch_level = (onset_pred_probs_batch > log_fixed_onset_threshold).float()
            fixed_thresh_tp_total += (
                        (onset_preds_binary_fixed_batch_level == 1) & (onset_targets_batch_data == 1)).sum().item()
            fixed_thresh_fp_total += (
                        (onset_preds_binary_fixed_batch_level == 1) & (onset_targets_batch_data == 0)).sum().item()
            fixed_thresh_fn_total += (
                        (onset_preds_binary_fixed_batch_level == 0) & (onset_targets_batch_data == 1)).sum().item()
            fixed_thresh_tn_total += (
                        (onset_preds_binary_fixed_batch_level == 0) & (onset_targets_batch_data == 0)).sum().item()

            # Definicje dla FTab i TDR
            # Upewnij się, że config.MAX_FRETS i config.FRET_SILENCE_CLASS_OFFSET są zdefiniowane
            # num_fret_classes to MAX_FRETS + 2 (0..MAX_FRETS, MAX_FRETS+1=cisza)
            # silence_fret_idx powinien być ostatnią klasą, czyli num_fret_classes - 1
            silence_fret_idx = num_fret_classes - 1
            time_per_frame = audio_hop_len_val / audio_sr_val

            for i in range(features_data.size(0)):  # Iteracja po próbkach w batchu
                true_len = sequence_lengths[i].item()

                # Wycięcie do rzeczywistej długości sekwencji
                gt_onsets_sample = onset_targets_batch_data[i, :true_len, :]
                gt_frets_sample = fret_targets_batch_data[i, :true_len, :]
                # Używamy zbinaryzowanych predykcji onsetów dla tej próbki
                pred_onsets_sample_binary = onset_preds_binary_fixed_batch_level[i, :true_len, :]
                pred_frets_sample_indices = fret_pred_indices_batch[i, :true_len, :]

                # --- Obliczanie FTab dla próbki ---
                # Maska dla ground truth - aktywne, jeśli próg nie jest ciszą ani paddingiem
                gt_active_mask = (gt_frets_sample != silence_fret_idx) & (gt_frets_sample != config.FRET_PADDING_VALUE)

                # Maska dla predykcji - aktywne, jeśli przewidziano onset ORAZ próg nie jest ciszą
                pred_active_mask = (pred_onsets_sample_binary == 1) & (pred_frets_sample_indices != silence_fret_idx)

                # Poprawnie przewidziane aktywne progi na strunach (True Positives dla FTab)
                correct_and_active_mask = gt_active_mask & pred_active_mask & (
                            gt_frets_sample == pred_frets_sample_indices)

                ftab_tp_total += correct_and_active_mask.sum().item()
                ftab_gt_active_total += gt_active_mask.sum().item()
                ftab_pred_active_total += pred_active_mask.sum().item()

                # --- Obliczanie mir_eval onset dla próbki ---
                gt_onsets_times_sample_list = []
                for string_idx in range(config.DEFAULT_NUM_STRINGS):  # Załóżmy, że config.DEFAULT_NUM_STRINGS istnieje
                    string_onsets_frames_np = gt_onsets_sample[:, string_idx].nonzero(as_tuple=False).squeeze(
                        -1).cpu().numpy()
                    if string_onsets_frames_np.ndim == 0 and string_onsets_frames_np.size > 0:  # Skalar
                        gt_onsets_times_sample_list.append(string_onsets_frames_np.item() * time_per_frame)
                    else:  # Wektor
                        gt_onsets_times_sample_list.extend(string_onsets_frames_np * time_per_frame)
                gt_onsets_times_sample_np = np.unique(np.array(gt_onsets_times_sample_list))

                pred_onsets_times_sample_list = []
                for string_idx in range(config.DEFAULT_NUM_STRINGS):
                    string_onsets_frames_np = pred_onsets_sample_binary[:, string_idx].nonzero(as_tuple=False).squeeze(
                        -1).cpu().numpy()
                    if string_onsets_frames_np.ndim == 0 and string_onsets_frames_np.size > 0:  # Skalar
                        pred_onsets_times_sample_list.append(string_onsets_frames_np.item() * time_per_frame)
                    else:  # Wektor
                        pred_onsets_times_sample_list.extend(string_onsets_frames_np * time_per_frame)
                pred_onsets_times_sample_np = np.unique(np.array(pred_onsets_times_sample_list))

                if len(gt_onsets_times_sample_np) > 0 or len(pred_onsets_times_sample_np) > 0:
                    onset_f1, onset_p, onset_r = mir_eval.onset.f_measure(
                        gt_onsets_times_sample_np, pred_onsets_times_sample_np, window=onset_eval_window
                    )
                    all_mir_eval_onset_scores_p.append(onset_p)
                    all_mir_eval_onset_scores_r.append(onset_r)
                    all_mir_eval_onset_scores_f1.append(onset_f1)
                elif len(gt_onsets_times_sample_np) == 0 and len(
                        pred_onsets_times_sample_np) == 0:  # Obie puste - idealnie
                    all_mir_eval_onset_scores_p.append(1.0)
                    all_mir_eval_onset_scores_r.append(1.0)
                    all_mir_eval_onset_scores_f1.append(1.0)
                # Jeśli jedna pusta, druga nie, to P/R/F1 będą 0.0, co jest poprawne z mir_eval

                # --- Obliczanie TDR (Twoja implementacja Note-Level Tablature F1) dla próbki ---
                gt_notes_raw_s = raw_labels_batch_from_loader[
                    i].cpu().numpy()  # Zakładamy, że raw_labels to (start, end, string, fret, pitch)
                gt_notes_for_eval = []
                if gt_notes_raw_s.ndim == 2 and gt_notes_raw_s.shape[0] > 0:
                    for note_idx in range(gt_notes_raw_s.shape[0]):
                        # Upewnij się, że indeksy w raw_labels_batch_from_loader pasują do oczekiwań
                        # Dla GuitarSet, typowo: 0: start_time, 1: end_time, 2: string, 3: fret
                        string_val = int(gt_notes_raw_s[note_idx, 2])
                        fret_val = int(gt_notes_raw_s[note_idx, 3])
                        if 0 <= string_val < config.DEFAULT_NUM_STRINGS and 0 <= fret_val <= config.MAX_FRETS:
                            gt_notes_for_eval.append({
                                'start_time': gt_notes_raw_s[note_idx, 0],
                                'end_time': gt_notes_raw_s[note_idx, 1],
                                'string': string_val,
                                'fret': fret_val,
                                # 'pitch_midi': gt_notes_raw_s[note_idx, 4] # Jeśli jest dostępne i potrzebne
                            })

                # frames_to_notes_for_eval powinno używać pred_onsets_sample_binary (już zbinaryzowane)
                predicted_notes_sample = note_conversion_utils.frames_to_notes_for_eval(
                    onset_preds_binary_frames=pred_onsets_sample_binary.cpu(),  # Przekazujemy już zbinaryzowane
                    fret_pred_indices_frames=pred_frets_sample_indices.cpu(),
                    frame_hop_length=audio_hop_len_val,
                    audio_sample_rate=audio_sr_val,
                    max_fret_value=config.MAX_FRETS  # Upewnij się, że config.MAX_FRETS jest zdefiniowane
                )

                tp_tdr = 0
                matched_pred_indices = set()
                for gt_note in gt_notes_for_eval:
                    for pred_idx, pred_note in enumerate(predicted_notes_sample):
                        if pred_idx in matched_pred_indices:
                            continue
                        # Dopasowanie nuty
                        onset_match = abs(gt_note['start_time'] - pred_note['start_time']) <= onset_eval_window
                        string_match = gt_note['string'] == pred_note['string']
                        fret_match = gt_note['fret'] == pred_note['fret']

                        if onset_match and string_match and fret_match:
                            tp_tdr += 1
                            matched_pred_indices.add(pred_idx)
                            break

                p_tdr = tp_tdr / len(predicted_notes_sample) if len(predicted_notes_sample) > 0 else (
                    1.0 if len(gt_notes_for_eval) == 0 else 0.0)
                r_tdr = tp_tdr / len(gt_notes_for_eval) if len(gt_notes_for_eval) > 0 else (
                    1.0 if len(predicted_notes_sample) == 0 else 0.0)
                f1_tdr = 2 * p_tdr * r_tdr / (p_tdr + r_tdr) if (p_tdr + r_tdr) > 0 else 0.0

                # Jeśli obie listy są puste, uznajemy za idealne dopasowanie (F1=1)
                if len(gt_notes_for_eval) == 0 and len(predicted_notes_sample) == 0:
                    p_tdr, r_tdr, f1_tdr = 1.0, 1.0, 1.0

                all_tdr_scores_p.append(p_tdr)
                all_tdr_scores_r.append(r_tdr)
                all_tdr_scores_f1.append(f1_tdr)

    num_batches = len(eval_dataloader) if len(eval_dataloader) > 0 else 1

    metrics_results = {  # ZMIENIONA NAZWA SŁOWNIKA
        "val_total_loss": epoch_total_loss / num_batches,
        "val_onset_loss": epoch_onset_loss / num_batches,
        "val_fret_loss": epoch_fret_loss / num_batches,
    }

    # Metryki klatkowe dla onsetów (stały próg) - obliczenie P, R, F1, Acc
    prec_fixed_val = (fixed_thresh_tp_total / (fixed_thresh_tp_total + fixed_thresh_fp_total)) \
        if (fixed_thresh_tp_total + fixed_thresh_fp_total) > 0 else 0.0
    rec_fixed_val = (fixed_thresh_tp_total / (fixed_thresh_tp_total + fixed_thresh_fn_total)) \
        if (fixed_thresh_tp_total + fixed_thresh_fn_total) > 0 else 0.0
    metrics_results["onset_precision_at_fixed_thresh_frame"] = prec_fixed_val
    metrics_results["onset_recall_at_fixed_thresh_frame"] = rec_fixed_val
    metrics_results["onset_f1_at_fixed_thresh_frame"] = \
        (2 * prec_fixed_val * rec_fixed_val / (prec_fixed_val + rec_fixed_val)) \
            if (prec_fixed_val + rec_fixed_val) > 0 else 0.0
    total_elements_for_acc = fixed_thresh_tp_total + fixed_thresh_tn_total + fixed_thresh_fp_total + fixed_thresh_fn_total
    metrics_results["onset_accuracy_at_fixed_thresh_frame"] = \
        ((fixed_thresh_tp_total + fixed_thresh_tn_total) / total_elements_for_acc) \
            if total_elements_for_acc > 0 else 0.0

    # NOWE: Obliczenie FTab, PTab, RTab
    ptab_val = ftab_tp_total / ftab_pred_active_total if ftab_pred_active_total > 0 else 0.0
    rtab_val = ftab_tp_total / ftab_gt_active_total if ftab_gt_active_total > 0 else 0.0
    ftab_val = (2 * ptab_val * rtab_val) / (ptab_val + rtab_val) if (ptab_val + rtab_val) > 0 else 0.0

    metrics_results["val_ptab"] = ptab_val
    metrics_results["val_rtab"] = rtab_val
    metrics_results["val_ftab"] = ftab_val

    # Metryki mir_eval onset (uśrednione)
    metrics_results["val_onset_precision_mir_eval"] = np.mean(
        all_mir_eval_onset_scores_p) if all_mir_eval_onset_scores_p else 0.0
    metrics_results["val_onset_recall_mir_eval"] = np.mean(
        all_mir_eval_onset_scores_r) if all_mir_eval_onset_scores_r else 0.0
    metrics_results["val_onset_f1_mir_eval"] = np.mean(
        all_mir_eval_onset_scores_f1) if all_mir_eval_onset_scores_f1 else 0.0

    # Metryki TDR (Twoja implementacja, uśrednione)
    metrics_results["val_tdr_precision"] = np.mean(all_tdr_scores_p) if all_tdr_scores_p else 0.0
    metrics_results["val_tdr_recall"] = np.mean(all_tdr_scores_r) if all_tdr_scores_r else 0.0
    metrics_results["val_tdr_f1"] = np.mean(all_tdr_scores_f1) if all_tdr_scores_f1 else 0.0

    # Agregacja i optymalny próg dla onsetów klatkowych
    aggregated_onset_probs = torch.cat(epoch_all_onset_probs_list,
                                       dim=0) if epoch_all_onset_probs_list else torch.empty(0)
    aggregated_onset_targets = torch.cat(epoch_all_onset_targets_list,
                                         dim=0) if epoch_all_onset_targets_list else torch.empty(0)

    if aggregated_onset_probs.numel() > 0 and aggregated_onset_targets.numel() > 0:
        opt_thresh_frame, opt_f1_frame, opt_p_frame, opt_r_frame, opt_acc_frame = \
            optimization_metrics.find_optimal_onset_metrics(aggregated_onset_probs.cpu(),
                                                            aggregated_onset_targets.cpu())
        metrics_results["val_onset_f1_optimal_thresh_frame"] = opt_f1_frame
        metrics_results["val_onset_precision_optimal_thresh_frame"] = opt_p_frame
        metrics_results["val_onset_recall_optimal_thresh_frame"] = opt_r_frame
        metrics_results["val_onset_accuracy_optimal_thresh_frame"] = opt_acc_frame
        metrics_results["val_optimal_threshold_epoch_frame"] = opt_thresh_frame
    else:
        metrics_results["val_onset_f1_optimal_thresh_frame"] = 0.0
        metrics_results["val_onset_precision_optimal_thresh_frame"] = 0.0
        metrics_results["val_onset_recall_optimal_thresh_frame"] = 0.0
        metrics_results["val_onset_accuracy_optimal_thresh_frame"] = 0.0
        metrics_results["val_optimal_threshold_epoch_frame"] = config.DEFAULT_ONSET_THRESHOLD

    return (
        metrics_results,  # ZMIENIONA NAZWA ZWRACANEGO SŁOWNIKA
        aggregated_onset_probs,
        aggregated_onset_targets,
    )