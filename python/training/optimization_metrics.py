import numpy as np
import config


def calculate_onset_metrics_for_threshold(probs, targets, threshold_val):
    preds_binary = (probs > threshold_val).float()
    tp = ((preds_binary == 1) & (targets == 1)).sum().item()
    fp = ((preds_binary == 1) & (targets == 0)).sum().item()
    fn = ((preds_binary == 0) & (targets == 1)).sum().item()
    tn = ((preds_binary == 0) & (targets == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return precision, recall, f1, accuracy


def find_optimal_onset_metrics(
        all_onset_probs_aggregated,
        all_onset_targets_aggregated,
):
    best_f1_score = -1.0
    optimal_threshold_val = config.DEFAULT_ONSET_THRESHOLD
    best_precision_val = 0.0
    best_recall_val = 0.0
    best_accuracy_val = 0.0

    all_onset_probs_cpu = all_onset_probs_aggregated.cpu()
    all_onset_targets_cpu = all_onset_targets_aggregated.cpu()

    possible_thresholds = np.arange(
        config.ONSET_THRESH_SEARCH_MIN,
        config.ONSET_THRESH_SEARCH_MAX + config.ONSET_THRESH_SEARCH_STEP,
        config.ONSET_THRESH_SEARCH_STEP
    )

    for current_threshold_val in possible_thresholds:
        precision, recall, f1, accuracy = calculate_onset_metrics_for_threshold(
            all_onset_probs_cpu, all_onset_targets_cpu, current_threshold_val
        )

        if f1 > best_f1_score:
            best_f1_score = f1
            optimal_threshold_val = current_threshold_val
            best_precision_val = precision
            best_recall_val = recall
            best_accuracy_val = accuracy

    return optimal_threshold_val, best_f1_score, best_precision_val, best_recall_val, best_accuracy_val