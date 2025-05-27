import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def evaluate_with_optimal_thresholds(
    model,
    dataloader,
    device,
    optimal_thresholds_dict,
    pos_weights_dict,
    task_weights_dict,
):
    model.eval()

    # Akumulatory dla strat (jeśli chcesz je również raportować)
    total_loss_accum = 0.0
    num_batches = 0

    # Akumulatory dla liczników TP, FP, FN dla każdego zadania
    # Użyjemy `calculate_metrics_counts` z `train.py`
    from v1.train import calculate_metrics_counts, precision_recall_f1

    # Upewnij się, że `calculate_loss` jest również dostępne, jeśli liczysz straty
    from v1.losses import calculate_loss

    final_metrics_counts = {
        task: {"tp": 0, "fp": 0, "fn": 0} for task in optimal_thresholds_dict.keys()
    }

    all_y_true_test = {task: [] for task in optimal_thresholds_dict.keys()}
    all_y_scores_test = {task: [] for task in optimal_thresholds_dict.keys()}

    with torch.no_grad():
        for batch_test in tqdm(dataloader, desc="Ewaluacja na zbiorze testowym"):
            features_test = batch_test["features"].to(device)
            output_logits_test = model(features_test)

            # Obliczanie strat (opcjonalnie)
            losses = calculate_loss(
                output_logits_test,
                batch_test,
                device,
                pos_weights_dict,
                task_weights_dict,
            )
            total_loss_accum += losses["total_loss"].item()
            num_batches += 1

            # Obliczanie metryk z optymalnymi progami
            # WAŻNE: Upewnij się, że `calculate_metrics_counts` przyjmuje `thresholds` jako słownik
            batch_metrics_counts = calculate_metrics_counts(
                output_logits_test,
                batch_test,
                device,
                thresholds=optimal_thresholds_dict,
            )

            for task in final_metrics_counts.keys():
                if (
                    task in batch_metrics_counts
                ):  # Sprawdzenie czy zadanie istnieje w wynikach (np. dla konturów)
                    for count_type in final_metrics_counts[task].keys():
                        final_metrics_counts[task][count_type] += batch_metrics_counts[
                            task
                        ][count_type]

            # Zbieranie danych do AUC-PR dla zbioru testowego (opcjonalnie, ale dobra praktyka)
            for task in optimal_thresholds_dict.keys():
                targets_b_t_f_test = batch_test[task].to(device)
                scores_b_t_f_test = torch.sigmoid(output_logits_test[task]).to(device)
                for i in range(targets_b_t_f_test.shape[0]):
                    length_test = batch_test["feature_lengths"][i].item()
                    true_sample_active_frames_test = (
                        targets_b_t_f_test[i, :length_test, :].flatten().cpu().numpy()
                    )
                    score_sample_active_frames_test = (
                        scores_b_t_f_test[i, :length_test, :].flatten().cpu().numpy()
                    )
                    all_y_true_test[task].extend(
                        true_sample_active_frames_test.tolist()
                    )
                    all_y_scores_test[task].extend(
                        score_sample_active_frames_test.tolist()
                    )

    # Obliczenie finalnych metryk P, R, F1
    final_metrics_prf = {}
    for task, counts in final_metrics_counts.items():
        p, r, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        final_metrics_prf[f"{task}_precision_test"] = p
        final_metrics_prf[f"{task}_recall_test"] = r
        final_metrics_prf[f"{task}_f1_test"] = f1

    # Obliczenie AUC-PR dla zbioru testowego
    from sklearn.metrics import auc  # Upewnij się, że jest zaimportowane

    auc_pr_scores_test = {}
    for task in optimal_thresholds_dict.keys():
        if len(all_y_true_test[task]) > 0 and np.any(all_y_true_test[task]):
            precision_c, recall_c, _ = precision_recall_curve(
                all_y_true_test[task], all_y_scores_test[task]
            )
            auc_pr_value = auc(recall_c, precision_c)
            auc_pr_scores_test[f"{task}_auc_pr_test"] = (
                auc_pr_value if not np.isnan(auc_pr_value) else 0.0
            )
        else:
            auc_pr_scores_test[f"{task}_auc_pr_test"] = 0.0

    final_metrics_prf.update(auc_pr_scores_test)
    avg_total_loss = total_loss_accum / num_batches if num_batches > 0 else 0.0

    return {"loss_test": avg_total_loss, "metrics_test": final_metrics_prf}
