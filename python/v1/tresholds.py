import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, f1_score
from tqdm import tqdm  # Dodajemy tqdm


def gather_predictions_and_targets(model, dataloader, device):
    """
    Zbiera surowe prawdopodobieństwa predykcji i rzeczywiste etykiety
    ze zbioru danych (np. walidacyjnego).
    """
    model.eval()
    all_y_true = {"notes": [], "onsets": [], "contours": []}
    all_y_scores = {"notes": [], "onsets": [], "contours": []}

    with torch.no_grad():
        for batch_data in tqdm(
            dataloader, desc="Zbieranie predykcji do optymalizacji progu"
        ):
            features = batch_data["features"].to(device)
            output_logits = model(features)

            for task in ["notes", "onsets", "contours"]:
                targets_b_t_f = batch_data[task].to(device)
                scores_b_t_f = torch.sigmoid(output_logits[task]).to(device)

                for i in range(targets_b_t_f.shape[0]):
                    length = batch_data["feature_lengths"][i].item()
                    true_sample_active_frames = (
                        targets_b_t_f[i, :length, :].flatten().cpu().numpy()
                    )
                    score_sample_active_frames = (
                        scores_b_t_f[i, :length, :].flatten().cpu().numpy()
                    )

                    all_y_true[task].extend(true_sample_active_frames.tolist())
                    all_y_scores[task].extend(score_sample_active_frames.tolist())
    return all_y_true, all_y_scores


def find_optimal_thresholds(
    all_y_true_val,
    all_y_scores_val,
    tasks=None,
    plot_curves=True,
):
    """
    Znajduje optymalne progi binaryzacji dla podanych zadań,
    maksymalizując F1-score.
    """
    if tasks is None:
        tasks = ["notes", "onsets", "contours"]
    optimal_thresholds = {}

    if plot_curves:
        # Przygotuj figury dla krzywych PR i F1 vs Próg
        fig_pr, axes_pr = plt.subplots(
            1, len(tasks), figsize=(18, 5), squeeze=False
        )  # squeeze=False aby axes_pr zawsze było 2D
        fig_f1, axes_f1 = plt.subplots(1, len(tasks), figsize=(18, 5), squeeze=False)
        fig_pr.suptitle("Krzywe Precision-Recall (Zbiór Walidacyjny)", fontsize=16)
        fig_f1.suptitle("F1-score vs. Próg (Zbiór Walidacyjny)", fontsize=16)

    print("\nObliczanie optymalnych progów na zbiorze walidacyjnym:")
    for i, task in enumerate(tasks):
        print(f"  Przetwarzanie zadania: {task}")
        current_y_true = np.array(all_y_true_val[task])
        current_y_scores = np.array(all_y_scores_val[task])

        if len(current_y_true) > 0 and np.any(current_y_true):
            precision, recall, thresholds_pr_sklearn = precision_recall_curve(
                current_y_true, current_y_scores
            )

            f1_scores_list = []
            candidate_thresholds = np.linspace(
                0.01, 0.99, 100
            )  # Możesz dostosować zakres i liczbę punktów

            for th_candidate in tqdm(
                candidate_thresholds, desc=f" Optymalizacja dla {task}", leave=False
            ):
                y_pred_binary = (current_y_scores >= th_candidate).astype(int)
                f1 = f1_score(current_y_true, y_pred_binary, zero_division=0)
                f1_scores_list.append(f1)

            if f1_scores_list:
                optimal_idx = np.argmax(f1_scores_list)
                optimal_threshold = candidate_thresholds[optimal_idx]
                optimal_f1 = f1_scores_list[optimal_idx]
                optimal_thresholds[task] = optimal_threshold
                print(
                    f"    Automatycznie wybrany próg (max F1) dla '{task}': {optimal_threshold:.4f} (F1-score: {optimal_f1:.4f})"
                )

                if plot_curves:
                    # Krzywa PR
                    axes_pr[0, i].plot(recall, precision, marker=".", label=f"{task}")
                    axes_pr[0, i].set_xlabel("Recall")
                    axes_pr[0, i].set_ylabel("Precision")
                    axes_pr[0, i].set_title(f"{task}")
                    axes_pr[0, i].legend()
                    axes_pr[0, i].grid(True)

                    # F1 vs Próg
                    axes_f1[0, i].plot(candidate_thresholds, f1_scores_list, marker=".")
                    axes_f1[0, i].scatter(
                        optimal_threshold,
                        optimal_f1,
                        color="red",
                        zorder=5,
                        label=f"Max F1 ({optimal_f1:.2f}) przy Th={optimal_threshold:.2f}",
                    )
                    axes_f1[0, i].set_xlabel("Próg (Threshold)")
                    axes_f1[0, i].set_ylabel("F1-score")
                    axes_f1[0, i].set_title(f"{task}")
                    axes_f1[0, i].legend()
                    axes_f1[0, i].grid(True)
            else:
                print(
                    f"    Nie można obliczyć F1-scores dla '{task}'. Używam domyślnego progu 0.5."
                )
                optimal_thresholds[task] = 0.5
        else:
            print(
                f"    Brak danych lub zróżnicowania klas dla '{task}'. Używam domyślnego progu 0.5."
            )
            optimal_thresholds[task] = 0.5

    if plot_curves:
        fig_pr.tight_layout(rect=[0, 0, 1, 0.96])
        fig_pr.show()
        fig_f1.tight_layout(rect=[0, 0, 1, 0.96])
        fig_f1.show()

    return optimal_thresholds
