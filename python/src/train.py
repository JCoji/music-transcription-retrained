import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.losses import calculate_loss
from src.visualization import plot_sample_data_with_predictions


def calculate_metrics_counts(
    model_output: dict, batch: dict, device: torch.device, threshold: float = 0.5
) -> dict:
    """
    Oblicza liczbę prawdziwie pozytywnych (TP), fałszywie pozytywnych (FP)
    oraz fałszywie negatywnych (FN) dla każdego zadania (nuty, początki, kontury) w danym batchu.

    Args:
        model_output (dict): Słownik zawierający logity z modelu dla kluczy 'notes', 'onsets', 'contours'.
        batch (dict): Słownik zawierający dane wejściowe, w tym targety ('notes', 'onsets', 'contours')
                      oraz maskę ('mask').
        device (torch.device): Urządzenie (np. 'cpu', 'cuda'), na którym wykonywane są obliczenia.
        threshold (float, optional): Próg do binaryzacji predykcji sigmoidalnych. Domyślnie 0.5.

    Returns:
        dict: Słownik zagnieżdżony, gdzie klucze pierwszego poziomu to zadania ('notes', 'onsets', 'contours'),
              a klucze drugiego poziomu to 'tp', 'fp', 'fn' z odpowiadającymi im wartościami liczbowymi.
    """
    # Pobranie targetów i maski z batcha, przeniesienie na odpowiednie urządzenie
    notes_target = batch["notes"].to(device).float()
    onsets_target = batch["onsets"].to(device).float()
    contours_target = batch["contours"].to(device).float()
    mask = batch["mask"].to(device)  # Maska o kształcie (B, T)

    # Pobranie logitów z modelu
    notes_logits = model_output["notes"]
    onsets_logits = model_output["onsets"]
    contours_logits = model_output["contours"]

    # Konwersja logitów na binarne predykcje przy użyciu funkcji sigmoid i progu
    notes_pred = (torch.sigmoid(notes_logits) > threshold).float()
    onsets_pred = (torch.sigmoid(onsets_logits) > threshold).float()
    contours_pred = (torch.sigmoid(contours_logits) > threshold).float()

    # Rozszerzenie maski do wymiarów targetów/predykcji (B, T, F), aby zastosować ją element-wise
    mask_notes = mask.unsqueeze(-1).expand_as(notes_target)
    mask_contours = mask.unsqueeze(-1).expand_as(
        contours_target
    )  # Ta sama maska dla nut i onsetów, inna dla konturów, jeśli F się różni

    # Zastosowanie maski do targetów i predykcji, aby uwzględnić tylko aktywne ramki czasowe
    notes_target_masked = notes_target * mask_notes
    onsets_target_masked = (
        onsets_target * mask_notes
    )  # Onsety używają tej samej maski co nuty
    contours_target_masked = contours_target * mask_contours

    notes_pred_masked = notes_pred * mask_notes
    onsets_pred_masked = (
        onsets_pred * mask_notes
    )  # Onsety używają tej samej maski co nuty
    contours_pred_masked = contours_pred * mask_contours

    # Obliczenie TP, FP, FN dla nut
    tp_notes = (notes_pred_masked * notes_target_masked).sum()
    fp_notes = (
        notes_pred_masked * (1 - notes_target_masked)
    ).sum()  # Predykcja pozytywna, target negatywny
    fn_notes = (
        (1 - notes_pred_masked) * notes_target_masked
    ).sum()  # Predykcja negatywna, target pozytywny

    # Obliczenie TP, FP, FN dla początków (onsetów)
    tp_onsets = (onsets_pred_masked * onsets_target_masked).sum()
    fp_onsets = (onsets_pred_masked * (1 - onsets_target_masked)).sum()
    fn_onsets = ((1 - onsets_pred_masked) * onsets_target_masked).sum()

    # Obliczenie TP, FP, FN dla konturów
    tp_contours = (contours_pred_masked * contours_target_masked).sum()
    fp_contours = (contours_pred_masked * (1 - contours_target_masked)).sum()
    fn_contours = ((1 - contours_pred_masked) * contours_target_masked).sum()

    return {
        "notes": {"tp": tp_notes.item(), "fp": fp_notes.item(), "fn": fn_notes.item()},
        "onsets": {
            "tp": tp_onsets.item(),
            "fp": fp_onsets.item(),
            "fn": fn_onsets.item(),
        },
        "contours": {
            "tp": tp_contours.item(),
            "fp": fp_contours.item(),
            "fn": fn_contours.item(),
        },
    }


def precision_recall_f1(
    tp: float, fp: float, fn: float, eps: float = 1e-8
) -> tuple[float, float, float]:
    """
    Oblicza precyzję, czułość (recall) oraz F1-score na podstawie liczby TP, FP, FN.

    Args:
        tp (float): Liczba prawdziwie pozytywnych.
        fp (float): Liczba fałszywie pozytywnych.
        fn (float): Liczba fałszywie negatywnych.
        eps (float, optional): Mała wartość dodawana do mianowników w celu uniknięcia dzielenia przez zero.
                               Domyślnie 1e-8.

    Returns:
        tuple[float, float, float]: Krotka zawierająca (precyzja, czułość, F1-score).
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return precision, recall, f1


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weights: dict,
    task_weights: dict,
    gradient_clip_val: float,
) -> dict:
    """
    Przeprowadza jedną epokę treningu modelu.

    Iteruje po wszystkich batchach w dataloaderze, wykonuje przejście w przód i w tył,
    optymalizuje wagi modelu i akumuluje wartości funkcji straty.

    Args:
        model (nn.Module): Model do trenowania.
        dataloader (torch.utils.data.DataLoader): DataLoader dostarczający dane treningowe.
        optimizer (torch.optim.Optimizer): Optymalizator używany do aktualizacji wag modelu.
        device (torch.device): Urządzenie (np. 'cpu', 'cuda'), na którym trenowany jest model.
        pos_weights (dict): Słownik z wagami 'pos_weight' dla funkcji straty BCEWithLogitsLoss.
        task_weights (dict): Słownik z wagami dla poszczególnych zadań w łącznej funkcji straty.
        gradient_clip_val (float): Maksymalna norma gradientów (używana do obcinania gradientów).

    Returns:
        dict: Słownik zawierający średnie wartości funkcji straty dla całej epoki ('total_loss',
              'loss_notes', 'loss_onsets', 'loss_contours').
    """
    model.train()  # Ustawienie modelu w tryb treningowy
    total_loss_accum = 0.0
    notes_loss_accum = 0.0
    onsets_loss_accum = 0.0
    contours_loss_accum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Trening", leave=False):
        features = batch["features"].to(device)  # Przeniesienie cech na urządzenie

        optimizer.zero_grad()  # Wyzerowanie gradientów przed nowym obliczeniem
        output_logits = model(features)  # Przejście w przód (predykcja)

        # Obliczenie funkcji straty
        losses = calculate_loss(output_logits, batch, device, pos_weights, task_weights)
        total_loss = losses["total_loss"]

        total_loss.backward()  # Propagacja wsteczna gradientów

        # Obcięcie gradientów, aby zapobiec eksplodującym gradientom
        clip_grad_norm_(model.parameters(), gradient_clip_val)

        optimizer.step()  # Aktualizacja wag modelu

        # Akumulacja strat
        total_loss_accum += losses["total_loss"].item()
        notes_loss_accum += losses["loss_notes"].item()
        onsets_loss_accum += losses["loss_onsets"].item()
        contours_loss_accum += losses["loss_contours"].item()
        num_batches += 1

    # Obliczenie średnich strat dla epoki
    return {
        "total_loss": total_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_notes": notes_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_onsets": onsets_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_contours": contours_loss_accum / num_batches if num_batches > 0 else 0.0,
    }


def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    pos_weights: dict,
    task_weights: dict,
    epoch_num: int,
    checkpoint_dir: str,
    model_name_stem: str,
    plot_every_n_epochs: int = 5,
    sample_to_plot_idx: int = 0,
) -> dict:
    """
    Przeprowadza jedną epokę ewaluacji modelu na zbiorze walidacyjnym lub testowym.

    Oblicza funkcje straty oraz metryki (precyzja, czułość, F1, AUC-PR).
    Opcjonalnie generuje wizualizacje predykcji dla wybranej próbki.

    Args:
        model (nn.Module): Model do ewaluacji.
        dataloader (torch.utils.data.DataLoader): DataLoader dostarczający dane ewaluacyjne.
        device (torch.device): Urządzenie do obliczeń.
        pos_weights (dict): Wagi 'pos_weight' dla funkcji straty.
        task_weights (dict): Wagi dla poszczególnych zadań w funkcji straty.
        epoch_num (int): Numer bieżącej epoki (dla celów logowania i nazewnictwa plików).
        checkpoint_dir (str): Katalog główny dla checkpointów i wykresów.
        model_name_stem (str): Nazwa modelu (bez rozszerzenia) używana do tworzenia podkatalogów.
        plot_every_n_epochs (int, optional): Częstotliwość generowania wizualizacji (co ile epok).
                                             Domyślnie 5.
        sample_to_plot_idx (int, optional): Indeks próbki z pierwszego batcha do wizualizacji.
                                            Domyślnie 0.

    Returns:
        dict: Słownik zawierający średnie straty ('loss') oraz obliczone metryki ('metrics').
    """
    model.eval()  # Ustawienie modelu w tryb ewaluacyjny
    total_loss_accum = 0.0
    notes_loss_accum = 0.0
    onsets_loss_accum = 0.0
    contours_loss_accum = 0.0

    # Listy do przechowywania wszystkich targetów i predykcji (prawdopodobieństw) dla AUC-PR
    all_y_true = {"notes": [], "onsets": [], "contours": []}
    all_y_scores = {"notes": [], "onsets": [], "contours": []}

    first_batch_for_plot = None
    first_batch_logits_for_plot = None

    # Akumulatory dla liczników TP, FP, FN
    metrics_counts = {
        "notes": {"tp": 0, "fp": 0, "fn": 0},
        "onsets": {"tp": 0, "fp": 0, "fn": 0},
        "contours": {"tp": 0, "fp": 0, "fn": 0},
    }
    num_batches = 0

    with torch.no_grad():  # Wyłączenie obliczania gradientów podczas ewaluacji
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Ewaluacja", leave=False)
        ):
            features = batch["features"].to(device)
            output_logits = model(features)  # Przejście w przód

            # Zapisanie pierwszego batcha i jego logitów na potrzeby wizualizacji
            if batch_idx == 0 and (
                epoch_num % plot_every_n_epochs == 0 or epoch_num == 1
            ):
                first_batch_for_plot = {k: v.cpu().clone() for k, v in batch.items()}
                first_batch_logits_for_plot = {
                    k: v.cpu().clone() for k, v in output_logits.items()
                }

            # Obliczenie i akumulacja strat
            losses = calculate_loss(
                output_logits, batch, device, pos_weights, task_weights
            )
            total_loss_accum += losses["total_loss"].item()
            notes_loss_accum += losses["loss_notes"].item()
            onsets_loss_accum += losses["loss_onsets"].item()
            contours_loss_accum += losses["loss_contours"].item()

            # Obliczenie i akumulacja liczników TP, FP, FN
            batch_metrics = calculate_metrics_counts(output_logits, batch, device)
            for task in metrics_counts.keys():
                for count_type in metrics_counts[task].keys():
                    metrics_counts[task][count_type] += batch_metrics[task][count_type]

            # Zbieranie danych do AUC-PR
            # Iterujemy po zadaniach i odpowiednich kluczach logitów
            for task, logits_key in [
                ("notes", "notes"),
                ("onsets", "onsets"),
                ("contours", "contours"),
            ]:
                targets_b_t_f = batch[task].to(
                    device
                )  # Targety dla zadania (Batch, Czas, Częstotliwości)
                scores_b_t_f = torch.sigmoid(output_logits[logits_key]).to(
                    device
                )  # Prawdopodobieństwa

                # Iterujemy po próbkach w batchu
                for i in range(targets_b_t_f.shape[0]):
                    # Uwzględniamy tylko aktywne ramki czasowe na podstawie 'feature_lengths'
                    length = batch["feature_lengths"][
                        i
                    ].item()  # Długość sekwencji dla danej próbki
                    # Spłaszczamy wymiary czasu i częstotliwości dla aktywnych ramek
                    true_sample_active_frames = (
                        targets_b_t_f[i, :length, :].flatten().cpu().numpy()
                    )
                    score_sample_active_frames = (
                        scores_b_t_f[i, :length, :].flatten().cpu().numpy()
                    )

                    all_y_true[task].extend(true_sample_active_frames.tolist())
                    all_y_scores[task].extend(score_sample_active_frames.tolist())
            num_batches += 1

    # Generowanie wizualizacji, jeśli warunki są spełnione
    if (
        epoch_num % plot_every_n_epochs == 0 or epoch_num == 1
    ) and first_batch_for_plot is not None:
        print(f"Generowanie wizualizacji dla epoki {epoch_num}...")
        # Przygotowanie prawdopodobieństw predykcji dla wybranej próbki
        preds_for_plot_probs = {
            "notes": torch.sigmoid(
                first_batch_logits_for_plot["notes"][sample_to_plot_idx]
            ).numpy(),
            "onsets": torch.sigmoid(
                first_batch_logits_for_plot["onsets"][sample_to_plot_idx]
            ).numpy(),
            "contours": torch.sigmoid(
                first_batch_logits_for_plot["contours"][sample_to_plot_idx]
            ).numpy(),
        }
        # Utworzenie ścieżki zapisu dla wykresu
        plot_dir = Path(checkpoint_dir) / "plots" / model_name_stem
        plot_dir.mkdir(
            parents=True, exist_ok=True
        )  # Utworzenie katalogu, jeśli nie istnieje
        fig_path = plot_dir / f"epoch_{epoch_num}_sample_{sample_to_plot_idx}.png"

        # Wywołanie funkcji wizualizującej
        # Upewnij się, że 'first_batch_for_plot' zawiera wszystkie klucze potrzebne przez funkcję wizualizującą
        plot_sample_data_with_predictions(
            batch=first_batch_for_plot,  # Zawiera dane referencyjne (targety)
            predictions_probs=preds_for_plot_probs,  # Przekazanie prawdopodobieństw
            sample_idx=sample_to_plot_idx,
            save_path=str(fig_path),
        )
        print(f"Wizualizacja zapisana w {fig_path}")

    # Obliczenie średnich strat
    avg_losses = {
        "total_loss": total_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_notes": notes_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_onsets": onsets_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_contours": contours_loss_accum / num_batches if num_batches > 0 else 0.0,
    }

    # Obliczenie metryk (P, R, F1) na podstawie zagregowanych liczników TP, FP, FN
    final_metrics = {}
    for task, counts in metrics_counts.items():
        p, r, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        final_metrics[f"{task}_precision"] = p
        final_metrics[f"{task}_recall"] = r
        final_metrics[f"{task}_f1"] = f1

    # Obliczenie AUC-PR
    auc_pr_scores = {}
    for task in ["notes", "onsets", "contours"]:
        # AUC-PR można obliczyć tylko, jeśli mamy co najmniej jedną próbkę każdej klasy
        if len(all_y_true[task]) > 0 and len(np.unique(all_y_true[task])) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(
                all_y_true[task], all_y_scores[task]
            )
            auc_pr_value = auc(recall_curve, precision_curve)
            # Zabezpieczenie przed NaN w AUC-PR
            auc_pr_scores[f"{task}_auc_pr"] = (
                auc_pr_value if not np.isnan(auc_pr_value) else 0.0
            )
        else:
            # Jeśli warunki nie są spełnione, ustawiamy AUC-PR na 0.0
            auc_pr_scores[f"{task}_auc_pr"] = 0.0
            print(
                f"OSTRZEŻENIE: Nie można obliczyć AUC-PR dla zadania '{task}' w epoce {epoch_num} z powodu niewystarczającej liczby danych lub braku zróżnicowania klas. Ustawiono na 0.0."
            )

    final_metrics.update(auc_pr_scores)
    return {"loss": avg_losses, "metrics": final_metrics}


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    pos_weights: dict,
    task_weights: dict,
    gradient_clip_val: float,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "best_model.pt",
    early_stopping_patience: int = 10,
    metric_for_checkpoint: str = "val_onsets_f1",
    plot_every_n_epochs: int = 5,
    sample_to_plot_idx: int = 0,
):
    """
    Główna funkcja trenująca model przez zadaną liczbę epok.

    Obsługuje pętlę treningową, ewaluację na zbiorze walidacyjnym, logowanie do TensorBoard,
    zapisywanie najlepszego modelu (checkpointing) oraz mechanizm wczesnego zatrzymania (early stopping).

    Args:
        model (nn.Module): Model do trenowania.
        train_loader (torch.utils.data.DataLoader): DataLoader dla danych treningowych.
        val_loader (torch.utils.data.DataLoader): DataLoader dla danych walidacyjnych.
        optimizer (torch.optim.Optimizer): Optymalizator.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler współczynnika uczenia.
        num_epochs (int): Maksymalna liczba epok treningu.
        device (torch.device): Urządzenie do obliczeń.
        pos_weights (dict): Wagi 'pos_weight' dla funkcji straty.
        task_weights (dict): Wagi dla poszczególnych zadań.
        gradient_clip_val (float): Wartość do obcinania gradientów.
        checkpoint_dir (str, optional): Katalog do zapisywania checkpointów i logów. Domyślnie "checkpoints".
        model_name (str, optional): Nazwa pliku dla najlepszego modelu. Domyślnie "best_model.pt".
        early_stopping_patience (int, optional): Liczba epok bez poprawy metryki, po której trening jest zatrzymywany.
                                                Domyślnie 10.
        metric_for_checkpoint (str, optional): Nazwa metryki używanej do wyboru najlepszego modelu
                                              i dla wczesnego zatrzymania (np. 'val_onsets_f1', 'val_loss_total').
                                              Domyślnie 'val_onsets_f1'.
        plot_every_n_epochs (int, optional): Częstotliwość generowania wizualizacji. Domyślnie 5.
        sample_to_plot_idx (int, optional): Indeks próbki do wizualizacji. Domyślnie 0.

    Returns:
        dict: Historia treningu zawierająca straty i metryki dla każdej epoki.
    """
    model_name_stem = model_name.replace(
        ".pt", ""
    )  # Nazwa modelu bez rozszerzenia dla katalogów logów/plotów

    # Inicjalizacja loggera TensorBoard
    log_path = Path(checkpoint_dir) / "logs" / model_name_stem
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"Logi TensorBoard będą zapisywane w: {log_path}")

    # Inicjalizacja najlepszej wartości metryki w zależności od trybu schedulera i typu metryki
    # Jeśli metryka to strata, dążymy do minimum, jeśli F1/Precision/Recall/AUC, dążymy do maksimum.
    best_metric_value = float("-inf")
    if "loss" in metric_for_checkpoint.lower():  # Jeśli monitorujemy stratę
        best_metric_value = float("inf")
    elif (
        scheduler and hasattr(scheduler, "mode") and scheduler.mode == "min"
    ):  # Jeśli scheduler dąży do minimum (np. dla straty)
        best_metric_value = float("inf")

    epochs_no_improve = 0  # Licznik epok bez poprawy metryki (dla early stopping)
    stopped_epoch = num_epochs  # Epoka, w której trening faktycznie się zakończył

    # Słownik do przechowywania historii treningu
    history = {
        "train_loss_total": [],
        "val_loss_total": [],
        "val_loss_notes": [],
        "val_loss_onsets": [],
        "val_loss_contours": [],
        "val_notes_f1": [],
        "val_onsets_f1": [],
        "val_contours_f1": [],
        "val_notes_precision": [],
        "val_notes_recall": [],
        "val_onsets_precision": [],
        "val_onsets_recall": [],
        "val_contours_precision": [],
        "val_contours_recall": [],
        "val_notes_auc_pr": [],
        "val_onsets_auc_pr": [],
        "val_contours_auc_pr": [],
    }

    # Przygotowanie ścieżki do zapisu najlepszego modelu
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_path / model_name

    print(f"Rozpoczynanie treningu na maksymalnie {num_epochs} epok...")
    print(
        f"Najlepszy model będzie zapisywany na podstawie metryki: '{metric_for_checkpoint}'"
    )
    print(f"Wczesne zatrzymanie po {early_stopping_patience} epokach bez poprawy.")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoka {epoch + 1}/{num_epochs} ---")

        # Faza treningu
        train_losses_dict = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weights,
            task_weights,
            gradient_clip_val,
        )
        history["train_loss_total"].append(train_losses_dict["total_loss"])
        writer.add_scalar(
            "Strata/trening_całkowita", train_losses_dict["total_loss"], epoch
        )
        writer.add_scalar("Strata/trening_nuty", train_losses_dict["loss_notes"], epoch)
        writer.add_scalar(
            "Strata/trening_poczatki", train_losses_dict["loss_onsets"], epoch
        )
        writer.add_scalar(
            "Strata/trening_kontury", train_losses_dict["loss_contours"], epoch
        )

        # Faza ewaluacji
        eval_results = evaluate_epoch(
            model,
            val_loader,
            device,
            pos_weights,
            task_weights,
            epoch_num=epoch + 1,
            checkpoint_dir=checkpoint_dir,
            model_name_stem=model_name_stem,
            plot_every_n_epochs=plot_every_n_epochs,
            sample_to_plot_idx=sample_to_plot_idx,
        )
        val_losses = eval_results["loss"]
        val_metrics = eval_results["metrics"]

        # Zapisywanie strat walidacyjnych do historii i TensorBoard
        history["val_loss_total"].append(val_losses["total_loss"])
        history["val_loss_notes"].append(val_losses["loss_notes"])
        history["val_loss_onsets"].append(val_losses["loss_onsets"])
        history["val_loss_contours"].append(val_losses["loss_contours"])
        writer.add_scalar("Strata/walidacja_całkowita", val_losses["total_loss"], epoch)
        writer.add_scalar("Strata/walidacja_nuty", val_losses["loss_notes"], epoch)
        writer.add_scalar("Strata/walidacja_poczatki", val_losses["loss_onsets"], epoch)
        writer.add_scalar(
            "Strata/walidacja_kontury", val_losses["loss_contours"], epoch
        )

        # Zapisywanie metryk walidacyjnych do historii i TensorBoard
        for task in ["notes", "onsets", "contours"]:
            for metric_type in ["f1", "precision", "recall", "auc_pr"]:
                key = f"{task}_{metric_type}"  # np. 'notes_f1'
                history_key = f"val_{key}"  # np. 'val_notes_f1'
                metric_val = val_metrics.get(
                    key, 0.0
                )  # Domyślnie 0.0, jeśli metryka nie została obliczona (np. AUC-PR)

                # Użyj setdefault, aby zainicjować listę, jeśli klucz jeszcze nie istnieje
                history.setdefault(history_key, []).append(metric_val)
                # Nazwy dla TensorBoard, np. F1/walidacja_nuty
                writer.add_scalar(
                    f"{metric_type.replace('_', '-').capitalize()}/walidacja_{task}",
                    metric_val,
                    epoch,
                )

        # Wyświetlanie podsumowania epoki
        print(
            f"  Strata Wal.: {val_losses['total_loss']:.4f} (N: {val_losses['loss_notes']:.4f}, O: {val_losses['loss_onsets']:.4f}, C: {val_losses['loss_contours']:.4f})"
        )
        print(
            f"  Metryki Wal.: Nuty F1={val_metrics.get('notes_f1', 0):.4f} (P={val_metrics.get('notes_precision', 0):.4f}, R={val_metrics.get('notes_recall', 0):.4f}, AUC-PR={val_metrics.get('notes_auc_pr', 0):.4f})"
        )
        print(
            f"                 Początki F1={val_metrics.get('onsets_f1', 0):.4f} (P={val_metrics.get('onsets_precision', 0):.4f}, R={val_metrics.get('onsets_recall', 0):.4f}, AUC-PR={val_metrics.get('onsets_auc_pr', 0):.4f})"
        )
        print(
            f"                 Kontury F1={val_metrics.get('contours_f1', 0):.4f} (P={val_metrics.get('contours_precision', 0):.4f}, R={val_metrics.get('contours_recall', 0):.4f}, AUC-PR={val_metrics.get('contours_auc_pr', 0):.4f})"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("WspółczynnikUczenia", current_lr, epoch)

        # Krok schedulera
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                # Dla ReduceLROnPlateau, potrzebujemy konkretnej metryki
                metric_for_scheduler_key = metric_for_checkpoint.replace(
                    "val_", ""
                )  # Usuń 'val_' z nazwy
                if "loss" in metric_for_checkpoint.lower():  # Jeśli metryka to strata
                    metric_val_for_scheduler = val_losses.get(metric_for_scheduler_key)
                else:  # Jeśli metryka to np. F1
                    metric_val_for_scheduler = val_metrics.get(metric_for_scheduler_key)

                if metric_val_for_scheduler is not None:
                    scheduler.step(metric_val_for_scheduler)
                else:
                    # Fallback, jeśli klucz metryki nie został znaleziony
                    print(
                        f"OSTRZEŻENIE: Metryka '{metric_for_checkpoint}' (klucz dla schedulera: {metric_for_scheduler_key}) nie znaleziona dla kroku schedulera. Używam val_losses['total_loss']."
                    )
                    scheduler.step(val_losses["total_loss"])
            else:
                # Dla innych schedulerów, które nie wymagają metryki (np. StepLR)
                scheduler.step()

            if (
                optimizer.param_groups[0]["lr"] != current_lr
            ):  # Sprawdzenie, czy LR się zmienił
                print(
                    f"  Współczynnik uczenia zmieniony na: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # Logika zapisywania najlepszego modelu i wczesnego zatrzymania
        # Określenie, czy monitorujemy stratę (dążymy do minimum) czy inną metrykę (dążymy do maksimum)
        is_loss_metric_for_checkpoint = "loss" in metric_for_checkpoint.lower()

        # Pobranie bieżącej wartości monitorowanej metryki
        if is_loss_metric_for_checkpoint:
            current_metric_value = val_losses.get(
                metric_for_checkpoint.replace("val_", "")
            )
        else:
            current_metric_value = val_metrics.get(
                metric_for_checkpoint.replace("val_", "")
            )

        # Fallback, jeśli z jakiegoś powodu metryka nie została znaleziona
        if current_metric_value is None:
            print(
                f"OSTRZEŻENIE: Monitorowana metryka '{metric_for_checkpoint}' nie została znaleziona w wynikach walidacji. Używam 'val_loss_total' jako domyślnej."
            )
            current_metric_value = val_losses["total_loss"]
            metric_for_checkpoint_effective = (
                "val_loss_total"  # Użyj pełnej nazwy dla jasności
            )
            is_loss_metric_for_checkpoint = True  # Zaktualizuj flagę
            # Zresetuj best_metric_value, jeśli tryb się zmienił (np. z max na min)
            if best_metric_value == float("-inf") and is_loss_metric_for_checkpoint:
                best_metric_value = float("inf")

        improved = False
        if is_loss_metric_for_checkpoint:  # Dążymy do minimum
            if current_metric_value < best_metric_value:
                best_metric_value = current_metric_value
                improved = True
        else:  # Dążymy do maksimum
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                improved = True

        if improved:
            print(
                f"  Poprawa metryki '{metric_for_checkpoint}': {best_metric_value:.4f}. Zapisywanie modelu do {best_model_path}..."
            )
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0  # Reset licznika braku poprawy
        else:
            epochs_no_improve += 1
            print(
                f"  Brak poprawy metryki '{metric_for_checkpoint}' przez {epochs_no_improve} epok (najlepsza: {best_metric_value:.4f}, obecna: {current_metric_value:.4f})."
            )

        epoch_duration = time.time() - epoch_start_time
        print(f"  Czas trwania epoki: {epoch_duration:.2f}s")

        # Sprawdzenie warunku wczesnego zatrzymania
        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\nZastosowano wczesne zatrzymanie! Metryka '{metric_for_checkpoint}' nie poprawiła się przez {early_stopping_patience} epok."
            )
            stopped_epoch = epoch + 1  # Zapisz, w której epoce faktycznie zatrzymano
            break

    writer.close()  # Zamknięcie loggera TensorBoard
    print(f"\nTrening zakończony po {stopped_epoch} epokach.")

    # Załadowanie wag najlepszego modelu
    print(
        f"Ładowanie najlepszego modelu z '{best_model_path}' (metryka '{metric_for_checkpoint}': {best_metric_value:.4f})"
    )
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(
            f"OSTRZEŻENIE: Nie znaleziono pliku najlepszego modelu w '{best_model_path}'. Model nie został załadowany z checkpointu."
        )

    return history
