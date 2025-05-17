# src/train.py
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
# Upewnij się, że wizualizacja jest dostępna
from src.visualization import plot_sample_data_with_predictions


def calculate_metrics_counts(
    model_output: dict, batch: dict, device: torch.device, thresholds: dict
) -> dict:
    notes_target = batch["notes"].to(device).float()
    onsets_target = batch["onsets"].to(device).float()
    contours_target = batch["contours"].to(device).float()
    mask = batch["mask"].to(device)

    notes_logits = model_output["notes"]
    onsets_logits = model_output["onsets"]
    contours_logits = model_output["contours"]

    notes_pred = (torch.sigmoid(notes_logits) > thresholds["notes"]).float()
    onsets_pred = (torch.sigmoid(onsets_logits) > thresholds["onsets"]).float()
    contours_pred = (
        torch.sigmoid(contours_logits)
        > thresholds.get("contours", 0.5)  # Użyj .get z domyślną wartością
    ).float()

    mask_notes = mask.unsqueeze(-1).expand_as(notes_target)
    mask_contours = mask.unsqueeze(-1).expand_as(contours_target)

    notes_target_masked = notes_target * mask_notes
    onsets_target_masked = onsets_target * mask_notes
    contours_target_masked = contours_target * mask_contours

    notes_pred_masked = notes_pred * mask_notes
    onsets_pred_masked = onsets_pred * mask_notes
    contours_pred_masked = contours_pred * mask_contours

    tp_notes = (notes_pred_masked * notes_target_masked).sum()
    fp_notes = (notes_pred_masked * (1 - notes_target_masked)).sum()
    fn_notes = ((1 - notes_pred_masked) * notes_target_masked).sum()

    tp_onsets = (onsets_pred_masked * onsets_target_masked).sum()
    fp_onsets = (onsets_pred_masked * (1 - onsets_target_masked)).sum()
    fn_onsets = ((1 - onsets_pred_masked) * onsets_target_masked).sum()

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
    loss_type: str = "BCE",
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: dict = None,
) -> dict:
    model.train()
    total_loss_accum = 0.0
    notes_loss_accum = 0.0
    onsets_loss_accum = 0.0
    contours_loss_accum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Trening", leave=False):
        features = batch["features"].to(device)
        optimizer.zero_grad()
        output_logits = model(features)
        losses = calculate_loss(
            model_output=output_logits,
            batch=batch,
            device=device,
            pos_weights=pos_weights,
            task_weights=task_weights,
            loss_type=loss_type,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
        )
        total_loss = losses["total_loss"]
        total_loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip_val)
        optimizer.step()

        total_loss_accum += losses["total_loss"].item()
        notes_loss_accum += losses["loss_notes"].item()
        onsets_loss_accum += losses["loss_onsets"].item()
        contours_loss_accum += losses["loss_contours"].item()
        num_batches += 1

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
    loss_type: str = "BCE",
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: dict = None,
    plot_every_n_epochs: int = 15,
    sample_to_plot_idx: int = 0,
    thresholds: dict = None,
) -> dict:
    model.eval()
    total_loss_accum = 0.0
    notes_loss_accum = 0.0
    onsets_loss_accum = 0.0
    contours_loss_accum = 0.0

    all_y_true = {"notes": [], "onsets": [], "contours": []}
    all_y_scores = {"notes": [], "onsets": [], "contours": []}

    first_batch_for_plot = None
    first_batch_logits_for_plot = None

    thresholds_to_use = (
        thresholds
        if thresholds is not None
        else {
            "notes": 0.5,
            "onsets": 0.5,
            "contours": 0.5,
        }  # Domyślne progi, jeśli nie ma innych
    )
    # W oryginalnym kodzie notatnika, progi były znajdowane później.
    # Dla spójności, jeśli `thresholds` nie są podane, używamy tych domyślnych 0.5.
    # W `train_model`, `evaluate_epoch` jest wywoływane z `thresholds=None` podczas walidacji.
    # Optymalne progi są obliczane PO treningu i używane do finalnej ewaluacji.

    metrics_counts = {
        task: {"tp": 0, "fp": 0, "fn": 0} for task in thresholds_to_use.keys()
    }
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Ewaluacja", leave=False)
        ):
            features = batch["features"].to(device)
            output_logits = model(features)

            if (
                batch_idx == 0
                and plot_every_n_epochs
                > 0  # Plot only if plot_every_n_epochs is positive
                and (
                    epoch_num % plot_every_n_epochs == 0
                    or epoch_num == 1
                    or epoch_num == 0
                )  # epoch_num 0 for initial eval
            ):
                first_batch_for_plot = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        first_batch_for_plot[k] = v.cpu().clone()
                    else:
                        first_batch_for_plot[k] = v
                first_batch_logits_for_plot = {
                    k: v.cpu().clone() for k, v in output_logits.items()
                }

            losses = calculate_loss(
                model_output=output_logits,
                batch=batch,
                device=device,
                pos_weights=pos_weights,
                task_weights=task_weights,
                loss_type=loss_type,
                focal_loss_gamma=focal_loss_gamma,
                focal_loss_alpha=focal_loss_alpha,
            )
            total_loss_accum += losses["total_loss"].item()
            notes_loss_accum += losses["loss_notes"].item()
            onsets_loss_accum += losses["loss_onsets"].item()
            contours_loss_accum += losses["loss_contours"].item()

            # Oblicz metryki tylko jeśli thresholds_to_use jest zdefiniowane poprawnie
            if thresholds_to_use:
                batch_metrics = calculate_metrics_counts(
                    output_logits, batch, device, thresholds=thresholds_to_use
                )
                for (
                    task
                ) in (
                    metrics_counts.keys()
                ):  # Iteruj po kluczach zdefiniowanych w metrics_counts
                    if task in batch_metrics:
                        for count_type in metrics_counts[task].keys():
                            metrics_counts[task][count_type] += batch_metrics[task][
                                count_type
                            ]
            # Zbieranie danych do AUC-PR
            for task_key in ["notes", "onsets", "contours"]:
                if (
                    task_key in output_logits and task_key in batch
                ):  # Upewnij się, że klucze istnieją
                    targets_b_t_f = batch[task_key].to(device)
                    scores_b_t_f = torch.sigmoid(output_logits[task_key]).to(device)
                    for i in range(targets_b_t_f.shape[0]):  # Iteruj po batch size
                        # Użyj feature_lengths do określenia rzeczywistej długości sekwencji
                        length = (
                            batch.get("feature_lengths")[i].item()
                            if batch.get("feature_lengths") is not None
                            else targets_b_t_f.shape[1]
                        )

                        true_sample_active_frames = (
                            targets_b_t_f[i, :length, :].flatten().cpu().numpy()
                        )
                        score_sample_active_frames = (
                            scores_b_t_f[i, :length, :].flatten().cpu().numpy()
                        )
                        all_y_true[task_key].extend(true_sample_active_frames.tolist())
                        all_y_scores[task_key].extend(
                            score_sample_active_frames.tolist()
                        )
            num_batches += 1

    if (
        plot_every_n_epochs > 0
        and (epoch_num % plot_every_n_epochs == 0 or epoch_num == 1 or epoch_num == 0)
        and first_batch_for_plot is not None
    ):
        print(f"Generowanie wizualizacji dla epoki {epoch_num}...")
        preds_for_plot_probs = {
            task: torch.sigmoid(
                first_batch_logits_for_plot[task][sample_to_plot_idx]
            ).numpy()
            for task in ["notes", "onsets", "contours"]
            if task in first_batch_logits_for_plot
        }
        plot_dir = Path(checkpoint_dir) / "plots" / model_name_stem
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_dir / f"epoch_{epoch_num}_sample_{sample_to_plot_idx}.png"

        # Użyj progu dla nut, jeśli dostępny, inaczej domyślny 0.5
        vis_threshold = (
            thresholds_to_use.get("notes", 0.5) if thresholds_to_use else 0.5
        )

        plot_sample_data_with_predictions(
            batch=first_batch_for_plot,
            predictions_probs=preds_for_plot_probs,
            sample_idx=sample_to_plot_idx,
            save_path=str(fig_path),
            threshold=vis_threshold,
        )
        print(f"Wizualizacja zapisana w {fig_path}")

    avg_losses = {
        "total_loss": total_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_notes": notes_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_onsets": onsets_loss_accum / num_batches if num_batches > 0 else 0.0,
        "loss_contours": contours_loss_accum / num_batches if num_batches > 0 else 0.0,
    }

    final_metrics = {}
    if thresholds_to_use:  # Oblicz P,R,F1 tylko jeśli są progi
        for task, counts in metrics_counts.items():
            p, r, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
            final_metrics[f"{task}_precision"] = p
            final_metrics[f"{task}_recall"] = r
            final_metrics[f"{task}_f1"] = f1

    auc_pr_scores = {}
    for task in ["notes", "onsets", "contours"]:
        if (
            task in all_y_true
            and len(all_y_true[task]) > 0
            and len(all_y_scores[task])
            == len(all_y_true[task])  # Dodatkowe sprawdzenie
            and len(np.unique(all_y_true[task]))
            > 1  # Potrzebne co najmniej dwie klasy do AUC
        ):
            try:
                precision_c, recall_c, _ = precision_recall_curve(
                    all_y_true[task], all_y_scores[task]
                )
                auc_pr_value = auc(recall_c, precision_c)
                auc_pr_scores[f"{task}_auc_pr"] = (
                    auc_pr_value if not np.isnan(auc_pr_value) else 0.0
                )
            except (
                ValueError
            ) as e:  # Łapanie błędu jeśli np. tylko jedna klasa w y_true
                print(f"Błąd przy obliczaniu AUC-PR dla {task}: {e}")
                auc_pr_scores[f"{task}_auc_pr"] = 0.0
        else:
            auc_pr_scores[f"{task}_auc_pr"] = 0.0
            # Zmniejszono gadatliwość ostrzeżeń
            if epoch_num > 1 and (
                not all_y_true[task] or len(np.unique(all_y_true[task])) <= 1
            ):
                print(
                    f"OSTRZEŻENIE: Nie można obliczyć AUC-PR dla zadania '{task}' w epoce {epoch_num} (brak danych lub jedna klasa)."
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
    loss_type: str = "BCE",
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: dict = None,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "best_model.pt",
    early_stopping_patience: int = 10,
    metric_for_checkpoint: str = "val_onsets_f1",
    plot_every_n_epochs: int = 15,
    sample_to_plot_idx: int = 0,
):
    model_name_stem = model_name.replace(".pt", "")
    log_path = Path(checkpoint_dir) / "logs" / model_name_stem
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"Logi TensorBoard będą zapisywane w: {log_path}")
    print(f"Używana funkcja straty: {loss_type.upper()}")
    if loss_type.upper() == "FOCAL":
        print(f"  Gamma dla Focal Loss: {focal_loss_gamma}")
        if focal_loss_alpha:
            print(f"  Alpha (specyficzne) dla Focal Loss: {focal_loss_alpha}")
        else:
            print(
                f"  Alpha dla Focal Loss będzie bazować na obliczonych/przekazanych 'pos_weights'."
            )

    best_metric_value = float("-inf")
    if "loss" in metric_for_checkpoint.lower():
        best_metric_value = float("inf")
    elif (
        scheduler and hasattr(scheduler, "mode") and scheduler.mode == "min"
    ):  # np. ReduceLROnPlateau z mode='min'
        best_metric_value = float("inf")

    epochs_no_improve = 0
    stopped_epoch = num_epochs

    history = {
        "train_loss_total": [],
        "val_loss_total": [],
        "val_loss_notes": [],
        "val_loss_onsets": [],
        "val_loss_contours": [],
        # Inicjalizuj wszystkie klucze metryk, których oczekujesz
        "val_notes_f1": [],
        "val_notes_precision": [],
        "val_notes_recall": [],
        "val_notes_auc_pr": [],
        "val_onsets_f1": [],
        "val_onsets_precision": [],
        "val_onsets_recall": [],
        "val_onsets_auc_pr": [],
        "val_contours_f1": [],
        "val_contours_precision": [],
        "val_contours_recall": [],
        "val_contours_auc_pr": [],
    }

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_path / model_name

    print(f"Rozpoczynanie treningu na maksymalnie {num_epochs} epok...")
    print(
        f"Najlepszy model będzie zapisywany na podstawie metryki: '{metric_for_checkpoint}'"
    )
    print(f"Wczesne zatrzymanie po {early_stopping_patience} epokach bez poprawy.")

    # Domyślne progi dla ewaluacji w trakcie treningu (jeśli `thresholds` to None)
    # Te progi są używane do obliczania P, R, F1 podczas walidacji epokowej.
    # Optymalne progi są znajdowane dopiero po zakończeniu treningu.
    # W oryginalnym kodzie evaluate_epoch miało hardkodowane progi, jeśli `thresholds` było None.
    # Możemy je tu zdefiniować lub przekazać `None` i pozwolić `evaluate_epoch` użyć swoich domyślnych.
    # Dla spójności, jeśli `evaluate_epoch` ma swoje domyślne, to nie trzeba tu definiować.
    # W `evaluate_epoch` domyślne progi to 0.5 dla każdego zadania.
    # Jeśli chcemy inne domyślne dla walidacji, można je tu ustawić i przekazać.
    # Na razie zostawiamy przekazywanie `thresholds=None` do `evaluate_epoch` podczas walidacji.

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoka {epoch + 1}/{num_epochs} ---")

        train_losses_dict = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weights,
            task_weights,
            gradient_clip_val,
            loss_type=loss_type,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
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

        eval_results = evaluate_epoch(
            model,
            val_loader,
            device,
            pos_weights,
            task_weights,
            epoch_num=epoch + 1,  # epoch + 1, bo epoki są numerowane od 1
            checkpoint_dir=checkpoint_dir,
            model_name_stem=model_name_stem,
            loss_type=loss_type,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
            plot_every_n_epochs=plot_every_n_epochs,
            sample_to_plot_idx=sample_to_plot_idx,
            thresholds=None,  # Użyj domyślnych progów w evaluate_epoch (0.5) lub tych przekazanych
        )
        val_losses = eval_results["loss"]
        val_metrics = eval_results["metrics"]

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

        for task in ["notes", "onsets", "contours"]:
            for metric_type in ["f1", "precision", "recall", "auc_pr"]:
                key = f"{task}_{metric_type}"
                history_key = f"val_{key}"  # np. 'val_notes_f1'
                metric_val = val_metrics.get(
                    key, 0.0
                )  # Domyślnie 0.0, jeśli metryka nie została obliczona
                history[history_key].append(
                    metric_val
                )  # Używamy bezpośrednio klucza, który już jest w `history`
                writer.add_scalar(
                    f"{metric_type.replace('_', '-').capitalize()}/walidacja_{task}",
                    metric_val,
                    epoch,
                )

        print(
            f"  Strata Wal.: {val_losses['total_loss']:.4f} (N: {val_losses['loss_notes']:.4f}, O: {val_losses['loss_onsets']:.4f}, C: {val_losses['loss_contours']:.4f})"
        )
        print(
            f"  Metryki Wal.: Nuty F1={val_metrics.get('notes_f1', 0):.4f} (P={val_metrics.get('notes_precision', 0):.4f}, R={val_metrics.get('notes_recall', 0):.4f}, AUC-PR={val_metrics.get('notes_auc_pr', 0):.4f})"
        )
        print(
            f"                Początki F1={val_metrics.get('onsets_f1', 0):.4f} (P={val_metrics.get('onsets_precision', 0):.4f}, R={val_metrics.get('onsets_recall', 0):.4f}, AUC-PR={val_metrics.get('onsets_auc_pr', 0):.4f})"
        )
        print(
            f"                Kontury F1={val_metrics.get('contours_f1', 0):.4f} (P={val_metrics.get('contours_precision', 0):.4f}, R={val_metrics.get('contours_recall', 0):.4f}, AUC-PR={val_metrics.get('contours_auc_pr', 0):.4f})"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("WspółczynnikUczenia", current_lr, epoch)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                # Usuń 'val_' z nazwy metryki dla schedulera
                metric_for_scheduler_key_base = metric_for_checkpoint.replace(
                    "val_", ""
                )
                if "loss" in metric_for_checkpoint.lower():
                    # np. val_loss_total -> loss_total
                    metric_val_for_scheduler = val_losses.get(
                        metric_for_scheduler_key_base
                    )
                else:
                    # np. val_onsets_f1 -> onsets_f1
                    metric_val_for_scheduler = val_metrics.get(
                        metric_for_scheduler_key_base
                    )

                if metric_val_for_scheduler is not None:
                    scheduler.step(metric_val_for_scheduler)
                else:
                    print(
                        f"OSTRZEŻENIE: Metryka '{metric_for_checkpoint}' (klucz bazowy dla schedulera: '{metric_for_scheduler_key_base}') nie znaleziona. Używam val_losses['total_loss']."
                    )
                    scheduler.step(val_losses["total_loss"])  # Fallback
            else:  # Dla innych schedulerów (np. StepLR)
                scheduler.step()

            if optimizer.param_groups[0]["lr"] != current_lr:
                print(
                    f"  Współczynnik uczenia zmieniony na: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # Logika zapisywania najlepszego modelu i wczesnego zatrzymania
        is_loss_metric_for_checkpoint = "loss" in metric_for_checkpoint.lower()
        metric_key_for_eval = metric_for_checkpoint.replace("val_", "")

        if is_loss_metric_for_checkpoint:
            current_metric_value = val_losses.get(metric_key_for_eval)
        else:
            current_metric_value = val_metrics.get(metric_key_for_eval)

        if current_metric_value is None:
            print(
                f"OSTRZEŻENIE: Monitorowana metryka '{metric_for_checkpoint}' (klucz: '{metric_key_for_eval}') nie znaleziona. Używam 'val_loss_total'."
            )
            current_metric_value = val_losses["total_loss"]
            # Zaktualizuj, jak interpretujemy best_metric_value, jeśli przeszliśmy na stratę
            if (
                not is_loss_metric_for_checkpoint
            ):  # Jeśli oryginalnie nie była to strata
                is_loss_metric_for_checkpoint = True
                if best_metric_value == float(
                    "-inf"
                ):  # Jeśli best_metric_value było ustawione na max
                    best_metric_value = float("inf")  # Zmień na min

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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"  Brak poprawy metryki '{metric_for_checkpoint}' przez {epochs_no_improve} epok (najlepsza: {best_metric_value:.4f}, obecna: {current_metric_value:.4f})."
            )

        epoch_duration = time.time() - epoch_start_time
        print(f"  Czas trwania epoki: {epoch_duration:.2f}s")

        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\nZastosowano wczesne zatrzymanie! Metryka '{metric_for_checkpoint}' nie poprawiła się przez {early_stopping_patience} epok."
            )
            stopped_epoch = epoch + 1
            break

    writer.close()
    print(f"\nTrening zakończony po {stopped_epoch} epokach.")
    print(
        f"Ładowanie najlepszego modelu z '{best_model_path}' (monitorowana metryka '{metric_for_checkpoint}': {best_metric_value:.4f})"
    )
    if os.path.exists(best_model_path):
        # Upewnij się, że model jest na odpowiednim urządzeniu przed załadowaniem stanu
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(
            f"OSTRZEŻENIE: Nie znaleziono pliku najlepszego modelu w '{best_model_path}'. Model może nie być w najlepszym stanie."
        )
    return history
