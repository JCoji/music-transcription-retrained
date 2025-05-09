from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from src.losses import binary_focal_loss


def train_one_epoch(model, train_loader, optimizer, device, gamma=2.0, alpha=None):
    model.train()
    total_loss = 0.0
    task_losses = defaultdict(float)

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        # Przenoszenie danych na urządzenie
        features = batch['features'].to(device)
        notes_target = batch['notes'].to(device)
        onsets_target = batch['onsets'].to(device)
        contours_target = batch['contours'].to(device)
        mask = batch['mask'].to(device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)

        # Obliczanie strat dla każdego zadania z uwzględnieniem maski
        loss_contours = binary_focal_loss(
            outputs['contours'], contours_target,
            gamma=gamma, alpha=alpha,
            mask=mask, reduction='mean',
            task_name='contours'
        )
        loss_notes = binary_focal_loss(
            outputs['notes'], notes_target,
            gamma=gamma, alpha=alpha,
            mask=mask, reduction='mean',
            task_name='notes'
        )
        loss_onsets = binary_focal_loss(
            outputs['onsets'], onsets_target,
            gamma=gamma, alpha=alpha,
            mask=mask, reduction='mean',
            task_name='onsets'
        )
        # Łączna strata
        total_batch_loss = loss_contours + loss_notes + loss_onsets
        # Backward pass
        total_batch_loss.backward()
        optimizer.step()

        # Aktualizacja statystyk
        total_loss += total_batch_loss.item()
        task_losses['contours'] += loss_contours.item()
        task_losses['notes'] += loss_notes.item()
        task_losses['onsets'] += loss_onsets.item()
        # Aktualizacja progress bara
        progress_bar.set_postfix({
            'loss': total_batch_loss.item(),
            'notes': loss_notes.item(),
            'onsets': loss_onsets.item(),
            'contours': loss_contours.item()
        })

    # Obliczanie średnich strat
    num_batches = len(train_loader)
    total_loss /= num_batches
    for task in task_losses:
        task_losses[task] /= num_batches

    return total_loss, dict(task_losses)


def evaluate(model, data_loader, device, gamma=2.0, alpha=None):
    model.eval()
    total_loss = 0.0
    task_losses = defaultdict(float)

    # Metryki
    metrics = {
        'notes': defaultdict(list),
        'onsets': defaultdict(list),
        'contours': defaultdict(list)
    }

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            features = batch['features'].to(device)
            notes_target = batch['notes'].to(device)
            onsets_target = batch['onsets'].to(device)
            contours_target = batch['contours'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            outputs = model(features)

            # Obliczanie strat
            loss_contours = binary_focal_loss(
                outputs['contours'], contours_target,
                gamma=gamma, alpha=alpha,
                mask=mask, reduction='mean'
            )

            loss_notes = binary_focal_loss(
                outputs['notes'], notes_target,
                gamma=gamma, alpha=alpha,
                mask=mask, reduction='mean'
            )

            loss_onsets = binary_focal_loss(
                outputs['onsets'], onsets_target,
                gamma=gamma, alpha=alpha,
                mask=mask, reduction='mean'
            )

            total_batch_loss = loss_contours + loss_notes + loss_onsets

            # Aktualizacja statystyk strat
            total_loss += total_batch_loss.item()
            task_losses['contours'] += loss_contours.item()
            task_losses['notes'] += loss_notes.item()
            task_losses['onsets'] += loss_onsets.item()

            # Obliczanie metryk dla każdego zadania
            for task in ['notes', 'onsets', 'contours']:
                preds = torch.sigmoid(outputs[task]) > 0.5
                targets = batch[task].to(device) > 0.5
                valid_mask = mask.unsqueeze(-1).expand_as(targets).bool()

                # Tylko niepaddingowane pozycje
                preds = preds[valid_mask]
                targets = targets[valid_mask]

                # Dokładność
                correct = (preds == targets).sum().item()
                total = targets.numel()
                metrics[task]['accuracy'].append(correct / total)

                # Precyzja, czułość, F1 (tylko jeśli są pozytywne przykłady)
                if targets.sum() > 0:
                    true_pos = (preds & targets).sum().item()
                    pred_pos = preds.sum().item()
                    actual_pos = targets.sum().item()

                    precision = true_pos / (pred_pos + 1e-8)
                    recall = true_pos / (actual_pos + 1e-8)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

                    metrics[task]['precision'].append(precision)
                    metrics[task]['recall'].append(recall)
                    metrics[task]['f1'].append(f1)

    # Obliczanie średnich
    num_batches = len(data_loader)
    total_loss /= num_batches
    for task in task_losses:
        task_losses[task] /= num_batches

    # Średnie metryki
    avg_metrics = {}
    for task in metrics:
        avg_metrics[task] = {
            k: np.mean(v) if v else 0.0
            for k, v in metrics[task].items()
        }

    return total_loss, dict(task_losses), avg_metrics


def run_training(
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-4,
        epochs=50,
        gamma=2.0,
        alpha_notes=0.75,
        alpha_onsets=0.9,
        alpha_contours=0.5,
        patience=5,
        output_dir="models"
):
    # Przygotowanie katalogu wyjściowego
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_model_path = output_path / "best_model.pth"

    # Konfiguracja treningu
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    # Historia treningu
    history = {
        'train_loss': [],
        'val_loss': [],
        'notes_f1': [],
        'onsets_f1': [],
        'contours_f1': []
    }

    # Pętla treningowa
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Trening
        train_loss, train_task_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            gamma=gamma,
            alpha={
                'notes': alpha_notes,
                'onsets': alpha_onsets,
                'contours': alpha_contours
            }
        )

        # Walidacja
        val_loss, val_task_losses, val_metrics = evaluate(
            model, val_loader, device,
            gamma=gamma,
            alpha={
                'notes': alpha_notes,
                'onsets': alpha_onsets,
                'contours': alpha_contours
            }
        )

        # Aktualizacja scheduler'a
        scheduler.step(val_loss)

        # Zapisywanie historii
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['notes_f1'].append(val_metrics['notes'].get('f1', 0))
        history['onsets_f1'].append(val_metrics['onsets'].get('f1', 0))
        history['contours_f1'].append(val_metrics['contours'].get('f1', 0))

        # Wyświetlanie wyników
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Notes: {train_task_losses['notes']:.4f}")
        print(f"  Onsets: {train_task_losses['onsets']:.4f}")
        print(f"  Contours: {train_task_losses['contours']:.4f}")

        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"  Notes F1: {val_metrics['notes'].get('f1', 0):.4f}")
        print(f"  Onsets F1: {val_metrics['onsets'].get('f1', 0):.4f}")
        print(f"  Contours F1: {val_metrics['contours'].get('f1', 0):.4f}")

        # Zapisywanie najlepszego modelu
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print("\n⭐ New best model saved!")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{patience})")

            # Wczesne zatrzymanie
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    # Załadowanie najlepszego modelu
    model.load_state_dict(torch.load(best_model_path))
    print(f"\nTraining completed. Best model saved to {best_model_path}")

    return model, history