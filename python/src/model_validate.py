import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from src.config import FREQ_BINS_NOTES, N_FREQ_BINS_NOTES


def evaluate_model_classification(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            notes = batch["notes"].cpu().numpy()
            mask = batch["mask"].cpu().numpy()

            outputs = model(features)
            pred_notes = torch.sigmoid(outputs["notes"]).cpu().numpy()
            pred_bin = (pred_notes > threshold).astype(int)

            # Ensure predictions and targets have the same length
            for p, t, m in zip(pred_bin, notes, mask):
                valid_len = int(m.sum())
                # Trim both to the same length
                min_len = min(p.shape[0], t.shape[0], valid_len)
                p_flat = p[:min_len].flatten()
                t_flat = t[:min_len].flatten()

                all_preds.extend(p_flat)
                all_targets.extend(t_flat)

    # Check if we have any samples to evaluate
    if not all_targets:
        print("No valid samples to evaluate!")
        return

    # Calculate metrics
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall:    {recall:.4f}")
    print(f"🎯 F1 Score:  {f1:.4f}")


def visualize_predictions(model, dataset, device, index=0):
    model.eval()
    sample = dataset[index]
    features = sample["features"].unsqueeze(0).to(device)  # [1, T, F]

    # Tworzymy ground truth z note_indices i note_values
    notes_true = torch.zeros((features.shape[1], N_FREQ_BINS_NOTES), dtype=torch.float32)
    if "note_indices" in sample and len(sample["note_indices"]) > 0:
        indices = torch.tensor(sample["note_indices"], dtype=torch.long)
        values = torch.tensor(sample["note_values"], dtype=torch.float32)
        notes_true[indices[:, 0], indices[:, 1]] = values

    with torch.no_grad():
        output = model(features)
        pred_notes = torch.sigmoid(output["notes"]).squeeze(0).cpu()  # [T, F]
        pred_notes_bin = (pred_notes > 0.5).float()

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].imshow(notes_true.T, aspect='auto', origin='lower', cmap='hot')
    axs[0].set_title("Ground Truth (Adnotacje nut)")
    axs[0].set_ylabel("Bin częstotliwości")

    axs[1].imshow(pred_notes_bin.T, aspect='auto', origin='lower', cmap='hot')
    axs[1].set_title("Predykcja modelu (Binarna mapa nut)")
    axs[1].set_ylabel("Bin częstotliwości")
    axs[1].set_xlabel("Czas (klatki)")

    plt.tight_layout()
    plt.show()
