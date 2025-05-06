import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm


def masked_bce_loss(pred, target, mask):
    if pred.size(1) != target.size(1):
        min_len = min(pred.size(1), target.size(1))
        pred = pred[:, :min_len]
        target = target[:, :min_len]
        mask = mask[:, :min_len]

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(pred, target)
    mask = mask.unsqueeze(-1).expand_as(loss)
    loss = loss * mask
    return loss.sum() / mask.sum()


def masked_mse_loss(pred, target, mask):
    min_time = min(pred.shape[1], target.shape[1])
    pred = pred[:, :min_time]
    target = target[:, :min_time]
    mask = mask[:, :min_time]

    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    mask = mask.unsqueeze(-1).expand_as(loss)
    loss = loss * mask
    return loss.sum() / mask.sum()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        features = batch["features"].to(device)
        notes = batch["notes"].to(device)
        contours = batch["contours"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss_notes = masked_bce_loss(outputs["notes"], notes, mask)
        loss_contours = masked_mse_loss(outputs["contours"], contours, mask)

        loss = loss_notes + loss_contours
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            features = batch["features"].to(device)
            notes = batch["notes"].to(device)
            contours = batch["contours"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(features)

            loss_notes = masked_bce_loss(outputs["notes"], notes, mask)
            loss_contours = masked_mse_loss(outputs["contours"], contours, mask)
            loss = loss_notes + loss_contours

            total_loss += loss.item()

    return total_loss / len(loader)


def run_training(
        model,
        train_loader,
        val_loader,
        device,
        epochs=20,
        lr=1e-3,
        save_path="best_model.pt"
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to {save_path}")
