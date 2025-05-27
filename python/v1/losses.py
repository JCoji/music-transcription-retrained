# v1/losses.py

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .focal_loss import FocalLoss  # Import FocalLoss


def calculate_pos_weights_for_dataset(
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_freq_bins_notes: int,
    num_freq_bins_contours: int,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Oblicza wagi 'pos_weight' dla funkcji straty BCEWithLogitsLoss na podstawie całego zbioru danych.
    Może być również używane jako 'pos_weight' (interpretowane jako alpha dla klasy pozytywnej) dla Focal Loss.

    Wagi te pomagają w radzeniu sobie ze niezbalansowanymi klasami w zadaniach transkrypcji.
    Iteruje po dataloaderze, zliczając pozytywne i negatywne wystąpienia dla każdej klasy
    (kosza częstotliwości) w zadaniach predykcji nut, onsetów i konturów, uwzględniając maskę.

    Args:
        dataloader: DataLoader dla zbioru danych (np. treningowego).
        device: Urządzenie (cpu/cuda), na którym będą wykonywane obliczenia.
        num_freq_bins_notes: Liczba koszy częstotliwości dla nut i onsetów.
        num_freq_bins_contours: Liczba koszy częstotliwości dla konturów.
        epsilon: Mała wartość dodawana do mianownika, aby uniknąć dzielenia przez zero.

    Returns:
        Słownik zawierający obliczone tensory 'pos_weight' dla zadań 'notes', 'onsets', 'contours'.
    """
    counts_positive = {
        "notes": torch.zeros(num_freq_bins_notes, device=device),
        "onsets": torch.zeros(num_freq_bins_notes, device=device),
        "contours": torch.zeros(num_freq_bins_contours, device=device),
    }
    total_active_frames_per_bin = {
        "notes": torch.zeros(num_freq_bins_notes, device=device),
        "onsets": torch.zeros(num_freq_bins_notes, device=device),
        "contours": torch.zeros(num_freq_bins_contours, device=device),
    }

    print("Obliczanie wag pos_weight/alpha na podstawie zbioru danych...")
    for batch in tqdm(dataloader, desc="Analiza batchy"):
        notes_target = batch["notes"].to(device)
        onsets_target = batch["onsets"].to(device)
        contours_target = batch["contours"].to(device)
        mask = batch["mask"].to(device)

        mask_notes = mask.unsqueeze(-1).expand_as(notes_target)
        mask_contours = mask.unsqueeze(-1).expand_as(contours_target)

        counts_positive["notes"] += (notes_target * mask_notes).sum(dim=(0, 1))
        total_active_frames_per_bin["notes"] += mask_notes.sum(dim=(0, 1))

        counts_positive["onsets"] += (onsets_target * mask_notes).sum(dim=(0, 1))
        total_active_frames_per_bin["onsets"] += mask_notes.sum(dim=(0, 1))

        counts_positive["contours"] += (contours_target * mask_contours).sum(dim=(0, 1))
        total_active_frames_per_bin["contours"] += mask_contours.sum(dim=(0, 1))

    pos_weights_calculated: Dict[str, torch.Tensor] = {}

    for task in ("notes", "onsets", "contours"):
        positives = counts_positive[task]
        total = total_active_frames_per_bin[task]
        negatives = total - positives

        if torch.any(negatives < 0):
            print(
                f"OSTRZEŻENIE: Wykryto ujemne wartości w 'counts_negative' dla zadania "
                f"{task}. Sprawdź maskę i dane."
            )
            negatives = torch.clamp(negatives, min=0)

        weights = negatives / (positives + epsilon)
        weights = torch.clamp(weights, max=110.0)
        weights[torch.isinf(weights) | torch.isnan(weights)] = 1.0

        pos_weights_calculated[task] = weights
        print(
            f"Wagi dla zadania '{task}': "
            f"min={weights.min():.2f}, max={weights.max():.2f}, średnia={weights.mean():.2f}"
        )

    return pos_weights_calculated


def calculate_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    pos_weights: Dict[str, torch.Tensor],
    task_weights: Dict[str, float],
    loss_type: str = "BCE",
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Oblicza łączną ważoną stratę dla batcha.

    Args:
        model_output: Słownik z logitami z modelu dla 'notes', 'onsets', 'contours'.
        batch: Słownik z danymi wejściowymi, zawierający targety i 'mask'.
        device: Urządzenie (cpu/cuda).
        pos_weights: Słownik z tensorami pos_weight. Dla BCE używane bezpośrednio.
            Dla FocalLoss, jeśli `focal_loss_alpha` nie jest podane,
            `pos_weights` jest przekazywane do `FocalLoss` jako argument `pos_weight`.
        task_weights: Słownik z wagami float dla każdego zadania.
        loss_type: Rodzaj funkcji straty ('BCE' lub 'Focal').
        focal_loss_gamma: Parametr gamma dla Focal Loss.
        focal_loss_alpha: Słownik z wartościami alpha dla Focal Loss dla każdego zadania.

    Returns:
        Słownik zawierający 'total_loss' oraz straty dla poszczególnych zadań.
    """
    notes_target = batch["notes"].to(device)
    onsets_target = batch["onsets"].to(device)
    contours_target = batch["contours"].to(device)
    mask = batch["mask"].to(device)

    notes_logits = model_output["notes"]
    onsets_logits = model_output["onsets"]
    contours_logits = model_output["contours"]

    losses_unreduced: Dict[str, torch.Tensor] = {}

    for task_key, logits, target in (
        ("notes", notes_logits, notes_target),
        ("onsets", onsets_logits, onsets_target),
        ("contours", contours_logits, contours_target),
    ):
        current_pos_weight = pos_weights[task_key].to(device)
        current_focal_alpha = focal_loss_alpha.get(task_key) if focal_loss_alpha else None

        if loss_type.upper() == "FOCAL":
            loss_fn = FocalLoss(
                alpha=current_focal_alpha,
                gamma=focal_loss_gamma,
                reduction="none",
                pos_weight=(current_pos_weight if current_focal_alpha is None else None),
            )
        elif loss_type.upper() == "BCE":
            loss_fn = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=current_pos_weight,
            )
        else:
            raise ValueError(f"Nieznany typ funkcji straty: {loss_type}")

        losses_unreduced[task_key] = loss_fn(logits, target.float())

    mask_notes = mask.unsqueeze(-1).expand_as(notes_target)
    mask_contours = mask.unsqueeze(-1).expand_as(contours_target)

    loss_notes = (
        (losses_unreduced["notes"] * mask_notes).sum()
        / (mask_notes.sum() + 1e-8)
    )
    loss_onsets = (
        (losses_unreduced["onsets"] * mask_notes).sum()
        / (mask_notes.sum() + 1e-8)
    )
    loss_contours = (
        (losses_unreduced["contours"] * mask_contours).sum()
        / (mask_contours.sum() + 1e-8)
    )

    total_loss = (
        task_weights["notes"] * loss_notes
        + task_weights["onsets"] * loss_onsets
        + task_weights["contours"] * loss_contours
    )

    return {
        "total_loss": total_loss,
        "loss_notes": loss_notes,
        "loss_onsets": loss_onsets,
        "loss_contours": loss_contours,
    }
