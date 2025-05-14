from typing import Dict

import torch
from torch import nn
from tqdm import tqdm


def calculate_pos_weights_for_dataset(
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_freq_bins_notes: int,
    num_freq_bins_contours: int,
    epsilon: float = 1e-8,
) -> dict:
    """
    Oblicza wagi 'pos_weight' dla funkcji straty BCEWithLogitsLoss na podstawie całego zbioru danych.

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
    # Inicjalizacja liczników
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

    print("Obliczanie wag pos_weight na podstawie zbioru danych...")
    for batch in tqdm(dataloader, desc="Analiza batchy"):
        notes_target = batch["notes"].to(device)
        onsets_target = batch["onsets"].to(device)
        contours_target = batch["contours"].to(device)
        mask = batch["mask"].to(device)

        # Rozszerzenie maski do wymiarów targetów (B, T, F)
        mask_notes_expanded = mask.unsqueeze(-1).expand_as(notes_target)
        mask_contours_expanded = mask.unsqueeze(-1).expand_as(contours_target)

        # Zliczanie pozytywnych wystąpień i aktywnych ramek dla każdego kosza częstotliwości
        counts_positive["notes"] += (notes_target * mask_notes_expanded).sum(dim=(0, 1))
        total_active_frames_per_bin["notes"] += mask_notes_expanded.sum(dim=(0, 1))

        counts_positive["onsets"] += (onsets_target * mask_notes_expanded).sum(
            dim=(0, 1)
        )
        total_active_frames_per_bin["onsets"] += mask_notes_expanded.sum(dim=(0, 1))

        counts_positive["contours"] += (contours_target * mask_contours_expanded).sum(
            dim=(0, 1)
        )
        total_active_frames_per_bin["contours"] += mask_contours_expanded.sum(
            dim=(0, 1)
        )

    pos_weights_calculated = {}
    for task in ["notes", "onsets", "contours"]:
        # Obliczenie liczby negatywnych wystąpień
        counts_negative_task = total_active_frames_per_bin[task] - counts_positive[task]

        # Zabezpieczenie przed ujemnymi wartościami w counts_negative (co nie powinno się zdarzyć przy poprawnej masce i danych)
        if torch.any(counts_negative_task < 0):
            print(
                f"OSTRZEŻENIE: Wykryto ujemne wartości w 'counts_negative' dla zadania {task}. Sprawdź maskę i dane."
            )
            counts_negative_task = torch.clamp(counts_negative_task, min=0)

        # Obliczenie wag pos_weight: stosunek negatywnych do pozytywnych
        current_pos_weights = counts_negative_task / (counts_positive[task] + epsilon)

        # Ograniczenie maksymalnej wagi dla stabilności numerycznej
        current_pos_weights = torch.clamp(
            current_pos_weights, max=170.0
        )  # Wartość 170.0 jest przykładowa

        # Obsługa NaN lub Inf (gdy counts_positive[task] + epsilon było bliskie zeru)
        current_pos_weights[
            torch.isinf(current_pos_weights) | torch.isnan(current_pos_weights)
        ] = 1.0

        pos_weights_calculated[task] = current_pos_weights
        print(
            f"Wagi dla zadania '{task}': min={current_pos_weights.min():.2f}, max={current_pos_weights.max():.2f}, średnia={current_pos_weights.mean():.2f}"
        )

    return pos_weights_calculated


def calculate_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    pos_weights: Dict[str, torch.Tensor],
    task_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Oblicza łączną ważoną stratę dla batcha.

    Uwzględnia maskę, wagi klas (pos_weight dla BCEWithLogitsLoss)
    oraz wagi poszczególnych zadań (notes, onsets, contours).

    Args:
        model_output: Słownik z logitami z modelu dla 'notes', 'onsets', 'contours'.
        batch: Słownik z danymi wejściowymi, zawierający targety ('notes', 'onsets', 'contours') i 'mask'.
        device: Urządzenie (cpu/cuda).
        pos_weights: Słownik z tensorami pos_weight dla każdego zadania.
        task_weights: Słownik z wagami float dla każdego zadania, określający ich ważność w łącznej stracie.

    Returns:
        Słownik zawierający 'total_loss' oraz straty dla poszczególnych zadań.
    """
    notes_target = batch["notes"].to(device)
    onsets_target = batch["onsets"].to(device)
    contours_target = batch["contours"].to(device)
    mask = batch["mask"].to(device)  # Maska o kształcie (B, T)

    notes_logits = model_output["notes"]
    onsets_logits = model_output["onsets"]
    contours_logits = model_output["contours"]

    # Inicjalizacja funkcji strat z wagami pos_weight
    bce_notes = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weights["notes"].to(device)
    )
    bce_onsets = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weights["onsets"].to(device)
    )
    bce_contours = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=pos_weights["contours"].to(device)
    )

    # Obliczenie strat (bez redukcji, aby zastosować maskę)
    loss_notes_unreduced = bce_notes(notes_logits, notes_target.float())
    loss_onsets_unreduced = bce_onsets(onsets_logits, onsets_target.float())
    loss_contours_unreduced = bce_contours(contours_logits, contours_target.float())

    # Przygotowanie masek o odpowiednich wymiarach (B, T, F)
    mask_notes_expanded = mask.unsqueeze(-1).expand_as(notes_target)
    mask_contours_expanded = mask.unsqueeze(-1).expand_as(contours_target)

    # Zastosowanie maski do strat
    loss_notes_masked = loss_notes_unreduced * mask_notes_expanded
    loss_onsets_masked = (
        loss_onsets_unreduced * mask_notes_expanded
    )  # Onsety używają tej samej maski co nuty
    loss_contours_masked = loss_contours_unreduced * mask_contours_expanded

    # Obliczenie średniej straty tylko dla aktywnych (niezamaskowanych) elementów
    eps = 1e-8  # Dla uniknięcia dzielenia przez zero, jeśli suma maski jest 0
    num_active_notes = mask_notes_expanded.sum()
    num_active_contours = mask_contours_expanded.sum()

    loss_notes = loss_notes_masked.sum() / (num_active_notes + eps)
    loss_onsets = loss_onsets_masked.sum() / (
        num_active_notes + eps
    )  # Onsety normalizowane przez tę samą liczbę co nuty
    loss_contours = loss_contours_masked.sum() / (num_active_contours + eps)

    # Łączna strata z uwzględnieniem wag zadań
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
