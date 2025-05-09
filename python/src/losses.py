import torch
import torch.nn.functional as F


def binary_focal_loss(inputs, targets, gamma=2.0, alpha=None, reduction='mean', mask=None, task_name=None):
    # Konwertuje logity na prawdopodobieństwa i cele na float
    probs = torch.sigmoid(inputs)
    targets = targets.float()

    # Oblicza binary cross entropy bez redukcji
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    # Oblicza wagę focal loss
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Stosuje współczynnik alpha jeśli podany
    if alpha is not None:
        if isinstance(alpha, dict):
            # Jeśli alpha jest słownikiem, wybierz odpowiednią wartość dla zadania
            alpha_value = alpha.get(task_name, 0.5)
        else:
            # Jeśli alpha jest liczbą
            alpha_value = alpha

        alpha_t = alpha_value * targets + (1 - alpha_value) * (1 - targets)
        bce_loss = alpha_t * bce_loss

    # Oblicza finalną wartość funkcji straty
    loss = focal_weight * bce_loss

    # Stosuje maskę jeśli podana
    if mask is not None:
        # Rozszerza maskę z [B, T] do [B, T, F]
        mask = mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask

        # Zapewnia prawidłową redukcję tylko dla niezmaskowanych elementów
        if reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'
    else:
        # Brak maski: stosuje standardową redukcję
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


def weighted_bce_with_mask(preds, targets, mask, pos_weight=0.5, label_smoothing=0.2):
    # label smoothing
    targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # maskowanie paddingu
    mask = mask.unsqueeze(-1)  # [B, T, 1]
    preds = preds * mask
    targets = targets * mask

    # obliczanie BCE
    bce = F.binary_cross_entropy(preds, targets, reduction='none')  # [B, T, F]

    # maska pozytywnych i negatywnych
    pos_mask = (targets > 0.5).float()
    neg_mask = (targets <= 0.5).float()

    # liczba pozytywnych i negatywnych próbek (z maską)
    n_pos = (pos_mask * mask).sum()
    n_neg = (neg_mask * mask).sum()

    # ważenie strat
    loss_pos = (bce * pos_mask * mask).sum()
    loss_neg = (bce * neg_mask * mask).sum()

    if n_pos > 0 and n_neg > 0:
        loss = (pos_weight * loss_pos / n_pos) + ((1 - pos_weight) * loss_neg / n_neg)
    else:
        loss = bce.mean()  # fallback

    return loss
