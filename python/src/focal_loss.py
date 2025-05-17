# src/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="none", pos_weight=None):
        """
        Focal Loss for binary classification.

        Args:
            alpha (float, optional): Weighting factor for the rare class (positive class).
                                     If pos_weight is also provided, pos_weight will be used as alpha
                                     for the positive class.
                                     If alpha is a tensor, it should be of the same shape as the input logits'
                                     last dimension (number of classes/bins).
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                                       'mean': the sum of the output will be divided by the number of
                                       elements in the output, 'sum': the output will be summed.
                                       Defaults to 'none' to allow manual masking.
            pos_weight (torch.Tensor, optional): A weight of positive examples. Must be a vector with length
                                                 equal to the number of classes/bins.
                                                 If provided, it's used to scale the loss for the positive class,
                                                 effectively setting alpha for the positive class.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model predictions, raw logits (B, ..., C).
            targets (torch.Tensor): Ground truth labels, same shape as logits (B, ..., C).
        Returns:
            torch.Tensor: Calculated focal loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)  # p if targets==1 else 1-p

        alpha_t = torch.ones_like(logits)  # Default alpha_t to 1

        # Determine alpha factor
        current_alpha_for_positive = None
        if self.alpha is not None:
            current_alpha_for_positive = self.alpha
        elif self.pos_weight is not None:
            current_alpha_for_positive = (
                self.pos_weight
            )  # Interpret pos_weight as alpha for positive class

        if current_alpha_for_positive is not None:
            if isinstance(current_alpha_for_positive, float):  # Scalar alpha
                alpha_for_positive_class = current_alpha_for_positive
                alpha_for_negative_class = 1.0 - current_alpha_for_positive
            elif isinstance(
                current_alpha_for_positive, torch.Tensor
            ):  # Tensor alpha (pos_weight)
                # Reshape pos_weight to be broadcastable
                if logits.dim() == 3:  # (B, T, F)
                    alpha_for_positive_class = current_alpha_for_positive.view(
                        1, 1, -1
                    ).expand_as(logits)
                elif logits.dim() == 2:  # (B, F)
                    alpha_for_positive_class = current_alpha_for_positive.view(
                        1, -1
                    ).expand_as(logits)
                else:  # Fallback for unexpected dims
                    alpha_for_positive_class = current_alpha_for_positive.expand_as(
                        logits
                    )
                if isinstance(
                    self.alpha, float
                ):  # if original alpha was scalar, use 1-alpha for negative
                    alpha_for_negative_class = 1.0 - self.alpha
                else:  # if pos_weight is used, negative class is not scaled by this factor here.
                    alpha_for_negative_class = torch.ones_like(logits)

            alpha_t = torch.where(
                targets == 1, alpha_for_positive_class, alpha_for_negative_class
            )

        focusing_factor = (1 - pt).pow(self.gamma)
        focal_loss = alpha_t * focusing_factor * BCE_loss

        if self.reduction == "mean":
            # Careful: mean over all elements, or mean over non-zero masked elements?
            # Standard PyTorch way is mean over all elements if reduction='mean'.
            # Masking should be applied after this if 'none' is used.
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
