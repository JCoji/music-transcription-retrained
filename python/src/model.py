import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import N_FREQ_BINS_NOTES, N_FREQ_BINS_CONTOURS


class TranscriptionModel(nn.Module):
    def __init__(self, n_freq_bins: int, hidden_size: int = 128, dropout: float = 0.3):
        super(TranscriptionModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.n_freq_bins = n_freq_bins

        self.bi_gru = nn.GRU(
            input_size=32 * self.n_freq_bins,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        self.notes_head = nn.Linear(hidden_size * 2, N_FREQ_BINS_NOTES)
        self.contours_head = nn.Linear(hidden_size * 2, N_FREQ_BINS_CONTOURS)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.encoder(x)  # [B, C, T, F]
        x = x.permute(0, 2, 1, 3)  # [B, T, C, F]
        x = x.flatten(2)  # [B, T, C*F]

        out, _ = self.bi_gru(x)

        notes = self.notes_head(out)
        contours = self.contours_head(out)

        return {
            "notes": notes,
            "contours": contours
        }
