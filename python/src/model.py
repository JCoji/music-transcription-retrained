import torch
import torch.nn as nn

class MultiTaskTranscriptionModel(nn.Module):
    def __init__(self, freq_bins_out_notes: int, freq_bins_out_contours: int,
                 hidden_channels: int = 32):
        super().__init__()

        #Wspólna baza
        self.conv_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        #Kontury
        self.conv_contours = nn.Sequential(
            nn.Conv2d(hidden_channels, 8, kernel_size=(3, 3 * 13), padding=(1, 19)),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 1, kernel_size=(5, 5), padding=2),
            nn.Sigmoid()
        )

        #Nuty (notes)
        self.conv_notes_reduce = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(7, 7), padding=3, stride=1),
            nn.ReLU()
        )
        self.conv_notes_output = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=(7, 3), padding=(3, 1)),
            nn.Sigmoid()
        )

        #Onsety
        self.conv_onsets_reduce = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(5, 5), padding=2, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.conv_onsets_output = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, 1, kernel_size=(3, 3), padding=1),
            nn.Sigmoid()
        )

        self.freq_bins_out_notes = freq_bins_out_notes
        self.freq_bins_out_contours = freq_bins_out_contours

    def forward(self, features):  # features: [B, T, F]
        B, T, F = features.shape
        x = features.unsqueeze(1)  # [B, 1, T, F]

        #Wspólna baza
        x_shared = self.conv_shared(x)  # [B, C, T, F]


        contours_map = self.conv_contours(x_shared)  # [B, 1, T, F]
        contours_out = contours_map.squeeze(1).permute(0, 2, 1)  # [B, F, T] → chcemy [B, T, F]
        contours_out = contours_out.permute(0, 2, 1).contiguous()  # [B, T, F]

        #Nuty
        notes_feats = self.conv_notes_reduce(contours_map)  # [B, C, T, F//3]
        notes_map = self.conv_notes_output(notes_feats)  # [B, 1, T, F//3]
        notes_out = notes_map.squeeze(1)  # [B, T, F_notes]

        #Onsety
        onsets_feats = self.conv_onsets_reduce(x_shared)  # [B, C, T, F//3]

        # Concatenacja z notes_map (tak jak Spotify)
        concat_feats = torch.cat([notes_map, onsets_feats], dim=1)  # [B, C+1, T, F//3]
        onsets_map = self.conv_onsets_output(concat_feats)  # [B, 1, T, F//3]
        onsets_out = onsets_map.squeeze(1)  # [B, T, F_notes]

        return {
            "contours": contours_out[:, :, :self.freq_bins_out_contours],
            "notes": notes_out[:, :, :self.freq_bins_out_notes],
            "onsets": onsets_out[:, :, :self.freq_bins_out_notes]
        }
