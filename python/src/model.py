from typing import Dict  # Dodano import dla type hintingu

import torch
import torch.nn as nn

from .config import N_FREQ_BINS_NOTES, N_FREQ_BINS_CONTOURS


class MultiTaskTranscriptionModel(nn.Module):
    """
    Model wielozadaniowej transkrypcji muzycznej oparty na CNN i RNN (LSTM).

    Przetwarza spektrogramy CQT w celu jednoczesnej predykcji nut,
    ich początków (onsetów) oraz konturów melodycznych.
    """

    def __init__(
        self,
        input_features: int = N_FREQ_BINS_NOTES,
        cnn_filters: list = [32, 64, 64],
        cnn_kernels: list = [(3, 3), (3, 3), (3, 3)],
        cnn_pools: list = [(1, 2), (1, 2), (1, 1)],
        rnn_hidden_size: int = 512,
        rnn_layers: int = 2,
        dropout: float = 0.3,
        output_notes_features: int = N_FREQ_BINS_NOTES,
        output_contours_features: int = N_FREQ_BINS_CONTOURS,
    ):
        """
        Inicjalizuje warstwy modelu.
        Args:
            input_features (int): Liczba cech wejściowych (np. koszy częstotliwości CQT).
            cnn_filters (list): Liczba filtrów w kolejnych warstwach splotowych.
            cnn_kernels (list): Rozmiary kerneli w warstwach splotowych.
            cnn_pools (list): Rozmiary okien pooling w warstwach MaxPool2d.
            rnn_hidden_size (int): Rozmiar warstwy ukrytej RNN.
            rnn_layers (int): Liczba warstw RNN.
            dropout (float): Współczynnik dropout.
            output_notes_features (int): Liczba cech wyjściowych dla predykcji nut i onsetów.
            output_contours_features (int): Liczba cech wyjściowych dla predykcji konturów.
        """
        super().__init__()

        self.input_features = input_features
        # print(f"Inicjalizacja modelu. Wymiar wejściowy CQT (input_features): {self.input_features}") # Można odkomentować do debugowania

        cnn_module_list = []
        in_channels = 1  # Jednokanałowe wejście (spektrogram)

        for i in range(len(cnn_filters)):
            cnn_module_list.append(
                nn.Conv2d(in_channels, cnn_filters[i], cnn_kernels[i], padding="same")
            )
            cnn_module_list.append(nn.BatchNorm2d(cnn_filters[i]))
            cnn_module_list.append(nn.ReLU())
            cnn_module_list.append(nn.MaxPool2d(cnn_pools[i]))
            cnn_module_list.append(nn.Dropout(dropout))
            in_channels = cnn_filters[i]

        self.cnn = nn.Sequential(*cnn_module_list)

        # Automatyczne obliczenie rozmiaru wejścia do RNN
        with torch.no_grad():  # Nie potrzebujemy gradientów do tego obliczenia
            # Użyj przykładowego tensora: (Batch, Kanały_wej, Czas, Częstotliwości_wej)
            # Wymiar czasu (np. 10) jest dowolny, nie wpływa na liczbę cech dla RNN
            dummy_input = torch.zeros(1, 1, 10, self.input_features)
            cnn_output_shape = self.cnn(dummy_input).shape
            # print(f"Kształt wyjścia z CNN (dla przykładowego T=10): {cnn_output_shape}") # Można odkomentować

            # Rozmiar wejścia RNN = kanały_wyjściowe_CNN * kosze_częstotliwości_po_poolingu
            rnn_input_size = cnn_output_shape[1] * cnn_output_shape[3]
            # print(f"Obliczony rozmiar wejścia do RNN: {rnn_input_size}") # Można odkomentować

        self.rnn = nn.LSTM(
            rnn_input_size,
            rnn_hidden_size,
            rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
        )

        # Rozmiar wyjścia z dwukierunkowego RNN
        rnn_output_size = rnn_hidden_size * 2

        self.fc_notes = nn.Linear(rnn_output_size, output_notes_features)
        self.fc_onsets = nn.Linear(
            rnn_output_size, output_notes_features
        )  # Taki sam wymiar jak nuty
        self.fc_contours = nn.Linear(rnn_output_size, output_contours_features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Definiuje przejście danych przez model.

        Args:
            x (torch.Tensor): Tensor wejściowy o kształcie (B, T, F_in),
                              gdzie B to batch size, T to liczba ramek czasowych,
                              a F_in to liczba cech wejściowych.
        Returns:
            Dict[str, torch.Tensor]: Słownik z logitami dla nut, onsetów i konturów.
        """
        # Wejście x: (B, T, F_in)
        x = x.unsqueeze(1)  # Dodanie wymiaru kanału -> (B, 1, T, F_in)

        x = self.cnn(x)  # Wyjście CNN: (B, C_out, T_out, F_out)
        # T_out to liczba ramek po CNN (zależy od poolingu w czasie)
        # F_out to liczba koszy częstotliwości po CNN (zależy od poolingu w częstotliwości)

        B, C_out, T_out, F_out = x.shape
        # Zmiana kolejności wymiarów dla RNN: (B, T_out, C_out, F_out)
        x = x.permute(0, 2, 1, 3)
        # Spłaszczenie wymiarów C_out i F_out: (B, T_out, C_out * F_out)
        x = x.reshape(B, T_out, C_out * F_out)

        x, _ = self.rnn(x)  # Wyjście RNN: (B, T_out, rnn_hidden_size * 2)

        notes_logits = self.fc_notes(x)
        onsets_logits = self.fc_onsets(x)
        contours_logits = self.fc_contours(x)

        return {
            "notes": notes_logits,
            "onsets": onsets_logits,
            "contours": contours_logits,
        }
