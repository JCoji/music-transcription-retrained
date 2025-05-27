import torch
import torch.nn as nn

class TabCNN(nn.Module):
    """
    Prosta sieć konwolucyjna (CNN) używana jako ekstraktor cech.
    Składa się z sekwencji bloków Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
    """
    def __init__(
        self,
        input_channels=1,
        output_channels_list=None,
        kernel_sizes=None,
        strides=None,
        paddings=None,
        pooling_kernels=None,
        pooling_strides=None,
    ):
        """
        Inicjalizuje warstwy CNN.

        Args:
            input_channels (int, optional): Liczba kanałów wejściowych (np. 1 dla skali szarości). Domyślnie 1.
            output_channels_list (list of int, optional): Lista liczb kanałów wyjściowych dla każdej warstwy konwolucyjnej.
                                                       Domyślnie [32, 64, 128].
            kernel_sizes (list of tuple, optional): Lista rozmiarów kerneli (wysokość, szerokość) dla każdej warstwy Conv2d.
                                                 Domyślnie [(3, 3)] dla każdej warstwy.
            strides (list of tuple, optional): Lista kroków (stride_h, stride_w) dla każdej warstwy Conv2d.
                                            Domyślnie [(1, 1)] dla każdej warstwy.
            paddings (list of tuple, optional): Lista paddingów (padding_h, padding_w) dla każdej warstwy Conv2d.
                                             Domyślnie [(1, 1)] dla każdej warstwy.
            pooling_kernels (list of tuple, optional): Lista rozmiarów kerneli (pool_h, pool_w) dla każdej warstwy MaxPool2d.
                                                    Domyślnie [(2, 1), (2, 1), (1, 1)] jeśli 3 warstwy, inaczej [(2,1)] dla każdej.
            pooling_strides (list of tuple, optional): Lista kroków (pool_stride_h, pool_stride_w) dla każdej warstwy MaxPool2d.
                                                    Domyślnie [(2, 1), (2, 1), (1, 1)] jeśli 3 warstwy, inaczej [(2,1)] dla każdej.

        Raises:
            AssertionError: Jeśli długości list parametrów konfiguracyjnych nie są równe.
        """
        super().__init__()
        default_len = 3 # Domyślna liczba warstw konwolucyjnych, jeśli parametry nie są podane

        # Ustawienie domyślnych wartości dla parametrów, jeśli nie zostały podane
        if output_channels_list is None:
            output_channels_list = [32, 64, 128][:default_len]
        if kernel_sizes is None:
            kernel_sizes = [(3, 3)] * len(output_channels_list)
        if strides is None:
            strides = [(1, 1)] * len(output_channels_list)
        if paddings is None:
            paddings = [(1, 1)] * len(output_channels_list) # Często padding=(kernel_size-1)/2 dla 'same'
        if pooling_kernels is None:
            # Domyślne wartości dla pooling_kernels, specyficzne dla 3 warstw lub ogólne
            if len(output_channels_list) == 3:
                pooling_kernels = [(2, 1), (2, 1), (1, 1)]
            else:
                pooling_kernels = [(2, 1)] * len(output_channels_list)
        if pooling_strides is None:
            # Domyślne wartości dla pooling_strides, analogicznie do pooling_kernels
            if len(output_channels_list) == 3:
                pooling_strides = [(2, 1), (2, 1), (1, 1)]
            else:
                pooling_strides = [(2, 1)] * len(output_channels_list)

        # Sprawdzenie, czy wszystkie listy parametrów mają tę samą długość
        assert (
            len(output_channels_list)
            == len(kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(pooling_kernels)
            == len(pooling_strides)
        ), (
            f"Długości list parametrów CNN muszą być równe. Otrzymano: "
            f"output_channels: {len(output_channels_list)}, kernels: {len(kernel_sizes)}, "
            f"strides: {len(strides)}, paddings: {len(paddings)}, "
            f"pool_kernels: {len(pooling_kernels)}, pool_strides: {len(pooling_strides)}"
        )

        self.conv_layers = nn.ModuleList() # Lista do przechowywania warstw konwolucyjnych
        current_channels = input_channels
        # Tworzenie kolejnych bloków konwolucyjnych
        for i in range(len(output_channels_list)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        output_channels_list[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                    ),
                    nn.BatchNorm2d(output_channels_list[i]), # Normalizacja wsadowa
                    nn.ReLU(), # Funkcja aktywacji ReLU
                    nn.MaxPool2d( # Warstwa Max Pooling
                        kernel_size=pooling_kernels[i], stride=pooling_strides[i]
                    ),
                )
            )
            current_channels = output_channels_list[i] # Aktualizacja liczby kanałów dla następnej warstwy

        self.output_channels = current_channels # Liczba kanałów na wyjściu z ostatniej warstwy CNN

    def forward(self, x):
        """
        Definiuje przejście danych przez sieć CNN.

        Args:
            x (torch.Tensor): Tensor wejściowy o kształcie (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Tensor cech po przejściu przez warstwy CNN.
        """
        # Przekazanie danych przez kolejne warstwy konwolucyjne
        for layer in self.conv_layers:
            x = layer(x)
        return x


class GuitarTabCRNN(nn.Module):
    """
    Model CRNN (Convolutional Recurrent Neural Network) do transkrypcji tabulatur gitarowych.

    Składa się z bloku CNN (TabCNN) do ekstrakcji cech z wejściowego spektrogramu,
    a następnie z dwukierunkowej sieci LSTM do modelowania sekwencji czasowych.
    Na końcu znajdują się warstwy w pełni połączone (FC) do predykcji onsetów i progów.
    """
    def __init__(
        self,
        num_frames_rnn_input_dim, # Poprawiona nazwa argumentu dla jasności
        rnn_hidden_size=256,
        rnn_layers=2,
        rnn_dropout=0.2,
        num_strings=6,
        max_frets=20, # Maksymalny numer progu (np. 20, 21, 22, 24)
        cnn_input_channels=1,
        cnn_output_channels_list=None,
        cnn_kernel_sizes=None,
        cnn_strides=None,
        cnn_paddings=None,
        cnn_pooling_kernels=None,
        cnn_pooling_strides=None,
    ):
        """
        Inicjalizuje model CRNN.

        Args:
            num_frames_rnn_input_dim (int): Oczekiwany wymiar wejściowy dla RNN po przetworzeniu przez CNN
                                           i przekształceniu (channels_out * reduced_n_mels).
            rnn_hidden_size (int, optional): Liczba jednostek w warstwie ukrytej RNN. Domyślnie 256.
            rnn_layers (int, optional): Liczba warstw RNN. Domyślnie 2.
            rnn_dropout (float, optional): Dropout dla RNN (stosowany między warstwami, jeśli rnn_layers > 1). Domyślnie 0.2.
            num_strings (int, optional): Liczba strun gitary. Domyślnie 6.
            max_frets (int, optional): Maksymalny numer progu do przewidzenia. Klasy progów to 0..max_frets
                                     oraz jedna dodatkowa klasa dla "ciszy" lub "braku dźwięku". Domyślnie 20.
            cnn_input_channels (int, optional): Liczba kanałów wejściowych dla CNN. Domyślnie 1.
            cnn_output_channels_list (list of int, optional): Parametry dla TabCNN.
            cnn_kernel_sizes (list of tuple, optional): Parametry dla TabCNN.
            cnn_strides (list of tuple, optional): Parametry dla TabCNN.
            cnn_paddings (list of tuple, optional): Parametry dla TabCNN.
            cnn_pooling_kernels (list of tuple, optional): Parametry dla TabCNN.
            cnn_pooling_strides (list of tuple, optional): Parametry dla TabCNN.
        """
        super().__init__()
        self.num_strings = num_strings
        # Liczba klas progów = max_frets (0-max_frets) + 1 (próg "cisza")
        self.num_fret_classes = max_frets + 2 # 0..max_frets to progi, max_frets+1 to cisza/brak dźwięku

        # Inicjalizacja części konwolucyjnej (CNN)
        self.cnn = TabCNN(
            input_channels=cnn_input_channels,
            output_channels_list=cnn_output_channels_list,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            paddings=cnn_paddings,
            pooling_kernels=cnn_pooling_kernels,
            pooling_strides=cnn_pooling_strides,
        )

        # Wymiar wejściowy dla RNN, powinien być zgodny z wyjściem CNN po przekształceniu
        self.rnn_input_dim = num_frames_rnn_input_dim

        # Inicjalizacja części rekurencyjnej (LSTM)
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_dim, # Wymiar cech na wejściu do LSTM dla każdej klatki czasowej
            hidden_size=rnn_hidden_size,   # Liczba jednostek w warstwie ukrytej LSTM
            num_layers=rnn_layers,         # Liczba warstw LSTM
            batch_first=True,              # Wejście i wyjście mają batch_size jako pierwszy wymiar
            bidirectional=True,            # Użycie dwukierunkowego LSTM
            dropout=rnn_dropout if rnn_layers > 1 else 0, # Dropout między warstwami LSTM
        )

        # Rozmiar wyjścia z dwukierunkowego LSTM (hidden_size dla kierunku w przód + hidden_size dla kierunku w tył)
        rnn_output_size = 2 * rnn_hidden_size

        # Warstwa w pełni połączona (FC) do predykcji logitów onsetów
        # Wyjście: (batch_size, num_frames, num_strings)
        self.onset_fc = nn.Linear(rnn_output_size, self.num_strings)

        # Warstwa w pełni połączona (FC) do predykcji logitów progów
        # Wyjście "spłaszczone": (batch_size, num_frames, num_strings * num_fret_classes)
        self.fret_fc = nn.Linear(
            rnn_output_size, self.num_strings * self.num_fret_classes
        )

    def forward(self, x):
        """
        Definiuje przejście danych przez model CRNN.

        Args:
            x (torch.Tensor): Tensor wejściowy (spektrogram) o kształcie (batch_size, num_mel_bins, num_frames).

        Returns:
            tuple: Krotka zawierająca:
                   - onset_logits (torch.Tensor): Logity dla predykcji onsetów
                                                  o kształcie (batch_size, reduced_n_frames, num_strings).
                   - fret_logits (torch.Tensor): Logity dla predykcji progów
                                                 o kształcie (batch_size, reduced_n_frames, num_strings, num_fret_classes).
        Raises:
            ValueError: Jeśli rzeczywisty wymiar cech po CNN nie zgadza się z oczekiwanym `rnn_input_dim`.
        """
        # Dodanie wymiaru kanału dla CNN (batch_size, 1, num_mel_bins, num_frames)
        x = x.unsqueeze(1)

        # Przejście przez CNN
        x_cnn = self.cnn(x) # Kształt wyjściowy: (batch_size, channels_out, reduced_n_mels, reduced_n_frames)

        batch_size, channels_out, reduced_n_mels, reduced_n_frames = x_cnn.shape

        # Przygotowanie danych wejściowych dla RNN
        # Permutacja wymiarów, aby ramki czasowe były drugim wymiarem: (batch_size, reduced_n_frames, channels_out, reduced_n_mels)
        x_rnn_input = x_cnn.permute(0, 3, 1, 2)
        # Spłaszczenie wymiarów cech (kanałów i pasm mel) dla każdej ramki czasowej
        # Kształt: (batch_size, reduced_n_frames, channels_out * reduced_n_mels)
        x_rnn_input = x_rnn_input.reshape(
            batch_size, reduced_n_frames, channels_out * reduced_n_mels
        )

        # Sprawdzenie zgodności wymiarów wejściowych dla RNN
        if x_rnn_input.shape[2] != self.rnn_input_dim:
            raise ValueError(
                f"Wymiar cech po CNN ({x_rnn_input.shape[2]}) "
                f"nie zgadza się z oczekiwanym rnn_input_dim ({self.rnn_input_dim}). "
                f"Sprawdź konfigurację CNN (pooling) oraz wartość `num_frames_rnn_input_dim`."
            )

        # Przejście przez RNN
        x_rnn_output, _ = self.rnn(x_rnn_input) # Kształt wyjściowy: (batch_size, reduced_n_frames, 2 * rnn_hidden_size)

        # Predykcja logitów onsetów
        onset_logits = self.onset_fc(x_rnn_output) # Kształt: (batch_size, reduced_n_frames, num_strings)

        # Predykcja logitów progów (spłaszczone)
        fret_logits_flat = self.fret_fc(x_rnn_output) # Kształt: (batch_size, reduced_n_frames, num_strings * num_fret_classes)

        # Przekształcenie logitów progów do pożądanego kształtu
        # Kształt: (batch_size, reduced_n_frames, num_strings, num_fret_classes)
        fret_logits = fret_logits_flat.reshape(
            batch_size, reduced_n_frames, self.num_strings, self.num_fret_classes
        )

        return onset_logits, fret_logits