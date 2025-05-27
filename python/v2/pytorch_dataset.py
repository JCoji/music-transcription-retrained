import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import torchaudio
import mirdata


def augment_time_stretch(y, sr, stretch_factor, raw_labels_tensor):
    y_stretched = librosa.effects.time_stretch(y=y, rate=stretch_factor)
    if raw_labels_tensor is not None and raw_labels_tensor.numel() > 0:
        raw_labels_tensor_adjusted = raw_labels_tensor.clone()
        raw_labels_tensor_adjusted[:, 0] /= stretch_factor
        raw_labels_tensor_adjusted[:, 1] /= stretch_factor
    else:
        raw_labels_tensor_adjusted = raw_labels_tensor
    return y_stretched, raw_labels_tensor_adjusted


def augment_add_noise(y, noise_level):
    noise = np.random.randn(len(y)) * noise_level
    return y + noise.astype(y.dtype)


def augment_random_gain(y, gain_factor):
    return y * gain_factor


class GuitarSetTabDataset(Dataset):
    def __init__(
        self,
        data_base_dir,
        split,
        hop_length,
        sample_rate,
        max_frets,
        n_fft,
        n_mels,
        labels_transform_func=None,
        data_home=None,
        apply_audio_augmentations=True,
        p_time_stretch=0.5,
        time_stretch_range=(0.85, 1.15),
        p_add_noise=0.5,
        noise_level_range=(0.0005, 0.005),
        p_random_gain=0.5,
        gain_range=(0.7, 1.3),
        apply_specaugment=True,
        spec_time_mask_max_percentage=0.1,
        spec_freq_mask_max_percentage=0.15,
    ):

        self.data_base_dir = data_base_dir
        self.split = split
        self.split_data_dir = os.path.join(
            self.data_base_dir, self.split
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_frets = max_frets
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.labels_transform_func = labels_transform_func

        self.data_home = (
            data_home
        )
        self.guitarset_loader = None

        self.apply_audio_augmentations = (
            apply_audio_augmentations if self.split == "train" else False
        )

        if self.apply_audio_augmentations:
            if self.data_home is None:
                raise ValueError(
                    "`data_home` musi być dostarczone dla 'train' split z włączonymi augmentacjami audio."
                )
            try:
                self.guitarset_loader = mirdata.initialize(
                    "guitarset", data_home=self.data_home
                )
                if not self.guitarset_loader.track_ids:
                    raise RuntimeError(
                        f"mirdata.GuitarSet nie znalazł żadnych utworów w data_home='{self.data_home}'."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Nie udało się zainicjalizować mirdata.GuitarSet z data_home='{self.data_home}'. Błąd: {e}"
                )

        self.features_files_or_audio_paths = []
        self.labels_files = []
        self.track_ids_base = []
        self.track_ids_full = []

        ids_file_path = os.path.join(self.data_base_dir, f"{self.split}_ids.txt")
        if not os.path.exists(ids_file_path):
            raise FileNotFoundError(
                f"Plik z ID dla splitu '{split}' nie został znaleziony: {ids_file_path}"
            )

        with open(ids_file_path, "r") as f:
            for line in f:
                track_id_key_from_file = (
                    line.strip()
                )
                if not track_id_key_from_file:
                    continue

                track_id_base_name_for_pt = os.path.splitext(
                    os.path.basename(track_id_key_from_file)
                )[0]
                lab_path = os.path.join(self.split_data_dir, f"{track_id_base_name_for_pt}_labels.pt")

                if not os.path.exists(lab_path):
                    continue

                if self.apply_audio_augmentations:
                    if self.guitarset_loader is None:
                        print(
                            f"Krytyczne: guitarset_loader nie jest zainicjalizowany dla {track_id_key_from_file} mimo apply_audio_augmentations=True."
                        )
                        continue
                    try:
                        track_obj = self.guitarset_loader.track(track_id_key_from_file)

                        audio_path_candidate = None
                        if (
                            hasattr(track_obj, "audio_mix_path")
                            and track_obj.audio_mix_path
                            and os.path.exists(track_obj.audio_mix_path)
                        ):
                            audio_path_candidate = track_obj.audio_mix_path
                        elif (
                            hasattr(track_obj, "audio_mic_path")
                            and track_obj.audio_mic_path
                            and os.path.exists(track_obj.audio_mic_path)
                        ):
                            audio_path_candidate = track_obj.audio_mic_path

                        if audio_path_candidate:
                            self.features_files_or_audio_paths.append(
                                audio_path_candidate
                            )
                            self.labels_files.append(lab_path)
                            self.track_ids_base.append(
                                track_id_base_name_for_pt
                            )
                            self.track_ids_full.append(
                                track_id_key_from_file
                            )
                    except mirdata.core.errors.TrackIdError:
                        pass
                    except Exception as e:
                        print(
                            f"Błąd przy pobieraniu ścieżki audio dla {track_id_key_from_file}: {e}"
                        )

                else:
                    feat_path = os.path.join(
                        self.split_data_dir, f"{track_id_base_name_for_pt}_features.pt")
                    if os.path.exists(
                        feat_path
                    ):
                        self.features_files_or_audio_paths.append(feat_path)
                        self.labels_files.append(lab_path)
                        self.track_ids_base.append(track_id_base_name_for_pt)
                        self.track_ids_full.append(track_id_key_from_file)

        self.p_time_stretch = p_time_stretch
        self.time_stretch_range = time_stretch_range
        self.p_add_noise = p_add_noise
        self.noise_level_range = noise_level_range
        self.p_random_gain = p_random_gain
        self.gain_range = gain_range

        self.apply_specaugment = apply_specaugment if self.split == "train" else False
        if self.apply_specaugment:
            self.spec_augment_transform = torch.nn.Sequential(
                torchaudio.transforms.TimeMasking(
                    time_mask_param=int(spec_time_mask_max_percentage * 1000)
                ),
                torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=int(spec_freq_mask_max_percentage * self.n_mels)
                ),
            )

        if not self.features_files_or_audio_paths:
            print(
                f"Ostrzeżenie: Brak plików danych dla splitu '{self.split}' w katalogu '{self.split_data_dir}' po przetworzeniu list ID. Dataset będzie pusty."
            )

    def __len__(self):
        return len(self.features_files_or_audio_paths)

    def __getitem__(self, idx):
        if idx >= len(self.features_files_or_audio_paths):
            raise IndexError(
                f"Indeks {idx} poza zakresem {len(self.features_files_or_audio_paths)}"
            )

        path_or_audio_identifier = self.features_files_or_audio_paths[
            idx
        ]
        labels_path = self.labels_files[idx]


        try:
            raw_labels = torch.load(labels_path, weights_only=True)
            current_labels_for_transform_func = raw_labels
        except Exception as e:
            raise RuntimeError(
                f"Nie można wczytać etykiet dla indeksu {idx} (Plik: {labels_path})"
            ) from e

        if self.apply_audio_augmentations:
            audio_path = (
                path_or_audio_identifier
            )
            try:
                y, sr_loaded = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                if sr_loaded != self.sample_rate:
                    y = librosa.resample(
                        y, orig_sr=sr_loaded, target_sr=self.sample_rate
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Nie można wczytać pliku audio {audio_path} dla indeksu {idx}"
                ) from e

            current_labels_for_transform_func = (
                raw_labels.clone()
            )

            if np.random.rand() < self.p_time_stretch:
                stretch_factor = np.random.uniform(
                    self.time_stretch_range[0], self.time_stretch_range[1]
                )
                y, current_labels_for_transform_func = augment_time_stretch(
                    y,
                    self.sample_rate,
                    stretch_factor,
                    current_labels_for_transform_func,
                )

            if np.random.rand() < self.p_random_gain:
                gain_factor = np.random.uniform(self.gain_range[0], self.gain_range[1])
                y = augment_random_gain(y, gain_factor)

            if np.random.rand() < self.p_add_noise:
                noise_level = np.random.uniform(
                    self.noise_level_range[0], self.noise_level_range[1]
                )
                y = augment_add_noise(y, noise_level)

            try:
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=y,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                )
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                features = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(
                    f"Błąd podczas obliczania spektrogramu dla {audio_path} (indeks {idx})"
                ) from e

        else:
            features_path = path_or_audio_identifier
            try:
                features = torch.load(features_path, weights_only=True)
            except Exception as e:
                raise RuntimeError(
                    f"Nie można wczytać cech dla indeksu {idx} (Plik: {features_path})"
                ) from e

        if self.apply_specaugment:  # Tylko dla 'train'
            features = features.unsqueeze(
                0
            )
            features = self.spec_augment_transform(features)
            features = features.squeeze(0)

        if self.labels_transform_func:
            labels = self.labels_transform_func(
                current_labels_for_transform_func,
                features,
                self.hop_length,
                self.sample_rate,
                self.max_frets,
            )
        else:
            labels = current_labels_for_transform_func

        return features, labels

    def get_full_track_id(self, idx):
        if idx >= len(self.track_ids_full):
            raise IndexError(
                f"Indeks {idx} poza zakresem listy track_ids_full ({len(self.track_ids_full)})"
            )
        return self.track_ids_full[idx]


def create_frame_labels(
    raw_labels_tensor, features_tensor, hop_length, sample_rate, max_frets
):
    if hop_length is None or sample_rate is None or max_frets is None:
        raise ValueError(
            "Brakujące parametry hop_length, sample_rate lub max_frets w create_frame_labels"
        )

    num_frames = features_tensor.shape[1]
    num_strings = 6
    onset_targets = torch.zeros((num_frames, num_strings), dtype=torch.float32)
    fret_targets = torch.full(
        (num_frames, num_strings), max_frets + 1, dtype=torch.long
    )
    time_per_frame = hop_length / sample_rate

    if raw_labels_tensor is not None and raw_labels_tensor.numel() > 0:
        for i in range(raw_labels_tensor.shape[0]):
            onset_sec = raw_labels_tensor[i, 0].item()
            offset_sec = raw_labels_tensor[i, 1].item()
            string_idx_float = raw_labels_tensor[i, 2].item()
            fret_num_float = raw_labels_tensor[i, 3].item()

            string_idx = int(string_idx_float)
            fret_num = int(fret_num_float)

            onset_frame = int(round(onset_sec / time_per_frame))
            offset_frame = int(round(offset_sec / time_per_frame))

            onset_frame = min(max(0, onset_frame), num_frames - 1)
            offset_frame = min(max(0, offset_frame), num_frames - 1)

            if string_idx < 0 or string_idx >= num_strings:
                continue

            if onset_frame < num_frames:
                onset_targets[onset_frame, string_idx] = 1.0

            current_fret_to_encode = (
                min(fret_num, max_frets) if fret_num >= 0 else (max_frets + 1)
            )

            start_fret_frame = onset_frame
            end_fret_frame = offset_frame

            for frame_idx in range(start_fret_frame, end_fret_frame + 1):
                if frame_idx < num_frames:
                    fret_targets[frame_idx, string_idx] = current_fret_to_encode

    return onset_targets, fret_targets
