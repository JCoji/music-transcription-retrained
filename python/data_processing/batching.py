import torch
from torch.nn.utils.rnn import pad_sequence
import config


def collate_fn_pad(batch_data):
    features_tensor_list = []
    onset_targets_tensor_list = []
    fret_targets_tensor_list = []
    raw_labels_list = []
    sequence_lengths = []

    for features_sample, (onset_targets_sample, fret_targets_sample), raw_labels_sample in batch_data:
        features_tensor_list.append(features_sample.T)
        onset_targets_tensor_list.append(onset_targets_sample)
        fret_targets_tensor_list.append(fret_targets_sample)
        raw_labels_list.append(raw_labels_sample)
        sequence_lengths.append(features_sample.shape[1])

    padded_features_batch = pad_sequence(features_tensor_list, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    padded_onset_targets_batch = pad_sequence(onset_targets_tensor_list, batch_first=True, padding_value=0.0)
    padded_fret_targets_batch = pad_sequence(
        fret_targets_tensor_list, batch_first=True, padding_value=config.FRET_PADDING_VALUE
    )
    lengths_tensor_batch = torch.tensor(sequence_lengths, dtype=torch.int64)

    return padded_features_batch, (padded_onset_targets_batch, padded_fret_targets_batch), lengths_tensor_batch, raw_labels_list