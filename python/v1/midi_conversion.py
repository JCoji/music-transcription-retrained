import numpy as np
import pretty_midi

from .config import ANNOTATION_HOP, NOTES_BINS_PER_SEMITONE, GUITAR_BASE_FREQUENCY


def predictions_to_midi(
    notes_binary: np.ndarray,
    onsets_binary: np.ndarray,
    output_midi_path: str,
    velocity: int = 100,
    min_note_duration_frames: int = 2,
    annotation_hop: float = ANNOTATION_HOP,
    notes_bins_per_semitone: int = NOTES_BINS_PER_SEMITONE,
    guitar_base_frequency: float = GUITAR_BASE_FREQUENCY,
):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=25)  # Acoustic Guitar (Steel)

    n_frames, n_freq_bins = notes_binary.shape
    time_per_frame = annotation_hop

    active_notes = {}

    for frame_idx in range(n_frames):
        current_time = frame_idx * time_per_frame

        for freq_bin_idx in range(n_freq_bins):
            if (
                onsets_binary[frame_idx, freq_bin_idx] == 1
                and freq_bin_idx not in active_notes
            ):
                active_notes[freq_bin_idx] = frame_idx

            if (
                freq_bin_idx in active_notes
                and notes_binary[frame_idx, freq_bin_idx] == 0
            ):
                start_frame = active_notes.pop(freq_bin_idx)
                end_frame = frame_idx

                duration_frames = end_frame - start_frame
                if duration_frames >= min_note_duration_frames:
                    semitone_offset = freq_bin_idx / notes_bins_per_semitone
                    base_midi_note = pretty_midi.hz_to_note_number(
                        guitar_base_frequency
                    )
                    midi_note_number = int(round(base_midi_note + semitone_offset))

                    start_time_sec = start_frame * time_per_frame
                    end_time_sec = end_frame * time_per_frame

                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=midi_note_number,
                        start=start_time_sec,
                        end=end_time_sec,
                    )
                    instrument.notes.append(note)

    for freq_bin_idx, start_frame in active_notes.items():
        end_frame = n_frames
        duration_frames = end_frame - start_frame
        if duration_frames >= min_note_duration_frames:
            semitone_offset = freq_bin_idx / notes_bins_per_semitone
            base_midi_note = pretty_midi.hz_to_note_number(guitar_base_frequency)
            midi_note_number = int(round(base_midi_note + semitone_offset))

            start_time_sec = start_frame * time_per_frame
            end_time_sec = end_frame * time_per_frame

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=midi_note_number,
                start=start_time_sec,
                end=end_time_sec,
            )
            instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    try:
        midi_data.write(output_midi_path)
        print(f"  Plik MIDI zapisany do: {output_midi_path}")
    except Exception as e:
        print(f"  Błąd podczas zapisywania pliku MIDI ({output_midi_path}): {e}")
