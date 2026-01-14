# 🎸 Guitar Tablature Transcription System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning system for **Automatic Guitar Tablature Transcription (GTT)** that converts polyphonic guitar audio recordings into tablature notation. This project uses a **Convolutional Recurrent Neural Network (CRNN)** architecture with multi-task learning to simultaneously predict note onsets and fret positions.

> **🏆 Achieved 0.8736 MPE F1-Score** on the GuitarSet dataset, surpassing state-of-the-art results for models trained exclusively on GuitarSet.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Thesis Documentation](#thesis-documentation)
- [Citation](#citation)
- [License](#license)

## Overview

This project addresses the challenging problem of transforming audio recordings into musical notation, specifically focusing on **polyphonic guitar tablature transcription**. Unlike traditional sheet music, tablature notation indicates the exact finger positions (string and fret numbers) on the guitar neck, making it more practical for guitarists.

### Key Challenges Addressed

- **Polyphony**: Simultaneous notes on up to 6 strings with overlapping harmonics
- **Fingering Ambiguity**: Same pitch achievable at multiple fret/string combinations
- **Timbral Variation**: Different guitar types, playing techniques, and recording conditions
- **Onset Detection**: Precise identification of note beginnings in complex audio

## Features

✅ **Multi-task Learning** - Simultaneous onset detection and fret classification  
✅ **CQT Spectrograms** - Constant-Q Transform for music-aware frequency representation  
✅ **Aggressive Data Augmentation** - Time stretch, noise, reverb, EQ, clipping, SpecAugment  
✅ **Bidirectional GRU** - Context-aware sequence modeling  
✅ **Tablature Export** - ASCII and MIDI output formats  
✅ **Hyperparameter Search** - Systematic experimentation framework  
✅ **Comprehensive Metrics** - MPE, TDR, and onset-level evaluation

## Architecture

The system uses a **CRNN (Convolutional Recurrent Neural Network)** architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: CQT Spectrogram                   │
│                    (168 bins × T frames)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   5-Layer CNN Encoder                       │
│  Conv2D(32) → Conv2D(64) → Conv2D(128) → Conv2D(128) → 128 │
│  + BatchNorm + ReLU + MaxPool after each layer              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              2-Layer Bidirectional GRU                      │
│              Hidden Size: 768, Dropout: 0.5                 │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐
        │   Onset Head      │ │   Fret Head       │
        │   FC → [T, 6]     │ │   FC → [T, 6, 22] │
        │   (per-string)    │ │   (per-string     │
        │                   │ │    per-fret)      │
        └───────────────────┘ └───────────────────┘
```

### Audio Processing Pipeline

| Parameter   | Value        | Description                 |
| ----------- | ------------ | --------------------------- |
| Sample Rate | 22,050 Hz    | Standard for music analysis |
| CQT Bins    | 168          | 7 octaves × 24 bins/octave  |
| Hop Length  | 512 samples  | ~23ms frame resolution      |
| F_min       | E2 (82.4 Hz) | Lowest guitar note          |
| Max Frets   | 20           | Standard guitar fretboard   |

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/music-transcription.git
cd music-transcription

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd python
pip install -r requirements.txt
```

### Dependencies

| Package     | Version     | Purpose                      |
| ----------- | ----------- | ---------------------------- |
| torch       | 2.5.1+cu121 | Deep learning framework      |
| torchaudio  | 2.5.1+cu121 | Audio processing             |
| librosa     | 0.11.0      | Audio feature extraction     |
| mirdata     | 0.3.9       | GuitarSet dataset loading    |
| nnAudio     | 0.3.3       | GPU-accelerated spectrograms |
| pretty_midi | ≥0.2.10     | MIDI file generation         |
| mir_eval    | -           | Evaluation metrics           |

## Dataset

This project uses the **GuitarSet** dataset, a publicly available collection designed specifically for guitar transcription research.

### GuitarSet Features

- **360 recordings** from 6 performers
- **Hexaphonic audio** (individual string signals)
- **Precise JAMS annotations** with onset, offset, pitch, string, and fret information
- **Multiple styles**: Rock, Jazz, Funk, Bossa Nova, Singer-Songwriter

### Data Preparation

```bash
cd python

# The dataset will be automatically downloaded via mirdata
python -c "import mirdata; mirdata.initialize('guitarset').download()"
```

The preprocessing pipeline splits data into:

- **Train**: 80% (288 tracks)
- **Validation**: 10% (36 tracks)
- **Test**: 10% (36 tracks)

## Usage

### Training

Open and run the Jupyter notebook for complete training workflow:

```bash
jupyter notebook python/guitar_ATM.ipynb
```

Or use the training pipeline programmatically:

```python
from training import pipeline
from data_processing import preparation, dataset
import config

# Prepare data splits
track_splits = preparation.prepare_track_splits(
    data_home=config.DATA_HOME_DEFAULT,
    problematic_files_list=config.PROBLEMATIC_FILES,
    test_split_fraction=config.TEST_SPLIT_SIZE,
    validation_split_fraction=config.VALIDATION_SPLIT_SIZE,
    seed=config.RANDOM_SEED,
    output_dir_for_ids=config.OUTPUT_BASE_DIR_DEFAULT
)

# Preprocess audio and labels
preparation.preprocess_guitarset_data(
    guitarset_data_home=config.DATA_HOME_DEFAULT,
    processed_output_base_dir=config.OUTPUT_BASE_DIR_DEFAULT,
    track_ids_map=track_splits,
    audio_sample_rate=config.SAMPLE_RATE,
    audio_hop_length=config.HOP_LENGTH,
    audio_n_cqt_bins=config.N_BINS_CQT,
    audio_cqt_bins_per_octave=config.BINS_PER_OCTAVE_CQT,
    audio_cqt_fmin=config.FMIN_CQT
)
```

### Hyperparameter Search

Configure experiments in `hyperparam_set_v1.json`:

```json
{
  "run_description": "Baseline_GRU_768",
  "LEARNING_RATE_INIT": 0.0003,
  "RNN_TYPE": "GRU",
  "RNN_HIDDEN_SIZE": 768,
  "RNN_LAYERS": 2,
  "RNN_DROPOUT": 0.5,
  "RNN_BIDIRECTIONAL": true,
  "ONSET_LOSS_WEIGHT": 9.0,
  "ONSET_POS_WEIGHT_MANUAL_VALUE": 6.0
}
```

### Inference

```python
from model.architecture import GuitarTabCRNN
from model.utils import load_best_model

# Load trained model
model = load_best_model(
    model_class=GuitarTabCRNN,
    model_path="path/to/best_model.pth",
    run_config_path="path/to/run_configuration.json",
    device="cuda"
)

# Generate predictions
onset_logits, fret_logits = model(cqt_features)
```

## Project Structure

```
music-transcription/
├── python/
│   ├── config.py                 # Global configuration
│   ├── guitar_ATM.ipynb          # Main training notebook
│   ├── hyperparam_set_v1.json    # Hyperparameter configurations
│   ├── requirements.txt          # Python dependencies
│   │
│   ├── data_processing/
│   │   ├── batching.py           # Collate functions for DataLoader
│   │   ├── dataset.py            # GuitarSetTabDataset + augmentations
│   │   ├── preparation.py        # Data splitting & preprocessing
│   │   └── validation.py         # Data integrity checks
│   │
│   ├── model/
│   │   ├── architecture.py       # TabCNN + GuitarTabCRNN definitions
│   │   └── utils.py              # Model loading utilities
│   │
│   ├── training/
│   │   ├── epoch_processing.py   # Train/eval epoch logic
│   │   ├── loss_functions.py     # CombinedLoss (onset + fret)
│   │   ├── note_conversion_utils.py  # Frame-to-note conversion
│   │   └── pipeline.py           # Training loop orchestration
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # MPE, TDR, onset metrics
│   │   ├── performance_metrics.py  # Test set evaluation
│   │   ├── post_run_analysis.py  # Results aggregation
│   │   └── tablature_export.py   # ASCII/MIDI export
│   │
│   └── vizualization/
│       └── plotting.py           # Training curves & spectrograms
│
└── thesis/                       # LaTeX Master's thesis (Polish)
    ├── chapters/                 # Thesis chapters (00-08)
    ├── config/                   # LaTeX configuration
    ├── fig/                      # Figures and diagrams
    └── biblio/                   # Bibliography
```

## Results

### Quantitative Evaluation

| Metric       | Score  | Description                           |
| ------------ | ------ | ------------------------------------- |
| **MPE F1**   | 0.8736 | Multi-Pitch Estimation (frame-level)  |
| **TDR F1**   | 0.8569 | Tablature Detection Rate (note-level) |
| **Onset F1** | ~0.85  | Note onset detection accuracy         |

### Data Augmentation Impact

The aggressive augmentation strategy was identified as the **decisive factor** for model performance. Augmentations simulate amateur recording conditions:

| Augmentation   | Probability | Parameters               |
| -------------- | ----------- | ------------------------ |
| Time Stretch   | 60%         | Rate: 0.8-1.2            |
| Noise Addition | 70%         | Level: 0.001-0.01        |
| Random Gain    | 70%         | Factor: 0.6-1.4          |
| Reverb         | 40%         | Decay: 0.1-0.45s         |
| EQ (Bandpass)  | 50%         | 250-400 Hz to 3-4.5 kHz  |
| Clipping       | 30%         | Threshold: 0.5-0.9       |
| SpecAugment    | ✓           | Time: 40, Freq: 26 masks |

### Comparison with State-of-the-Art

This implementation achieves results **superior to previously published methods** for models trained exclusively on the GuitarSet dataset, demonstrating the effectiveness of the data-centric augmentation approach.

## Thesis Documentation

This repository includes the complete **Master's thesis** (in Polish) documenting the research:

**Title**: _Przekształcanie nagrań dźwiękowych na zapis nutowy_  
(_Transforming Audio Recordings into Musical Notation_)

**Institution**: Silesian University of Technology, Faculty of Automatic Control, Electronics and Computer Science

### Thesis Structure

| Chapter | Title          | Content                               |
| ------- | -------------- | ------------------------------------- |
| 1       | Introduction   | Problem formulation, MIR context      |
| 2       | Topic Analysis | Signal processing, ATM theory         |
| 3       | System Design  | Architecture, dataset, implementation |
| 4       | Experiments    | Methodology, results, analysis        |
| 5       | Conclusions    | Summary, future directions            |

## References

Key publications that influenced this work:

- **GuitarSet**: Xi et al. (2018) - Dataset creation and baseline methods
- **TabCNN**: Wiggins & Kim (2019) - CNN-based tablature transcription
- **Onsets and Frames**: Hawthorne et al. (2018) - Multi-task learning paradigm
- **AMT Survey**: Benetos et al. (2019) - Comprehensive ATM overview

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> 🤖 _This README was generated with the assistance of AI._
