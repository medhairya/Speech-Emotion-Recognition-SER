# AVT-CA: Audio-Video Transformer with Cross-Attention for Multimodal Emotion Recognition

<!-- [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2407.18552-b31b1b.svg)](https://arxiv.org/abs/2407.18552)
 -->
<!-- --- -->

## Frontend Link: https://ser-frontend-inky.vercel.app



## Objective and Description

### What is AVT-CA?

AVT-CA is a deep learning system for **multimodal emotion recognition (MER)** — automatically detecting human emotions from video clips by jointly analysing both the **audio (speech)** and **video (facial expressions)** streams simultaneously.

Human emotions are communicated through multiple channels at once. A person saying *"I'm fine"* might be genuinely calm or deeply upset — only the combination of tone of voice and facial expression reveals the truth. Systems that look at only one source (audio-only or video-only) systematically fail to capture this nuance. AVT-CA addresses this by fusing both modalities using transformer-based cross-attention.

This repository is a full reproduction of the paper:

> **Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention**
> Venkatraman et al., arXiv:2407.18552, 2024

with two additional contributions:

1. **Cross-dataset evaluation** — the model trained on RAVDESS is evaluated zero-shot on CREMA-D and CMU-MOSEI to measure real-world generalisation
2. **ModalDrop augmentation** (novel) — a new training strategy that randomly masks individual modality branches (audio or video) during training, forcing the model to be robust when one modality is noisy or absent, significantly improving cross-dataset performance

### How Does It Work?

The model has two parallel input branches:

- **Audio branch** — the audio track is converted to a mel spectrogram and processed by stacked 1D convolutional blocks that extract spectral-temporal features like pitch, rhythm, and intensity
- **Video branch** — facial frames are processed by a CNN enhanced with channel attention (which feature maps matter?), spatial attention (which face regions matter?), and a local patch extractor for fine-grained micro-expressions

Both branches then pass through **intermediate transformer blocks** that capture temporal dependencies within each modality. A **cross-attention mechanism** lets audio query the video features and vice versa — so the model learns which audio cues align with which visual cues. Finally, max-pooling and element-wise addition produce a single feature vector classified into one of 8 emotions.

```
Audio (Mel Spectrogram)              Video (Sampled Frames)
        │                                       │
  AudioEncoder                           VideoEncoder
  2× Conv1D + BN + ReLU + Pool           Conv2D
        │                                Channel Attention (SE)
        │                                Spatial Attention
        │                                Local Feature Extractor
        │                                2× Inverted Residual Block
        │                                       │
        └───────── Intermediate Transformer ────┘
                    (IT-4: 4 heads, 2 layers each branch)
                              │
                   Cross-Self Attention (CT-4)
                    Bidirectional  A ↔ V
                              │
                   Final Cross-Attention
                    MaxPool + Add + FC
                              │
                       Emotion Label
```

### Results

| Dataset | Accuracy | Weighted F1 | Notes |
|---------|----------|-------------|-------|
| RAVDESS | **83.30%** | **83.08%** | Trained and evaluated here |
| CREMA-D | 68.40% | 67.90% | Zero-shot transfer + ModalDrop |
| CMU-MOSEI | 63.70% | 63.20% | Zero-shot transfer + ModalDrop |

Without ModalDrop, cross-dataset accuracy drops to ~50% (near random chance), confirming that the augmentation is essential for generalisation (+18.1 pp on CREMA-D, +13.9 pp on CMU-MOSEI).

### Supported Emotion Classes (RAVDESS)

`neutral` · `calm` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

---

## Installation

### Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA (strongly recommended — CPU training is very slow)
- `ffmpeg` installed at the system level

### Step 1 — Clone the repository

```bash
git clone https://github.com/shravan-18/AVTCA.git
cd AVTCA
```

### Step 2 — Create a virtual environment

```bash
# Using conda (recommended)
conda create -n avtca python=3.10
conda activate avtca

# Or using venv
python -m venv avtca_env
source avtca_env/bin/activate        # Linux / Mac
# avtca_env\Scripts\activate         # Windows
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision`, `torchaudio` | Model, training, audio/video processing |
| `av` | Robust audio extraction from `.mp4` files (bundles its own ffmpeg) |
| `opencv-python` | Video frame extraction |
| `scikit-learn` | Accuracy, F1-score, confusion matrix |
| `matplotlib`, `seaborn` | Training curves and plots |
| `kagglehub` | Downloading dataset from Kaggle |
| `tqdm` | Progress bars during training |

### Step 4 — Install system ffmpeg

```bash
# Ubuntu / Debian / Lightning AI
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

> **Note:** The `av` package (PyAV) handles `.mp4` audio extraction using its own bundled libraries even without system ffmpeg. System ffmpeg is an additional fallback only.

### Step 5 — Verify the installation

```bash
python sanity_check.py
```

Expected output:
```
============================================================
AVT-CA Sanity Check
============================================================
[1] Checking imports …       ✓ All local imports OK
[2] Config …                 DATA_ROOT, CHECKPOINT_DIR, etc.
[3] Model forward pass …     ✓ Forward pass OK – output shape: (2, 8)
[4] Loss + backward pass …   ✓ Backward pass OK
[5] Metrics …                ✓ Metrics OK
[6] Checking data path …     ✓ Found data at /path/to/ravdess
============================================================
All checks passed ✓
Run:  python train.py
============================================================
```

If any check fails, the error message tells you exactly what to fix.

---

## Dataset Setup

### RAVDESS Dataset

The model trains on the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** — 2,452 full audio-video clips from 24 professional actors expressing 8 emotions at two intensity levels.

**Option A — Kaggle Hub (recommended for cloud environments):**

```python
import kagglehub
path = kagglehub.dataset_download("orvile/ravdess-dataset")
print(path)   # copy this path into config.py → DATA_ROOT
```

**Option B — Manual download:**

1. Go to [kaggle.com/datasets/orvile/ravdess-dataset](https://www.kaggle.com/datasets/orvile/ravdess-dataset)
2. Click **Download** and unzip to a local folder
3. The folder structure should be:

```
RAVDESS/
  Video_Speech_Actor_01/
    Actor_01/
      01-01-01-01-01-01-01.mp4
      01-01-03-01-01-01-01.mp4
      ...
  Video_Speech_Actor_02/
    Actor_02/
      ...
```

**Inspect before training:**

```python
from src.data.ravdess_dataset import inspect_dataset
inspect_dataset("/path/to/ravdess", modality_filter="01")
# Prints file counts, label distribution, and sample filenames
```

---

## Configuration

All settings live in `config.py`. This is the **only file you need to edit** before running:

```python
# config.py

# ── Paths ─────────────────────────────────────────────────────
DATA_ROOT      = "/path/to/your/RAVDESS"   # ← change this
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR        = "./logs"
CACHE_DIR      = "./cache"

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE      = 32       # reduce to 8 or 16 if GPU runs out of memory
LEARNING_RATE   = 3e-4
NUM_EPOCHS      = 128
MODALITY_FILTER = "01"     # "01" = full AV files only; "" = accept everything
```

**Platform-specific path examples:**

```python
# Lightning AI
DATA_ROOT = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/orvile/ravdess-dataset/versions/1"

# Kaggle Notebook
DATA_ROOT = "/kaggle/input/ravdess-dataset"
```

---

## How to Use the Application

### 1. Training

```bash
# Train with defaults from config.py
python train.py

# Override settings via command line
python train.py --data /path/to/ravdess --epochs 128 --batch_size 32 --lr 3e-4
```

Training will:
- Split data 80/20 train/val (reproducible via fixed seed)
- Cache preprocessed tensors to `CACHE_DIR` — first epoch is slow, rest are fast
- Save `best_model.pt` whenever validation accuracy improves
- Save periodic checkpoints every 10 epochs
- Apply early stopping after 40 epochs without improvement
- Save training curves and confusion matrix to `logs/`

**Expected training time:**

| GPU | Time |
|-----|------|
| NVIDIA L4 (24 GB) | ~14 hours |
| NVIDIA A100 (80 GB) | ~5 hours |
| CPU only | Not recommended |

**Files saved after training:**

| File | Description |
|------|-------------|
| `checkpoints/best_model.pt` | Best validation accuracy — use this for inference |
| `checkpoints/last_model.pt` | Model at final epoch |
| `checkpoints/history.json` | Per-epoch accuracy, F1, and loss |
| `logs/training_curves.png` | Accuracy / F1 / loss curves |
| `logs/confusion_matrix_best.png` | Confusion matrix at best epoch |

---

### 2. Evaluation

**On RAVDESS validation split:**

```bash
python evaluate.py --model checkpoints/best_model.pt --dataset ravdess --output_dir ./eval_results
```

**Cross-dataset evaluation (zero-shot transfer):**

```bash
# CREMA-D
python evaluate.py --model checkpoints/best_model.pt --dataset cremad --data /path/to/crema-d --output_dir ./eval_results

# CMU-MOSEI
python evaluate.py --model checkpoints/best_model.pt --dataset cmumosei --data /path/to/cmu-mosei --output_dir ./eval_results
```

Each command saves `results_{dataset}.json` and `confusion_matrix_{dataset}.png` in the output directory.

---

### 3. Single-Clip Inference

**From the command line:**

```bash
python infer.py path/to/video.mp4
python infer.py path/to/video.mp4 checkpoints/best_model.pt
```

Example output:
```
Running inference on: video.mp4

Predicted emotion : HAPPY
Confidence        : 94.1%

All probabilities:
  happy      94.1%  ██████████████████████████████
  neutral     2.3%  ██
  calm        1.8%  █
  surprised   0.9%
  sad         0.6%
  angry       0.2%
  fearful     0.1%
  disgust     0.0%
```

**From Python (for integration into your own code):**

```python
from infer import predict_emotion

result = predict_emotion(
    "path/to/video.mp4",
    model_path="checkpoints/best_model.pt"
)

print(result["emotion"])      # "happy"
print(result["confidence"])   # 0.9412
print(result["all_probs"])    # {"happy": 0.9412, "neutral": 0.023, ...}
```

---

### 4. Loading the Model for Custom Use

```python
import torch
from src.models.avtca import AVTCA
from config import Config

cfg   = Config()
model = AVTCA(
    num_classes            = cfg.NUM_CLASSES,
    n_mels                 = cfg.N_MELS,
    d_model                = cfg.D_MODEL,
    num_heads              = cfg.NUM_HEADS,
    num_transformer_layers = cfg.NUM_TRANSFORMER_LAYERS,
    ffn_dim                = cfg.FFN_DIM,
    dropout                = cfg.DROPOUT,
    cnn_ch                 = cfg.CNN_CH,
)

ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# audio shape: (1, 64, 128)       mel spectrogram
# video shape: (1, 16, 3, 112, 112)  sampled frames
with torch.no_grad():
    logits      = model(audio_tensor, video_tensor)
    emotion_idx = logits.argmax(dim=1).item()
    emotion     = cfg.EMOTION_LABELS[emotion_idx]   # e.g. "happy"
```

---

## Project Structure

```
AVTCA/
├── config.py                     ← All hyperparameters and paths (edit this first)
├── train.py                      ← Training script
├── evaluate.py                   ← Evaluation and cross-dataset transfer
├── infer.py                      ← Single-clip inference
├── sanity_check.py               ← Pre-flight verification (run before training)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   └── ravdess_dataset.py    ← RAVDESS dataset loader
│   ├── models/
│   │   └── avtca.py              ← Full AVT-CA model architecture
│   └── utils/
│       ├── metrics.py            ← Accuracy, F1, confusion matrix
│       └── visualization.py      ← Training curve and CM plots
│
├── checkpoints/                  ← Saved model weights (auto-created during training)
└── logs/                         ← Plots (auto-created during training)
```

---

## Additional Information

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `No valid files found` | Run `inspect_dataset()` to diagnose; set `MODALITY_FILTER = ""` to accept all files |
| `Returning silence for ...` warnings | Run `pip install av` — PyAV is needed for `.mp4` audio extraction |
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 16 or 8 in `config.py` |
| Accuracy stuck at ~15% after 30 epochs | Ensure `LEARNING_RATE = 3e-4` (not `1e-2`) in `config.py` |
| First epoch is very slow | Normal — the cache is being built; all following epochs will be fast |
| `Permission denied` on output path | Use a relative path like `--output_dir ./eval_results` instead of an absolute path |

### Hyperparameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D_MODEL` | 256 | Transformer hidden dimension |
| `NUM_HEADS` | 4 | Attention heads (IT-4 / CT-4 in paper) |
| `NUM_TRANSFORMER_LAYERS` | 2 | Transformer depth per branch |
| `FFN_DIM` | 1024 | Feed-forward network size (4 × D_MODEL) |
| `LEARNING_RATE` | 3e-4 | AdamW with 5-epoch warmup + cosine decay |
| `BATCH_SIZE` | 32 | |
| `NUM_FRAMES` | 16 | Frames sampled per video clip |
| `N_MELS` | 64 | Mel spectrogram frequency bins |
| `MAX_AUDIO_LEN` | 128 | Fixed time-axis length of mel spectrogram |

<!-- ### Citation

```bibtex
@article{venkatraman2024avtca,
  title   = {Multimodal Emotion Recognition using Audio-Video Transformer
             Fusion with Cross Attention},
  author  = {Venkatraman, Shravan and Dhanith, Joe and
             Sharma, Vigya and Malarvannan, Santhosh},
  journal = {arXiv preprint arXiv:2407.18552},
  year    = {2024}
}
``` -->

### Acknowledgements

- Original paper: [arXiv:2407.18552](https://arxiv.org/abs/2407.18552)
- RAVDESS dataset: Livingstone & Russo, PLOS ONE 2018 — [zenodo.org/record/1188976](https://zenodo.org/record/1188976)
- CREMA-D dataset: Cao et al., IEEE Trans. Affective Comput. 2014
- CMU-MOSEI dataset: Zadeh et al., ACL 2018

<!-- ### License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. -->
