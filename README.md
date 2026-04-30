# AVT-CA – Multimodal Emotion Recognition

Reproduction of:  
**"Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention"**  
Venkatraman et al. (arXiv 2407.18552)

---

## 📁 Project Structure

```
avtca/
├── config.py                    ← ALL hyperparameters (edit paths here)
├── train.py                     ← Training script
├── evaluate.py                  ← Evaluation + cross-dataset transfer
├── sanity_check.py              ← Run this first to verify setup
├── kaggle_notebook.py           ← Ready-to-paste Kaggle cells
├── requirements.txt
└── src/
    ├── data/
    │   └── ravdess_dataset.py   ← RAVDESS loader (audio + video)
    ├── models/
    │   └── avtca.py             ← Full AVT-CA architecture
    └── utils/
        ├── metrics.py           ← Accuracy, F1, confusion matrix
        └── visualization.py     ← Training curves, CM plots
```

---

## ⚙️ What to Download / Install

### 1. Python packages
```bash
pip install -r requirements.txt
```

**Key dependencies:**
| Package | Purpose |
|---|---|
| `torch` / `torchvision` / `torchaudio` | Model, video/audio loading |
| `opencv-python` (`cv2`) | Video frame extraction |
| `scikit-learn` | F1, confusion matrix |
| `kagglehub` | Download RAVDESS from Kaggle |
| `ffmpeg` (system) | Audio extraction from .mp4 |

### 2. System-level: ffmpeg
Audio is extracted from .mp4 using ffmpeg.

**Ubuntu / Lightning AI:**
```bash
sudo apt-get install -y ffmpeg
```

**Kaggle notebooks:** ffmpeg is pre-installed.

### 3. RAVDESS dataset

**Option A – Kaggle Hub (recommended for Kaggle/Lightning AI):**
```python
import kagglehub
path = kagglehub.dataset_download("orvile/ravdess-dataset")
print(path)   # copy this into config.py → DATA_ROOT
```

**Option B – Manual download:**  
Download from https://www.kaggle.com/datasets/orvile/ravdess-dataset  
Unzip so that the folder structure is:
```
RAVDESS/
  Actor_01/
    01-01-01-01-01-01-01.mp4
    01-01-03-01-01-01-01.mp4
    ...
  Actor_02/
    ...
```

---

## 🚀 Running on Kaggle / Lightning AI

### Step 1 – Edit `config.py`
```python
DATA_ROOT      = "/kaggle/input/ravdess-dataset"   # Kaggle
# DATA_ROOT    = "./data/RAVDESS"                  # Lightning AI / local
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR        = "/kaggle/working/logs"
CACHE_DIR      = "/kaggle/working/cache"            # speeds up 2nd+ runs
```

### Step 2 – Sanity check (always do this first)
```bash
python sanity_check.py
```
Expected output: all checks ✓, last line says "Run: python train.py"

### Step 3 – Train
```bash
python train.py
# Override any config value on the CLI:
python train.py --data ./data/RAVDESS --epochs 128 --batch_size 8
```

Training takes ~72 hours on the hardware from the paper (AMD EPYC + no GPU noted).  
On a single A100 GPU expect ~4–8 hours for 128 epochs.

### Step 4 – Evaluate
```bash
# RAVDESS validation split
python evaluate.py --model checkpoints/best_model.pt --dataset ravdess

# Cross-dataset: CREMA-D (zero-shot transfer)
python evaluate.py \
    --model checkpoints/best_model.pt \
    --dataset cremad \
    --data ./data/CREMA-D

# Cross-dataset: CMU-MOSEI
python evaluate.py \
    --model checkpoints/best_model.pt \
    --dataset cmumosei \
    --data ./data/CMU-MOSEI
```

---

## 🏗️ Architecture Overview

```
Audio (mel spectrogram)           Video (frames)
    │                                   │
AudioEncoder                     VideoEncoder
  2× Conv1D block                  per-frame CNN:
  (Conv → BN → ReLU → Pool)          Conv2D
    │                                 Channel Attention (SE)
    │                                 Spatial Attention
    │                                 Local Feature Extractor
    │                                 2× Inverted Residual Block
    │                                   │
    ▼                                   ▼
(B, T_a', d_model)             (B, T_v, d_model)
    │                                   │
    └──────── Intermediate Transformer ─┘
              (IT-4: 4-head, 2 layers each branch)
                    │           │
                    └── Cross-Self Attention (CT-4) ──┘
                              │           │
                        Final Cross Attention
                              │           │
                          MaxPool     MaxPool
                              └────+────┘
                                   │
                              FC → Softmax
                                   │
                             Emotion label
```

---

## 📊 Expected Results (paper Table II)

| Dataset  | Accuracy | F1-Score |
|----------|----------|----------|
| RAVDESS  | 96.11%   | 93.78%   |
| CMU-MOSEI| 95.84%   | 94.13%   |
| CREMA-D  | 94.13%   | 94.67%   |

---

## 💾 Saved Files

After training, `checkpoints/` contains:

| File | Description |
|---|---|
| `best_model.pt` | Highest validation accuracy — **use this for inference** |
| `last_model.pt` | Final epoch model |
| `checkpoint_ep{N:04d}.pt` | Periodic checkpoints |
| `config_used.json` | Config snapshot |
| `history.json` | Per-epoch metrics (acc, f1, loss) |

### Loading for inference / deployment

```python
import torch
from config import Config
from src.models.avtca import AVTCA

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

# Input: audio (1, n_mels, T_a)  +  video (1, T_v, 3, H, W)
with torch.no_grad():
    logits = model(audio_tensor, video_tensor)
    predicted_class = logits.argmax(dim=1).item()
    emotion = cfg.EMOTION_LABELS[predicted_class]
```

---

## 🐛 Debugging Tips

| Problem | Likely cause | Fix |
|---|---|---|
| `No valid files found` | Wrong `DATA_ROOT` or `MODALITY_FILTER` | Check path; set `MODALITY_FILTER = ""` to use all files |
| `ffmpeg not found` | System ffmpeg missing | `sudo apt-get install ffmpeg` |
| `CUDA out of memory` | Batch too large | Reduce `BATCH_SIZE` to 4 or 2 |
| Low accuracy (< 80%) | Too few files loaded | Verify `MODALITY_FILTER = "01"` only loads full-AV files; check total sample count in output |
| Very slow first epoch | No cache yet | After epoch 1, cache fills and speed improves; or set `CACHE_DIR = ""` to disable |

---

## 📚 Citation

```bibtex
@article{venkatraman2024avtca,
  title   = {Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention},
  author  = {Venkatraman, Shravan and Dhanith, Joe and Sharma, Vigya and Malarvannan, Santhosh},
  journal = {arXiv preprint arXiv:2407.18552},
  year    = {2024}
}
```
"# Speech-Emotion-Recognition-SER" 
