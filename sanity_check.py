"""
sanity_check.py
───────────────
Run this BEFORE training to verify:
  1. All imports work
  2. Model forward pass runs without errors
  3. Dataset can load (if data path exists)
  4. Shapes are correct throughout

Usage:
    python sanity_check.py
"""

import torch
import sys

print("=" * 60)
print("AVT-CA Sanity Check")
print("=" * 60)

# ── 1. Imports ────────────────────────────────────────────────
print("\n[1] Checking imports …")
try:
    from config import Config
    from src.models.avtca import AVTCA
    from src.utils.metrics import RunningMetrics, compute_metrics
    print("    ✓ All local imports OK")
except ImportError as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)

try:
    import cv2
    import torchaudio
    import torchvision
    import sklearn
    import numpy as np
    print("    ✓ External packages OK")
except ImportError as e:
    print(f"    ✗ Missing package: {e}")
    print("    Run:  pip install -r requirements.txt")
    sys.exit(1)

# ── 2. Config ─────────────────────────────────────────────────
print("\n[2] Config …")
cfg = Config()
print(f"    DATA_ROOT      : {cfg.DATA_ROOT}")
print(f"    CHECKPOINT_DIR : {cfg.CHECKPOINT_DIR}")
print(f"    D_MODEL        : {cfg.D_MODEL}")
print(f"    NUM_CLASSES    : {cfg.NUM_CLASSES}")
print(f"    BATCH_SIZE     : {cfg.BATCH_SIZE}")
print(f"    NUM_EPOCHS     : {cfg.NUM_EPOCHS}")

# ── 3. Model forward pass ─────────────────────────────────────
print("\n[3] Model forward pass (random inputs) …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

model = AVTCA(
    num_classes            = cfg.NUM_CLASSES,
    n_mels                 = cfg.N_MELS,
    d_model                = cfg.D_MODEL,
    num_heads              = cfg.NUM_HEADS,
    num_transformer_layers = cfg.NUM_TRANSFORMER_LAYERS,
    ffn_dim                = cfg.FFN_DIM,
    dropout                = cfg.DROPOUT,
    cnn_ch                 = cfg.CNN_CH,
).to(device)

n_params = model.count_parameters()
print(f"    Trainable parameters: {n_params:,}")

B = 2
audio = torch.randn(B, cfg.N_MELS, cfg.MAX_AUDIO_LEN).to(device)
video = torch.randn(B, cfg.NUM_FRAMES, 3, cfg.FRAME_H, cfg.FRAME_W).to(device)

with torch.no_grad():
    logits = model(audio, video)

assert logits.shape == (B, cfg.NUM_CLASSES), \
    f"Expected ({B}, {cfg.NUM_CLASSES}), got {logits.shape}"
print(f"    ✓ Forward pass OK – output shape: {tuple(logits.shape)}")

# ── 4. Loss & backward ────────────────────────────────────────
print("\n[4] Loss + backward pass …")
labels = torch.randint(0, cfg.NUM_CLASSES, (B,)).to(device)
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits, labels)

# Need grad for backward
model.train()
logits2 = model(audio, video)
loss2    = criterion(logits2, labels)
loss2.backward()
print(f"    ✓ Backward pass OK – loss = {loss2.item():.4f}")

# ── 5. Metrics ────────────────────────────────────────────────
print("\n[5] Metrics …")
rm = RunningMetrics()
rm.update(logits.cpu(), labels.cpu(), loss=loss.item())
m  = rm.compute(cfg.EMOTION_LABELS)
print(f"    accuracy  : {m['accuracy']:.4f}")
print(f"    f1_weighted: {m['f1_weighted']:.4f}")
print("    ✓ Metrics OK")

# ── 6. Data path ──────────────────────────────────────────────
print("\n[6] Checking data path …")
from pathlib import Path
dp = Path(cfg.DATA_ROOT)
if dp.exists():
    mp4_files = list(dp.rglob("*.mp4"))[:5]
    print(f"    ✓ Found data at {dp}")
    print(f"    First .mp4 files: {[f.name for f in mp4_files]}")
else:
    print(f"    ⚠  DATA_ROOT not found: {dp}")
    print("    Edit config.py → DATA_ROOT before training.")

# ── Done ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All checks passed ✓")
print("Run:  python train.py")
print("=" * 60)
