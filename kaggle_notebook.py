# kaggle_notebook.py
# ═══════════════════════════════════════════════════════════
#  AVT-CA — Kaggle / Lightning AI single-file runner
#  Paste each cell into a Kaggle notebook or run as a script.
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELL 1 – Install dependencies
# ─────────────────────────────────────────────────────────────
# !pip install kagglehub torchaudio torchvision opencv-python scikit-learn matplotlib seaborn tqdm --quiet

# ─────────────────────────────────────────────────────────────
# CELL 2 – Download RAVDESS dataset from Kaggle
# ─────────────────────────────────────────────────────────────
import kagglehub
import os

# This downloads to ~/.cache/kagglehub/datasets/orvile/ravdess-dataset/...
dataset_path = kagglehub.dataset_download("orvile/ravdess-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# List top-level contents so you can set DATA_ROOT correctly
for item in sorted(os.listdir(dataset_path))[:10]:
    print(" ", item)

# ─────────────────────────────────────────────────────────────
# CELL 3 – Configure paths
# ─────────────────────────────────────────────────────────────
# Point DATA_ROOT at the downloaded dataset.
# Adjust the sub-path if kagglehub creates an extra folder.
import os, sys
sys.path.insert(0, "/kaggle/working/avtca")  # repo root if you cloned there

DATA_ROOT      = dataset_path          # auto-set from kagglehub
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR        = "/kaggle/working/logs"
CACHE_DIR      = "/kaggle/working/cache"

print("DATA_ROOT     :", DATA_ROOT)
print("CHECKPOINT_DIR:", CHECKPOINT_DIR)

# ─────────────────────────────────────────────────────────────
# CELL 4 – Override config and run training
# ─────────────────────────────────────────────────────────────
from config import Config
from train  import train

cfg = Config()
cfg.DATA_ROOT      = DATA_ROOT
cfg.CHECKPOINT_DIR = CHECKPOINT_DIR
cfg.LOG_DIR        = LOG_DIR
cfg.CACHE_DIR      = CACHE_DIR
cfg.NUM_EPOCHS     = 128           # as in paper; reduce to 20 for quick test
cfg.BATCH_SIZE     = 8
cfg.NUM_WORKERS    = 2             # Kaggle has 2 CPUs per default

history = train(cfg)

# ─────────────────────────────────────────────────────────────
# CELL 5 – Evaluate on RAVDESS val split
# ─────────────────────────────────────────────────────────────
import subprocess
result = subprocess.run(
    [
        "python", "evaluate.py",
        "--model", f"{CHECKPOINT_DIR}/best_model.pt",
        "--dataset", "ravdess",
        "--data",  DATA_ROOT,
        "--output_dir", "/kaggle/working/eval_results",
    ],
    capture_output=False,
)

# ─────────────────────────────────────────────────────────────
# CELL 6 – (Optional) Cross-dataset eval on CREMA-D
# ─────────────────────────────────────────────────────────────
# Download CREMA-D first:
# cremad_path = kagglehub.dataset_download("ejlok1/cremad")
#
# subprocess.run([
#     "python", "evaluate.py",
#     "--model",   f"{CHECKPOINT_DIR}/best_model.pt",
#     "--dataset", "cremad",
#     "--data",    cremad_path,
#     "--output_dir", "/kaggle/working/eval_results",
# ])
