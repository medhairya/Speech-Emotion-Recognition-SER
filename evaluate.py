"""
evaluate.py
───────────
Load a trained RAVDESS model and evaluate on the same dataset (val split)
or on a cross-dataset (CMU-MOSEI / CREMA-D).

Usage:
    # Evaluate on RAVDESS val split (sanity check)
    python evaluate.py --model checkpoints/best_model.pt --dataset ravdess

    # Cross-dataset evaluation (zero-shot transfer)
    python evaluate.py --model checkpoints/best_model.pt --dataset cremad  --data ./data/CREMA-D
    python evaluate.py --model checkpoints/best_model.pt --dataset cmumosei --data ./data/CMU-MOSEI

NOTE:
    CMU-MOSEI uses 6 emotions; CREMA-D uses 6 emotions.
    The model was trained on RAVDESS 8-class.
    In cross-dataset mode we map target dataset labels to the closest
    RAVDESS label indices (see CROSS_DATASET_MAPS below).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from src.models.avtca import AVTCA
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_confusion_matrix
from src.data.ravdess_dataset import RAVDESSDataset, EMOTION_NAMES, _load_waveform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Cross-dataset label mappings
# ─────────────────────────────────────────────────────────────

# Map emotion name → RAVDESS label index (0-based)
RAVDESS_NAME_TO_IDX = {n: i for i, n in enumerate(EMOTION_NAMES)}

# CREMA-D emotion names (anger disgust fear happy neutral sad)
CREMAD_EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
CREMAD_TO_RAVDESS = {i: RAVDESS_NAME_TO_IDX[n] for i, n in enumerate(CREMAD_EMOTIONS)}
# Maps CREMA-D label (0-5) → RAVDESS label (0-7)

# CMU-MOSEI emotion names (happy sad angry fearful disgusted surprised)
CMUMOSEI_EMOTIONS = ["happy", "sad", "angry", "fearful", "disgust", "surprised"]
CMUMOSEI_TO_RAVDESS = {i: RAVDESS_NAME_TO_IDX[n] for i, n in enumerate(CMUMOSEI_EMOTIONS)}


# ─────────────────────────────────────────────────────────────
# Generic video/audio dataset for cross-dataset evaluation
# ─────────────────────────────────────────────────────────────

class GenericVideoDataset(Dataset):
    """
    Simple loader for a directory of video files where the label is
    encoded in the filename according to `label_fn(path)`.

    Args:
        root_dir    : directory containing video files (searched recursively)
        label_fn    : callable(path) → int or None  (None = skip file)
        num_frames  : frames to sample per clip
        frame_size  : (H, W)
        n_mels      : mel bins
        sr          : audio sample rate
        max_audio_len: fixed mel time length
    """

    def __init__(
        self,
        root_dir,
        label_fn,
        num_frames=16,
        frame_size=(112, 112),
        n_mels=64,
        sr=22050,
        max_audio_len=128,
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.n_mels = n_mels
        self.sr = sr
        self.max_audio_len = max_audio_len

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=512, n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        self.samples = []
        for p in sorted(self.root_dir.rglob("*.mp4")):
            lbl = label_fn(p)
            if lbl is not None:
                self.samples.append({"path": str(p), "label": lbl})
        for p in sorted(self.root_dir.rglob("*.wav")):
            lbl = label_fn(p)
            if lbl is not None:
                self.samples.append({"path": str(p), "label": lbl})

        log.info(f"GenericVideoDataset: {len(self.samples)} samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = self._load_audio(item["path"])
        video = self._load_video(item["path"])
        return audio, video, item["label"]

    def _load_audio(self, path):
        waveform, orig_sr = _load_waveform(path, self.sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel).squeeze(0)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        T_cur = mel.shape[1]
        if T_cur >= self.max_audio_len:
            mel = mel[:, :self.max_audio_len]
        else:
            mel = F.pad(mel, (0, self.max_audio_len - T_cur))
        return mel

    def _load_video(self, path):
        if path.endswith(".wav"):
            return torch.zeros(self.num_frames, 3, *self.frame_size)
        cap = cv2.VideoCapture(path)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, self.frame_size[::-1])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()
        frames = np.stack(frames).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frames = (frames - mean) / (std + 1e-8)
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float()


# ─────────────────────────────────────────────────────────────
# Per-dataset label parsers
# ─────────────────────────────────────────────────────────────

def cremad_label_fn(path: Path) -> int:
    """
    CREMA-D filename: {actor}_{sentence}_{emotion}_{level}.mp4
    e.g. 1001_DFA_ANG_XX.mp4
    Emotion codes: ANG=angry, DIS=disgust, FEA=fear,
                   HAP=happy, NEU=neutral, SAD=sad
    Returns RAVDESS-mapped label.
    """
    CREMAD_CODE = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    code = parts[2].upper()
    local_lbl = CREMAD_CODE.get(code)
    if local_lbl is None:
        return None
    return CREMAD_TO_RAVDESS[local_lbl]


def cmumosei_label_fn(path: Path) -> int:
    """
    CMU-MOSEI files are usually pre-labelled. This function expects files
    to be in subdirectories named after the emotion:
        ./happy/video1.mp4
        ./sad/video2.mp4
    Adjust if your CMU-MOSEI layout is different.
    Returns RAVDESS-mapped label.
    """
    parent = path.parent.name.lower()
    local_lbl = None
    for i, name in enumerate(CMUMOSEI_EMOTIONS):
        if parent.startswith(name):
            local_lbl = i
            break
    if local_lbl is None:
        return None
    return CMUMOSEI_TO_RAVDESS[local_lbl]


# ─────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device, class_names=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_trues, all_losses = [], [], []

    with torch.no_grad():
        for audio, video, labels in tqdm(loader, desc="Evaluating"):
            audio  = audio.to(device)
            video  = video.to(device)
            labels = labels.to(device)

            logits = model(audio, video)
            loss   = criterion(logits, labels)

            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_trues.extend(labels.cpu().tolist())
            all_losses.append(loss.item())

    metrics = compute_metrics(all_trues, all_preds, class_names)
    metrics["avg_loss"] = float(np.mean(all_losses))
    return metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dataset", default="ravdess",
                        choices=["ravdess", "cremad", "cmumosei"],
                        help="Dataset to evaluate on")
    parser.add_argument("--data",    default="", help="Data root (for cross-dataset)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="./eval_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    cfg = Config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
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
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded model from {args.model}  (epoch {ckpt.get('epoch', '?')})")

    # ── Build dataset ─────────────────────────────────────────
    loader_kwargs = dict(
        num_frames    = cfg.NUM_FRAMES,
        frame_size    = (cfg.FRAME_H, cfg.FRAME_W),
        n_mels        = cfg.N_MELS,
        sr            = cfg.SAMPLE_RATE,
        max_audio_len = cfg.MAX_AUDIO_LEN,
    )

    if args.dataset == "ravdess":
        data_root = args.data or cfg.DATA_ROOT
        ds = RAVDESSDataset(
            root_dir        = data_root,
            split           = "val",
            train_ratio     = cfg.TRAIN_RATIO,
            modality_filter = cfg.MODALITY_FILTER,
            seed            = cfg.SEED,
            cache_dir       = cfg.CACHE_DIR,
            **loader_kwargs,
        )
        class_names = EMOTION_NAMES
        dataset_label = "RAVDESS (val)"

    elif args.dataset == "cremad":
        data_root = args.data
        if not data_root:
            log.error("Provide --data path for CREMA-D")
            sys.exit(1)
        ds = GenericVideoDataset(data_root, cremad_label_fn, **loader_kwargs)
        class_names = EMOTION_NAMES  # labels are mapped to RAVDESS space
        dataset_label = "CREMA-D → RAVDESS label space"

    elif args.dataset == "cmumosei":
        data_root = args.data
        if not data_root:
            log.error("Provide --data path for CMU-MOSEI")
            sys.exit(1)
        ds = GenericVideoDataset(data_root, cmumosei_label_fn, **loader_kwargs)
        class_names = EMOTION_NAMES
        dataset_label = "CMU-MOSEI → RAVDESS label space"

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    # ── Evaluate ──────────────────────────────────────────────
    log.info(f"Evaluating on {dataset_label} …")
    metrics = evaluate(model, loader, device, class_names)

    log.info("\n" + "=" * 60)
    log.info(f"Dataset : {dataset_label}")
    log.info(f"Accuracy: {metrics['accuracy'] * 100:.2f} %")
    log.info(f"F1 (wt) : {metrics['f1_weighted'] * 100:.2f} %")
    log.info(f"Avg Loss: {metrics['avg_loss']:.4f}")
    log.info("\n" + metrics["classification_report"])
    log.info("=" * 60)

    # ── Save results ──────────────────────────────────────────
    result_file = output_dir / f"results_{args.dataset}.json"
    safe_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    result_file.write_text(json.dumps(safe_metrics, indent=2))
    log.info(f"Saved metrics → {result_file}")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names[: cfg.NUM_CLASSES],
        str(output_dir / f"confusion_matrix_{args.dataset}.png"),
        title=f"Confusion Matrix – {dataset_label}",
    )


if __name__ == "__main__":
    main()
