"""
RetinAgent - ViT-MAE Fine-tuning Script
AMD Developer Hackathon 2026 | Track 3: Vision & Multimodal AI

Fine-tunes facebook/vit-mae-base on APTOS 2019 dataset
for Diabetic Retinopathy Grade Classification (0-4).

Compatible with:
  - NVIDIA CUDA (local training)
  - AMD ROCm (PyTorch ROCm build)
  - CPU (slow, for debugging only)

Usage:
  python train.py
  python train.py --epochs 30 --batch_size 32 --data_dir ./Dataset
"""

import os
import argparse
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from transformers import ViTMAEModel
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────
# 1.  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────
# 2.  DEVICE  (works for CUDA / AMD ROCm / CPU)
# ─────────────────────────────────────────────────────────────────
# NOTE: AMD ROCm exposes GPUs as 'cuda' in PyTorch – no extra code needed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
# 2b. FOCAL LOSS  — improves rare class (Grade 3/4) detection
# ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples so model focuses on rare/hard cases.
    gamma=0 → standard CrossEntropy.  gamma=2 → standard for imbalanced datasets."""
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets,
                               label_smoothing=self.label_smoothing,
                               reduction='none')       # [B]
        pt   = torch.exp(-ce)                          # prob of correct class
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────
def circle_crop(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detects the fundus circular boundary and crops to its bounding box,
    removing uninformative black background (SRS FR-02, Step 1).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_bgr  # fallback: return original

    # Largest contour = fundus circle
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = img_bgr[y:y + h, x:x + w]
    return cropped


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """
    Applies CLAHE per-channel in LAB color space (SRS FR-02, Step 2).
    clip_limit=2.0, tile_grid_size=(8,8)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_image(image_path: str, size: int = 224) -> Image.Image:
    """
    Full preprocessing pipeline per SRS FR-02:
      1. Circle crop
      2. CLAHE
      3. Resize to 224×224
    Returns a PIL RGB image (normalization done by torchvision transforms).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_bgr = circle_crop(img_bgr)
    img_bgr = apply_clahe(img_bgr)
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ─────────────────────────────────────────────────────────────────
# 4.  DATASET
# ─────────────────────────────────────────────────────────────────
# ImageNet normalization – matches ViT-MAE pre-training (SRS FR-02, Step 4)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class APTOSDataset(Dataset):
    """
    APTOS 2019 dataset supporting subfolder structures.
    
    Folder structure handled:
        data_dir/
            train.csv
            colored_images/ (or train_images/)
                0/ (or No_DR/)
                1/ (or Mild/)
                ...
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        # Create a mapping of id_code to actual path for faster lookup
        print(f"[Dataset] Indexing images in {image_dir}...")
        self.image_path_map = {}
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.endswith('.png'):
                    code = f.replace('.png', '')
                    self.image_path_map[code] = os.path.join(root, f)
        print(f"[Dataset] Found {len(self.image_path_map)} images.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        id_code   = row["id_code"]
        label     = int(row["diagnosis"])
        
        # Get path from our pre-built map
        img_path = self.image_path_map.get(id_code)
        
        if img_path is None:
            # Fallback for original structure
            img_path = os.path.join(self.image_dir, f"{id_code}.png")

        try:
            img_pil = preprocess_image(img_path)
        except Exception as e:
            # If image fails (corrupt), return a blank image and log
            print(f"Error loading {id_code}: {e}")
            img_pil = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, label


# ─────────────────────────────────────────────────────────────────
# 5.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────
class DRClassifier(nn.Module):
    """
    ViT-MAE backbone + custom classification head (SRS §5.2).

    Architecture:
        ViTMAE Encoder  (768-dim CLS token output)
        → Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 5)

    Differential learning rates:
        backbone : 2e-5  (fine-tune pre-trained weights carefully)
        head     : 1e-3  (train new layers faster)
    """

    MODEL_NAME = "facebook/vit-mae-base"

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        print(f"[Model] Loading {self.MODEL_NAME} …")
        self.backbone = ViTMAEModel.from_pretrained(self.MODEL_NAME)

        # CRITICAL FIX: Disable random masking for classification.
        # ViTMAEModel masks 75% of patches by default — catastrophic for classification.
        self.backbone.config.mask_ratio = 0.0
        print("[Model] Masking disabled (mask_ratio=0.0) for classification.")

        hidden_size = self.backbone.config.hidden_size  # 768 for vit-mae-base

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),  # Better accuracy than LayerNorm; safe with drop_last=True
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # ViTMAEModel returns (last_hidden_state, mask, ids_restore)
        # We use the CLS token (index 0) from last_hidden_state
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        logits    = self.classifier(cls_token)           # [B, 5]
        return logits

    def get_param_groups(self, backbone_lr: float = 2e-5, head_lr: float = 1e-3):
        """Returns parameter groups with differential learning rates."""
        return [
            {"params": self.backbone.parameters(),   "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


# ─────────────────────────────────────────────────────────────────
# 6.  CLASS-WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────────────
def compute_class_weights(labels: pd.Series, num_classes: int = 5) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle APTOS class imbalance
    (SRS §5.2.2 – weighted cross-entropy loss).
    """
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalize
    print("[Weights] Class distribution:", dict(zip(range(num_classes), counts.astype(int))))
    print("[Weights] Class weights     :", np.round(weights, 4).tolist())
    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(labels: pd.Series, num_classes: int = 5) -> WeightedRandomSampler:
    """
    WeightedRandomSampler: assigns each training sample a weight
    inversely proportional to its class frequency, then samples
    WITH replacement so every batch sees all 5 DR grades equally.
    This directly combats the 49% Grade-0 dominance in APTOS 2019.
    """
    counts       = np.bincount(labels, minlength=num_classes).astype(float)
    class_w      = 1.0 / (counts + 1e-6)
    sample_w     = np.array([class_w[lbl] for lbl in labels], dtype=np.float32)
    sampler      = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_w),
        num_samples = len(sample_w),
        replacement = True,
    )
    print("[Sampler] WeightedRandomSampler created — minority classes oversampled.")
    return sampler


# ─────────────────────────────────────────────────────────────────
# 7.  COSINE LR SCHEDULER WITH LINEAR WARMUP
# ─────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine decay (SRS §5.2.3).
    Works by scaling the optimizer LR each epoch.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lrs       = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale    = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale


# ─────────────────────────────────────────────────────────────────
# 8.  TRAIN / VALIDATE EPOCH
# ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch: int, total_epochs: int):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []

    pbar = tqdm(
        loader,
        desc=f"  🔵 Train Epoch {epoch}/{total_epochs}",
        ncols=100,
        leave=True,
        colour="blue",
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # AMP: mixed precision forward pass
        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    kappa    = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    acc      = accuracy_score(labels_all, preds_all)
    return avg_loss, kappa, acc


@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch: int = 0, total_epochs: int = 0, label: str = "Val"):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    colour = "green" if label == "Val" else "yellow"
    desc   = f"  🟢 {label}   Epoch {epoch}/{total_epochs}" if epoch else f"  🟡 {label}"

    pbar = tqdm(
        loader,
        desc=desc,
        ncols=100,
        leave=True,
        colour=colour,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    kappa    = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    acc      = accuracy_score(labels_all, preds_all)
    return avg_loss, kappa, acc, labels_all, preds_all


# ─────────────────────────────────────────────────────────────────
# 9.  PLOTTING UTILITY  (saves after every epoch)
# ─────────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d2e",
    "axes.edgecolor":    "#3a3f5c",
    "axes.grid":         True,
    "grid.color":        "#2a2d40",
    "grid.linestyle":    "--",
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#aaaaaa",
    "ytick.color":       "#aaaaaa",
    "legend.facecolor":  "#1a1d2e",
    "legend.edgecolor":  "#3a3f5c",
    "lines.linewidth":   2.2,
    "lines.markersize":  5,
}

COLOR_TRAIN = "#4fc3f7"   # sky blue
COLOR_VAL   = "#f06292"   # pink
COLOR_TARGET= "#ff7043"   # orange-red


def save_training_curves(history: dict, save_dir: str, current_epoch: int = None):
    """
    Saves a 4-panel training dashboard after every epoch.
    Panels: Loss | Cohen's Kappa | Accuracy | Learning Rate
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(history["train_loss"])
    if n == 0:
        return
    epochs = list(range(1, n + 1))

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(22, 10))
        fig.suptitle(
            f"RetinAgent  ▸  ViT-MAE Fine-tuning  "
            f"({'Epoch ' + str(current_epoch) if current_epoch else 'Final'})",
            fontsize=15, fontweight="bold", color="#ffffff", y=0.98,
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

        # ── Panel 1: Loss ─────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, history["train_loss"], color=COLOR_TRAIN,
                 marker="o", label="Train Loss")
        ax1.plot(epochs, history["val_loss"],   color=COLOR_VAL,
                 marker="s", linestyle="--", label="Val Loss")
        ax1.set_title("📉  Loss", fontsize=12, pad=10)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
        ax1.legend()

        # ── Panel 2: Cohen's Kappa ────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history["train_kappa"], color=COLOR_TRAIN,
                 marker="o", label="Train Kappa")
        ax2.plot(epochs, history["val_kappa"],   color=COLOR_VAL,
                 marker="s", linestyle="--", label="Val Kappa")
        ax2.axhline(0.85, color=COLOR_TARGET, linestyle=":",
                    linewidth=1.8, label="Target  κ = 0.85")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("📊  Cohen's Kappa (Quadratic)", fontsize=12, pad=10)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Kappa")
        ax2.legend()

        # Annotate best val kappa
        best_kappa = max(history["val_kappa"])
        best_ep    = history["val_kappa"].index(best_kappa) + 1
        ax2.annotate(
            f" Best: {best_kappa:.4f}",
            xy=(best_ep, best_kappa),
            xytext=(best_ep + 0.5, best_kappa - 0.06),
            arrowprops=dict(arrowstyle="->", color=COLOR_TARGET),
            color=COLOR_TARGET, fontsize=9,
        )

        # ── Panel 3: Accuracy ─────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, history["train_acc"], color=COLOR_TRAIN,
                 marker="o", label="Train Acc")
        ax3.plot(epochs, history["val_acc"],   color=COLOR_VAL,
                 marker="s", linestyle="--", label="Val Acc")
        ax3.axhline(0.80, color=COLOR_TARGET, linestyle=":",
                    linewidth=1.8, label="Target  80%")
        ax3.set_ylim(0, 1.05)
        ax3.set_title("🎯  Accuracy", fontsize=12, pad=10)
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
        ax3.legend()

        # ── Panel 4: Learning Rate ────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, history["lr"], color="#aed581", marker="D",
                 markersize=4, label="Head LR")
        ax4.set_title("📈  Learning Rate (Head)", fontsize=12, pad=10)
        ax4.set_xlabel("Epoch"); ax4.set_ylabel("LR")
        ax4.set_yscale("log")
        ax4.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(save_dir, "training_curves.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  📊 Graph updated → {out_path}")


# ─────────────────────────────────────────────────────────────────
# 10.  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────
def main(args):
    set_seed(SEED)

    print("=" * 60)
    print("  RetinAgent – ViT-MAE Fine-tuning")
    print(f"  Device : {device}")
    print(f"  Data   : {args.data_dir}")
    print(f"  Epochs : {args.epochs} | Batch: {args.batch_size}")
    print("=" * 60)

    # ── 10.1  Load CSV ──────────────────────────────────────────
    csv_path = os.path.join(args.data_dir, "train.csv")
    df = pd.read_csv(csv_path)
    print(f"[Data] Total samples: {len(df)}")
    print(f"[Data] Label distribution:\n{df['diagnosis'].value_counts().sort_index()}")

    # ── 10.2  Train / Val / Test split (80/10/10) ───────────────
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["diagnosis"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=SEED
    )
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── 10.3  Datasets & Loaders ────────────────────────────────
    # Auto-detect image directory (handles both original and sovitrath Kaggle structure)
    for candidate in ("train_images", "colored_images", "."):
        _cand = os.path.join(args.data_dir, candidate)
        if os.path.isdir(_cand) and any(
            f.endswith(".png") for _, _, files in os.walk(_cand) for f in files
        ):
            img_dir = _cand
            break
    else:
        img_dir = args.data_dir
    print(f"[Data] Image directory: {img_dir}")

    train_ds = APTOSDataset(train_df, img_dir, transform=TRAIN_TRANSFORMS)
    val_ds   = APTOSDataset(val_df,   img_dir, transform=VAL_TRANSFORMS)
    test_ds  = APTOSDataset(test_df,  img_dir, transform=VAL_TRANSFORMS)

    num_workers = min(4, os.cpu_count() or 1)

    # WeightedRandomSampler: oversamples Grade 3 & 4 (rare classes)
    # shuffle=True is INCOMPATIBLE with sampler — sampler replaces it
    # drop_last=True: prevents last batch from being size=1 (BatchNorm1d requires >1 sample)
    # With batch_size=128 and 2929 samples, last batch has 113 samples — safe
    train_sampler = build_sampler(train_df["diagnosis"])
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=train_sampler,
                               num_workers=num_workers, pin_memory=True,
                               drop_last=True)
    val_loader    = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader   = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    # ── 10.4  Model ─────────────────────────────────────────────
    model = DRClassifier(num_classes=5, dropout=0.3).to(device)

    # ── 10.5  Loss ──────────────────────────────────────────────
    # FocalLoss focuses training on hard/rare examples (Grade 3 & 4).
    # gamma=2 is standard; label_smoothing prevents overconfidence.
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

    # AMP GradScaler
    scaler = GradScaler()

    # ── 10.6  Optimizer (differential LR) ───────────────────────
    param_groups = model.get_param_groups(
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
    )
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # ── 10.7  Scheduler (warmup + cosine decay) ──────────────────
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=args.epochs,
    )

    # ── 10.8  Training loop ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    best_kappa      = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    history = {
        "train_loss": [], "val_loss": [],
        "train_kappa": [], "val_kappa": [],
        "train_acc": [],   "val_acc": [],
        "lr": [],
    }

    print("\n[Training started]\n")
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch - 1)          # update LR before epoch
        current_lr = optimizer.param_groups[1]["lr"]   # head LR

        epoch_start = time.time()

        # ── Print epoch header ──────────────────────────────────
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Epoch {epoch:>3}/{args.epochs}   │   Head LR: {current_lr:.2e}")
        print(sep)

        # ── Train ───────────────────────────────────────────────
        train_loss, train_kappa, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            epoch=epoch, total_epochs=args.epochs,
        )

        # ── Validate ────────────────────────────────────────────
        val_loss, val_kappa, val_acc, val_labels, val_preds = val_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, total_epochs=args.epochs, label="Val",
        )

        epoch_time = time.time() - epoch_start

        # ── Log to history ──────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_kappa"].append(train_kappa)
        history["val_kappa"].append(val_kappa)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # ── Pretty summary table ────────────────────────────────
        best_marker = " 🏆" if val_kappa >= best_kappa else ""
        target_kappa_hit = "✅" if val_kappa >= 0.85 else "  "
        target_acc_hit   = "✅" if val_acc  >= 0.80 else "  "

        print(f"\n  {'Metric':<22} {'Train':>10}  {'Val':>10}")
        print(f"  {'─'*44}")
        print(f"  {'Loss':<22} {train_loss:>10.4f}  {val_loss:>10.4f}")
        print(f"  {'Kappa (Quadratic)':<22} {train_kappa:>10.4f}  {val_kappa:>10.4f}  {target_kappa_hit}{best_marker}")
        print(f"  {'Accuracy':<22} {train_acc*100:>9.2f}%  {val_acc*100:>9.2f}%  {target_acc_hit}")
        print(f"  {'Time':<22} {epoch_time:>10.1f}s")

        # ── Save best model ─────────────────────────────────────
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                # Save as Python floats to avoid numpy scalar unpickling issues
                "val_kappa":   float(val_kappa),
                "val_acc":     float(val_acc),
                "val_loss":    float(val_loss),
            }, best_model_path)
            print(f"\n  ✅ Best model saved  →  {best_model_path}")

        # ── Save graph after every epoch ────────────────────────
        save_training_curves(history, args.output_dir, current_epoch=epoch)

        # ── Save CSV log after every epoch ──────────────────────
        hist_df = pd.DataFrame(history)
        hist_path = os.path.join(args.output_dir, "training_history.csv")
        hist_df.to_csv(hist_path, index=False)

    total_time = (time.time() - total_start) / 60

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅  TRAINING COMPLETE")
    print(f"  Total Time  : {total_time:.1f} minutes")
    print(f"  Best Kappa  : {best_kappa:.4f}  (Target: >0.85)")
    print("═" * 60)

    # ── Final graph (already saved per-epoch, save once more) ────
    save_training_curves(history, args.output_dir, current_epoch=None)

    # ── 10.10  Final evaluation on test set ──────────────────────
    print("\n📋  TEST SET EVALUATION")
    print("─" * 60)
    # weights_only=False needed for PyTorch 2.6+ compatibility
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_kappa, test_acc, test_labels, test_preds = val_epoch(
        model, test_loader, criterion, device, label="Test",
    )

    print(f"\n  Test Loss  : {test_loss:.4f}")
    print(f"  Test Kappa : {test_kappa:.4f}  {'✅' if test_kappa>=0.85 else '❌'}  (Target: >0.85)")
    print(f"  Test Acc   : {test_acc*100:.2f}%  {'✅' if test_acc>=0.80 else '❌'}  (Target: >80%)")
    print()

    grade_names = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)",
                   "Grade 3 (Severe)", "Grade 4 (Proliferative)"]
    print("\n📊  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=grade_names))

    # ── 10.11  Save final model weights ─────────────────────────
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n[Saved] Final model weights   → {final_path}")
    print(f"[Saved] Best checkpoint       → {best_model_path}")
    print(f"[Saved] Training history CSV  → {hist_path}")
    print(f"[Saved] Training graphs       → {os.path.join(args.output_dir, 'training_curves.png')}")


# ─────────────────────────────────────────────────────────────────
# 11.  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetinAgent ViT-MAE Fine-tuning")

    parser.add_argument("--data_dir",    type=str,   default="./aptos2019-blindness-detection",
                        help="Root directory containing train.csv and train_images/")
    parser.add_argument("--output_dir",  type=str,   default="./checkpoints",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--epochs",      type=int,   default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size",  type=int,   default=128,
                        help="Batch size (default: 128 for MI300X, reduce if OOM)")
    parser.add_argument("--backbone_lr", type=float, default=2e-5,
                        help="Learning rate for ViT-MAE backbone (default: 2e-5)")
    parser.add_argument("--head_lr",     type=float, default=1e-3,
                        help="Learning rate for classification head (default: 1e-3)")

    args = parser.parse_args()
    main(args)
