"""
dataset.py
==========
Preprocessing & DataLoader untuk sign language landmark data.

Struktur folder yang diharapkan:
    data/
        aku/
            1.npz, 2.npz, 3.npz, ...
        kamu/
            1.npz, 2.npz, ...
        ...

Tiap .npz mengandung key: 'pose' [T,9,4], 'face' [T,46,3], 'hands' [T,2,21,3]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi fitur
# ---------------------------------------------------------------------------

# Index visibility di pose (kolom ke-3, 0-indexed)
POSE_VIS_COL = 3

def extract_features(npz_path: str | Path) -> np.ndarray:
    """
    Load satu .npz dan gabungkan semua landmark menjadi flat feature vector
    per frame: [T, D]

    Komposisi fitur per frame:
      - pose  : 9 titik × 3 (x,y,z) = 27  (visibility dibuang)
      - face  : 46 titik × 3         = 138
      - hands : 2 × 21 × 3           = 126
      Total D = 291
    """
    data = np.load(str(npz_path))

    pose  = data["pose"]   # [T, 9,  4]
    face  = data["face"]   # [T, 46, 3]
    hands = data["hands"]  # [T, 2, 21, 3]

    T = pose.shape[0]

    # Buang kolom visibility dari pose, ambil x,y,z saja
    pose_xyz = pose[:, :, :3]                     # [T, 9,  3]
    hands_flat = hands.reshape(T, -1)             # [T, 126]

    # Flatten semua
    feat = np.concatenate([
        pose_xyz.reshape(T, -1),                  # [T, 27]
        face.reshape(T, -1),                      # [T, 138]
        hands_flat,                               # [T, 126]
    ], axis=1)                                    # [T, 291]

    return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# Padding / truncating ke fixed length
# ---------------------------------------------------------------------------

def pad_or_truncate(seq: np.ndarray, max_len: int) -> tuple[np.ndarray, int]:
    """
    Normalkan panjang sequence ke max_len.
    Mengembalikan (padded_seq [max_len, D], original_len).
    Padding dengan 0 di akhir (post-padding).
    """
    T, D = seq.shape
    actual_len = min(T, max_len)

    out = np.zeros((max_len, D), dtype=np.float32)
    out[:actual_len] = seq[:actual_len]

    return out, actual_len


# ---------------------------------------------------------------------------
# NaN imputation
# ---------------------------------------------------------------------------

def impute_nan(seq: np.ndarray) -> np.ndarray:
    """
    Isi NaN dengan interpolasi linear antar frame.
    Jika seluruh kolom NaN (landmark tidak pernah terdeteksi), isi dengan 0.
    """
    seq = seq.copy()
    for col in range(seq.shape[1]):
        col_data = seq[:, col]
        nan_mask = np.isnan(col_data)
        if nan_mask.all():
            seq[:, col] = 0.0
            continue
        if nan_mask.any():
            x = np.arange(len(col_data))
            seq[:, col] = np.interp(x, x[~nan_mask], col_data[~nan_mask])
    return seq


# ---------------------------------------------------------------------------
# Normalisasi per-sequence (z-score)
# ---------------------------------------------------------------------------

def normalize_sequence(seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization per sequence (bukan global)."""
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0, keepdims=True)
    return (seq - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------

def scan_dataset(root_dir: str | Path) -> tuple[list[Path], list[str]]:
    """
    Scan folder root_dir, kembalikan (list_npz_paths, list_labels).
    Label diambil dari nama subfolder.
    """
    root = Path(root_dir)
    paths, labels = [], []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        npz_files = sorted(class_dir.glob("*.npz"))
        if not npz_files:
            logger.warning("Kelas '%s' tidak punya file .npz", class_dir.name)
            continue
        for npz in npz_files:
            paths.append(npz)
            labels.append(class_dir.name)

    logger.info("Ditemukan %d sample dari %d kelas", len(paths), len(set(labels)))
    return paths, labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SignLandmarkDataset(Dataset):
    """
    Dataset untuk sign language recognition dari file .npz.

    Parameters
    ----------
    paths      : List path file .npz
    labels     : List label string (nama gestur)
    le         : LabelEncoder yang sudah di-fit
    max_len    : Panjang sequence yang dinormalisasi
    normalize  : Apakah pakai z-score normalization
    """

    def __init__(
        self,
        paths: list[Path],
        labels: list[str],
        le: LabelEncoder,
        max_len: int = 150,
        normalize: bool = True,
    ) -> None:
        self.paths     = paths
        self.labels    = labels
        self.le        = le
        self.max_len   = max_len
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path  = self.paths[idx]
        label = self.labels[idx]

        # 1. Ekstrak fitur [T, D]
        feat = extract_features(path)

        # 2. Imputasi NaN
        feat = impute_nan(feat)

        # 3. Normalisasi
        if self.normalize:
            feat = normalize_sequence(feat)

        # 4. Pad / truncate ke max_len
        feat_padded, actual_len = pad_or_truncate(feat, self.max_len)

        # 5. Encode label
        label_enc = self.le.transform([label])[0]

        return (
            torch.from_numpy(feat_padded),           # [max_len, D]
            torch.tensor(label_enc, dtype=torch.long),
            torch.tensor(actual_len, dtype=torch.long),  # untuk packed sequence
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    root_dir: str | Path,
    max_len: int = 150,
    test_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    random_state: int = 42,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, LabelEncoder, int, int]:
    """
    Scan folder, split 80:20, kembalikan train/val DataLoader.

    Returns
    -------
    train_loader, val_loader, label_encoder, num_classes, input_dim
    """
    paths, labels = scan_dataset(root_dir)

    # Fit label encoder
    le = LabelEncoder()
    le.fit(labels)
    num_classes = len(le.classes_)
    logger.info("Kelas (%d): %s", num_classes, list(le.classes_))

    # Split stratified
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    logger.info("Train: %d | Val: %d", len(train_paths), len(val_paths))

    train_ds = SignLandmarkDataset(train_paths, train_labels, le, max_len, normalize)
    val_ds   = SignLandmarkDataset(val_paths,   val_labels,   le, max_len, normalize)

    # Input dim: ambil dari sample pertama
    sample_feat, _, _ = train_ds[0]
    input_dim = sample_feat.shape[1]  # D = 291
    logger.info("Input dim: %d", input_dim)

    # Weighted sampler untuk handle class imbalance
    train_sampler = None
    if use_weighted_sampler:
        label_counts = np.bincount([le.transform([l])[0] for l in train_labels])
        weights_per_class = 1.0 / (label_counts + 1e-8)
        sample_weights = [weights_per_class[le.transform([l])[0]] for l in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, le, num_classes, input_dim


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    train_loader, val_loader, le, n_cls, in_dim = build_dataloaders(root)
    x, y, lengths = next(iter(train_loader))
    print(f"Batch x     : {x.shape}")       # [B, max_len, 291]
    print(f"Batch y     : {y.shape}")       # [B]
    print(f"Batch lengths: {lengths}")
    print(f"Num classes : {n_cls}")
    print(f"Input dim   : {in_dim}")
