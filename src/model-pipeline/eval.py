"""
evaluate.py ===========
Evaluasi model GRU yang sudah dilatih.

Jalankan:
    python evaluate.py --data data/ --checkpoint checkpoints/best_model.pt --config checkpoints/config.json

Output:
    - Classification report (per kelas)
    - Confusion matrix (gambar + .npy)
    - Learning curve
    - Top-K accuracy
    - Inference speed benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from dataset import SignLandmarkDataset, scan_dataset, build_dataloaders
from train import SignGRU, run_epoch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load model dari checkpoint
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
) -> tuple[SignGRU, LabelEncoder, int]:
    """
    Load model dari checkpoint + config.json.
    Kembalikan (model, label_encoder, max_len).
    """
    config = json.loads(Path(config_path).read_text())

    model = SignGRU(
        input_dim     = config["input_dim"],
        num_classes   = config["num_classes"],
        hidden_dim    = config["hidden_dim"],
        num_layers    = config["num_layers"],
        dropout       = config["dropout"],
        bidirectional = config["bidirectional"],
        proj_dim      = config["proj_dim"],
        use_attention = config.get("use_attention", False),
    ).to(device)

    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    le = LabelEncoder()
    le.classes_ = np.array(config["label_classes"])

    return model, le, config["max_len"]


# ---------------------------------------------------------------------------
# Prediksi seluruh loader
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kembalikan (y_true, y_pred, y_prob) untuk seluruh loader.
    y_prob: [N, num_classes] softmax probabilities.
    """
    all_true, all_pred, all_prob = [], [], []

    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        logits = model(x, lengths)
        probs  = torch.softmax(logits, dim=1)

        all_true.append(y.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


# ---------------------------------------------------------------------------
# Plot utilities (matplotlib optional)
# ---------------------------------------------------------------------------

def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    plt = _try_import_mpl()
    if plt is None:
        logger.warning("matplotlib tidak tersedia, skip confusion matrix plot")
        return

    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    n = len(class_names)
    figsize = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm_plot[i, j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                    fontsize=7)

    fig.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info("Confusion matrix disimpan: %s", output_path)


def plot_learning_curve(
    history: dict,
    output_path: str | Path,
) -> None:
    plt = _try_import_mpl()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(
        [a * 100 for a in history["train_acc"]], label="Train Acc"
    )
    axes[1].plot(
        [a * 100 for a in history["val_acc"]], label="Val Acc"
    )
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info("Learning curve disimpan: %s", output_path)


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    plt = _try_import_mpl()
    if plt is None:
        return

    per_class_acc = []
    for cls_idx in range(len(class_names)):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append((y_pred[mask] == cls_idx).mean())

    # Urutkan dari terendah ke tertinggi
    sorted_idx = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(6, len(class_names) * 0.35)))
    colors = ["#d32f2f" if a < 0.7 else "#f57c00" if a < 0.9 else "#388e3c"
              for a in sorted_acc]
    bars = ax.barh(sorted_names, [a * 100 for a in sorted_acc], color=colors)

    for bar, acc in zip(bars, sorted_acc):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}", va="center", fontsize=8,
        )

    ax.axvline(x=np.mean(per_class_acc) * 100, color="navy", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(per_class_acc):.1%}")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlim(0, 110)
    ax.legend()
    fig.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info("Per-class accuracy disimpan: %s", output_path)


# ---------------------------------------------------------------------------
# Inference speed benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    input_dim: int,
    max_len: int,
    device: torch.device,
    n_runs: int = 200,
    batch_size: int = 1,
) -> dict:
    """Ukur rata-rata inference time per sample (ms)."""
    model.eval()
    dummy_x = torch.randn(batch_size, max_len, input_dim).to(device)
    dummy_l = torch.full((batch_size,), max_len, dtype=torch.long).to(device)

    # Warmup
    for _ in range(10):
        model(dummy_x, dummy_l)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy_x, dummy_l)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_samples = n_runs * batch_size
    ms_per_sample = (elapsed / total_samples) * 1000
    fps           = total_samples / elapsed

    return {
        "ms_per_sample": round(ms_per_sample, 3),
        "fps":           round(fps, 1),
        "device":        str(device),
    }


# ---------------------------------------------------------------------------
# Evaluasi utama
# ---------------------------------------------------------------------------

def evaluate(
    data_dir: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    history_path: str | None = None,
    batch_size: int = 32,
    device_str: str = "auto",
    top_k: int = 3,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    # Load model
    model, le, max_len = load_model(checkpoint_path, config_path, device)
    class_names = list(le.classes_)
    config = json.loads(Path(config_path).read_text())

    # Build val loader (pakai split 80:20 yang sama dengan training)
    _, val_loader, _, _, _ = build_dataloaders(
        data_dir,
        max_len=max_len,
        batch_size=batch_size,
    )

    # Prediksi
    logger.info("Running inference pada validation set...")
    y_true, y_pred, y_prob = predict_all(model, val_loader, device)

    # ---- Metrics ----
    overall_acc = (y_true == y_pred).mean()

    # Top-K
    topk_acc = top_k_accuracy_score(y_true, y_prob, k=min(top_k, len(class_names)))

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ---- Print ----
    print("=" * 60)
    print(f"  Overall Accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print(f"  Top-{top_k} Accuracy  : {topk_acc:.4f}  ({topk_acc*100:.2f}%)")
    print("=" * 60)
    print(report)

    # ---- Kelas dengan performa terburuk ----
    per_class_acc = []
    for i in range(len(class_names)):
        mask = y_true == i
        acc  = (y_pred[mask] == i).mean() if mask.sum() > 0 else 0.0
        per_class_acc.append(acc)

    worst_5 = sorted(range(len(class_names)), key=lambda i: per_class_acc[i])[:5]
    print("\nKelas dengan akurasi terendah:")
    for i in worst_5:
        print(f"  {class_names[i]:20s} : {per_class_acc[i]:.4f}")

    # ---- Benchmark ----
    bench = benchmark_inference(model, config["input_dim"], max_len, device)
    print(f"\nInference speed  : {bench['ms_per_sample']} ms/sample ({bench['fps']} fps)")

    # ---- Simpan ----
    np.save(str(output_dir / "confusion_matrix.npy"), cm)

    plot_confusion_matrix(cm, class_names,
                          output_dir / "confusion_matrix.png", normalize=True)
    plot_per_class_accuracy(y_true, y_pred, class_names,
                            output_dir / "per_class_accuracy.png")

    if history_path and Path(history_path).exists():
        history = np.load(history_path, allow_pickle=True).item()
        plot_learning_curve(history, output_dir / "learning_curve.png")

    # Simpan ringkasan JSON
    summary = {
        "overall_acc":    round(float(overall_acc),  4),
        f"top{top_k}_acc": round(float(topk_acc),    4),
        "inference":      bench,
        "per_class_acc":  {class_names[i]: round(float(per_class_acc[i]), 4)
                           for i in range(len(class_names))},
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Hasil evaluasi disimpan di: %s", output_dir)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluasi Sign Language GRU")
    p.add_argument("--data",       required=True, help="Root folder dataset")
    p.add_argument("--checkpoint", required=True, help="Path best_model.pt")
    p.add_argument("--config",     required=True, help="Path config.json")
    p.add_argument("--output",     default="eval_output", help="Output dir")
    p.add_argument("--history",    default=None,  help="Path history.npy (untuk learning curve)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k",      type=int, default=3)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    evaluate(
        data_dir        = args.data,
        checkpoint_path = args.checkpoint,
        config_path     = args.config,
        output_dir      = args.output,
        history_path    = args.history,
        batch_size      = args.batch_size,
        top_k           = args.top_k,
    )
