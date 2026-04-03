"""
train.py
========
Definisi model GRU + training loop + hyperparameter tuning (Optuna).

Jalankan training standar:
    python train.py --data data/ --epochs 50 --output checkpoints/

Jalankan hyperparameter search (Optuna):
    python train.py --data data/ --tune --n-trials 30 --output checkpoints/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import build_dataloaders

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SignGRU(nn.Module):
    """
    GRU-based sequence classifier untuk sign language recognition.

    Arsitektur:
        Input [B, T, D]
          → Dropout input
          → Linear projection (D → proj_dim)  [opsional, jika proj_dim > 0]
          → GRU (num_layers, bidirectional)
          → Ambil hidden state terakhir
          → Dropout
          → LayerNorm
          → FC head (hidden_dim → num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        proj_dim: int = 128,          # 0 = tidak pakai projection
        use_attention: bool = False,  # simple self-attention atas GRU output
    ) -> None:
        super().__init__()

        self.bidirectional  = bidirectional
        self.num_layers     = num_layers
        self.hidden_dim     = hidden_dim
        self.use_attention  = use_attention
        self.num_directions = 2 if bidirectional else 1

        # Input dropout
        self.input_dropout = nn.Dropout(p=dropout * 0.5)

        # Opsional: projection layer untuk kompres fitur mentah
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout * 0.5),
            )
            gru_input_dim = proj_dim
        else:
            self.proj = None
            gru_input_dim = input_dim

        # GRU core
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * self.num_directions

        # Opsional: attention pooling
        if use_attention:
            self.attn_fc = nn.Linear(gru_out_dim, 1)

        # Classifier head
        self.dropout  = nn.Dropout(p=dropout)
        self.norm     = nn.LayerNorm(gru_out_dim)
        self.fc       = nn.Linear(gru_out_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "fc.weight" in name:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, D]
        lengths: torch.Tensor,                   # [B]  actual lengths
    ) -> torch.Tensor:                           # [B, num_classes]

        x = self.input_dropout(x)

        if self.proj is not None:
            x = self.proj(x)

        # Pack untuk efisiensi (ignore padding di GRU)
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        gru_out_packed, hidden = self.gru(packed)

        if self.use_attention:
            # Unpack untuk attention
            from torch.nn.utils.rnn import pad_packed_sequence
            gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
            # Attention score [B, T, 1]
            scores = self.attn_fc(gru_out)
            # Mask padding positions
            B, T, _ = gru_out.shape
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            scores = scores.squeeze(-1)              # [B, T]
            scores = scores.masked_fill(~mask, -1e9)
            attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
            context = (gru_out * attn_weights).sum(dim=1)              # [B, H]
            out = context
        else:
            # Ambil hidden state terakhir dari layer teratas
            # hidden: [num_layers * num_directions, B, H]
            if self.bidirectional:
                # Gabungkan forward & backward dari layer terakhir
                h_fwd = hidden[-2]  # [B, H]
                h_bwd = hidden[-1]  # [B, H]
                out = torch.cat([h_fwd, h_bwd], dim=1)   # [B, 2H]
            else:
                out = hidden[-1]   # [B, H]

        out = self.dropout(out)
        out = self.norm(out)
        out = self.fc(out)         # [B, num_classes]
        return out


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Kembalikan True jika harus stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state:
            model.load_state_dict(self.best_state)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float]:
    """Satu epoch train atau eval. Kembalikan (avg_loss, accuracy)."""
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            logits = model(x, lengths)
            loss   = criterion(logits, y)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training loop utama
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str,
    # Hyperparameter
    max_len: int        = 150,
    batch_size: int     = 32,
    hidden_dim: int     = 256,
    num_layers: int     = 2,
    dropout: float      = 0.3,
    bidirectional: bool = True,
    proj_dim: int       = 128,
    use_attention: bool = False,
    lr: float           = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int         = 60,
    patience: int       = 12,
    label_smoothing: float = 0.1,
    num_workers: int    = 0,
    # Lainnya
    device_str: str     = "auto",
    normalize: bool     = True,
) -> dict:
    """
    Training lengkap. Kembalikan dict metrics terbaik.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    logger.info("Device: %s", device)

    # --- Data ---
    train_loader, val_loader, le, num_classes, input_dim = build_dataloaders(
        data_dir,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
    )

    # Simpan label encoder
    np.save(str(output_dir / "label_classes.npy"), le.classes_)

    # --- Model ---
    model = SignGRU(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        proj_dim=proj_dim,
        use_attention=use_attention,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %s", f"{n_params:,}")

    # --- Loss, optimizer, scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    stopper   = EarlyStopping(patience=patience)

    # --- Loop ---
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        logger.info(
            "Epoch %3d | tr_loss=%.4f tr_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.2e",
            epoch, tr_loss, tr_acc, vl_loss, vl_acc,
            optimizer.param_groups[0]["lr"],
        )

        # Simpan best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info("  → Best model saved (val_acc=%.4f)", best_val_acc)

        if stopper.step(vl_loss, model):
            logger.info("Early stopping pada epoch %d", epoch)
            break

    stopper.restore_best(model)

    # Simpan config + history
    config = dict(
        input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
        proj_dim=proj_dim, use_attention=use_attention, max_len=max_len,
        label_classes=list(le.classes_),
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    np.save(str(output_dir / "history.npy"), history)

    logger.info("Training selesai. Best val_acc: %.4f", best_val_acc)
    return {"best_val_acc": best_val_acc, "history": history}


# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna)
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    data_dir: str,
    output_dir: str,
    n_trials: int = 30,
    epochs_per_trial: int = 25,
) -> dict:
    """
    Cari hyperparameter terbaik pakai Optuna.
    Instal dulu: pip install optuna
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna dulu: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        hp = dict(
            hidden_dim    = trial.suggest_categorical("hidden_dim",    [128, 256, 512]),
            num_layers    = trial.suggest_int("num_layers",            1, 3),
            dropout       = trial.suggest_float("dropout",             0.1, 0.5, step=0.1),
            bidirectional = trial.suggest_categorical("bidirectional", [True, False]),
            proj_dim      = trial.suggest_categorical("proj_dim",      [0, 64, 128, 256]),
            use_attention = trial.suggest_categorical("use_attention", [True, False]),
            lr            = trial.suggest_float("lr",                  1e-4, 5e-3, log=True),
            weight_decay  = trial.suggest_float("weight_decay",        1e-5, 1e-3, log=True),
            batch_size    = trial.suggest_categorical("batch_size",    [16, 32, 64]),
            label_smoothing = trial.suggest_float("label_smoothing",   0.0, 0.2, step=0.05),
            max_len       = trial.suggest_categorical("max_len",       [100, 150, 200]),
        )
        trial_out = Path(output_dir) / f"trial_{trial.number}"
        result = train(
            data_dir=data_dir,
            output_dir=str(trial_out),
            epochs=epochs_per_trial,
            patience=6,
            **hp,
        )
        return result["best_val_acc"]

    study = optuna.create_study(
        direction="maximize",
        study_name="sign_gru_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info("Best trial val_acc: %.4f", study.best_value)
    logger.info("Best params: %s", best)

    # Simpan hasil tuning
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "best_hyperparams.json").write_text(
        json.dumps({"val_acc": study.best_value, "params": best}, indent=2)
    )

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Sign Language GRU")
    p.add_argument("--data",    required=True, help="Root folder dataset")
    p.add_argument("--output",  default="checkpoints", help="Output dir")
    p.add_argument("--epochs",  type=int,   default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden",  type=int,   default=256)
    p.add_argument("--layers",  type=int,   default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-len", type=int,   default=150)
    p.add_argument("--no-bidir",   action="store_true", help="Unidirectional GRU")
    p.add_argument("--attention",  action="store_true", help="Pakai attention pooling")
    p.add_argument("--tune",       action="store_true", help="Jalankan Optuna tuning")
    p.add_argument("--n-trials",   type=int, default=30, help="Jumlah trial Optuna")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_parser().parse_args()

    if args.tune:
        best_hp = tune_hyperparameters(args.data, args.output, n_trials=args.n_trials)
        logger.info("Re-training dengan best hyperparams selama %d epoch...", args.epochs)
        train(data_dir=args.data, output_dir=args.output, epochs=args.epochs, **best_hp)
    else:
        train(
            data_dir=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            max_len=args.max_len,
            bidirectional=not args.no_bidir,
            use_attention=args.attention,
        )
