"""Train a model on the benchmark dataset splits."""

import argparse
import json
import logging
import sys
from logging import Logger
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.cnn import CNN
from models.inceptiontime import InceptionTime
from models.minirocket import MiniRocket
from models.moment import Moment
from models.resnet import ResNet
from models.tcn import TCN

MODELS = {
    "cnn": CNN,
    "tcn": TCN,
    "inceptiontime": InceptionTime,
    "resnet": ResNet,
    "moment": Moment,
    "minirocket": MiniRocket,
}


def _get_logger(filename: str | None = None) -> Logger:
    logger = logging.getLogger(__name__)

    format = "%(asctime)s, %(levelname)s: %(message)s"
    formatter = logging.Formatter(format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().setLevel(logging.INFO)

    if filename is not None:
        logfile = filename
        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logging.getLogger().addHandler(file_handler)

    return logger


def _get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def load_split(data_dir: Path, name: str):
    """Load a saved split from disk, returning (data, labels) as float32 arrays."""
    data = np.load(data_dir / f"{name}_data.npy")

    with open(data_dir / f"{name}_labels.json") as file:
        labels = json.load(file)
        file.close()
    labels = np.array(list(labels.values()), dtype=np.float32)

    with open(data_dir / f"{name}_metadata.json") as file:
        metadata = json.load(file)
        file.close()

    return data.astype(np.float32), labels, metadata


def make_loader(data: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    """Wrap numpy arrays in a DataLoader. Adds channel dim: (N, T) → (N, 1, T)."""
    x = torch.from_numpy(data).unsqueeze(1)
    y = torch.from_numpy(labels)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, criterion, device):
    """Compute loss, exact-match accuracy, and macro F1 on a data loader."""
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            total_loss += criterion(logits, y).item() * len(x)

            all_preds.append((logits.sigmoid() >= 0.5).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets).astype(int)

    loss = total_loss / len(loader.dataset)
    exact_match = (preds == targets).all(axis=1).mean()

    f1_per_label = f1_score(targets, preds, average=None)
    f1_micro = f1_score(targets, preds, average="micro")
    f1_macro = f1_score(targets, preds, average="macro")
    acc_hamming = float(np.mean(targets == preds))

    return {"loss": loss,
            "exact_match": exact_match,
            "hamming_acc": acc_hamming,
            "f1_per_label": list(f1_per_label),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,}


def train_nn(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = _get_logger(filename=f"{args.out_dir}/{args.model}.log")
    device = _get_device()

    logger.info(f"Device: {device}")

    # --- Load data ---
    logger.info(f"Loading data from {data_dir}")
    train_data, train_labels, _ = load_split(data_dir, "train")
    val_data, val_labels, _ = load_split(data_dir, "val")
    test_data, test_labels, _ = load_split(data_dir, "test")

    logger.info(f"Number of samples: train={len(train_labels)}  val={len(val_labels)}  test={len(test_labels)}")

    train_loader = make_loader(train_data, train_labels, args.batch_size, shuffle=True)
    val_loader = make_loader(val_data, val_labels, args.batch_size, shuffle=False)
    test_loader = make_loader(test_data, test_labels, args.batch_size, shuffle=False)

    # --- Load model ---
    num_classes = train_labels.shape[1]

    model = MODELS[args.model](in_channels=1, num_classes=num_classes).to(device)
    logger.info(f"Model: {model.__class__.__name__}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Train model ---
    logger.info("Training")

    # for early stopping
    best_val_loss = float("inf")
    no_improvement_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
        val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        # Early stopping and save best model
        if val_loss < best_val_loss - args.delta:
            best_val_loss = val_loss
            no_improvement_counter = 0
            torch.save(model.state_dict(), out_dir / f"best_{args.model}.pt")
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= args.patience:
                logger.info(f"Early stopping in epoch {epoch}.")
                break

    logger.info(f"Best val loss: {best_val_loss:.4f}  →  {out_dir / f'best_{args.model}.pt'}")

    # --- Model evaluation ---
    logger.info("Evaluation")

    # load best model
    model.load_state_dict(torch.load(out_dir / f"best_{args.model}.pt", map_location=device))
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    json.dump(val_metrics, open(out_dir / f"best_{args.model}_val_metrics.json", "w"))
    json.dump(test_metrics, open(out_dir / f"best_{args.model}_test_metrics.json", "w"))

    logger.info(f"{'':>10}  {'loss':>8}  {'exact_match':>11}  {'hamming_acc':>11} {'f1_macro':>8}")
    logger.info(f"{'val':>10}  {val_metrics['loss']:>8.4f}  {val_metrics['exact_match']:>11.4f}  {val_metrics['hamming_acc']:>11.4f} {val_metrics['f1_macro']:>8.4f}")
    logger.info(f"{'test':>10}  {test_metrics['loss']:>8.4f}  {test_metrics['exact_match']:>11.4f}  {test_metrics['hamming_acc']:>11.4f} {test_metrics['f1_macro']:>8.4f}")


def evaluate_rocket(model, data, targets):
    preds = model.predict(np.expand_dims(data, axis=1))

    exact_match = (preds == targets).all(axis=1).mean()

    f1_per_label = f1_score(targets, preds, average=None)
    f1_micro = f1_score(targets, preds, average="micro")
    f1_macro = f1_score(targets, preds, average="macro")
    acc_hamming = float(np.mean(targets == preds))

    return {"loss": None,
            "exact_match": exact_match,
            "hamming_acc": acc_hamming,
            "f1_per_label": list(f1_per_label),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro, }


def train_rocket(args):
    logger = _get_logger(filename=f"{args.out_dir}/{args.model}.log")
    device = torch.device("cpu")

    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info(f"Loading data from {data_dir}")
    train_data, train_labels, _ = load_split(data_dir, "train")
    val_data, val_labels, _ = load_split(data_dir, "val")
    test_data, test_labels, _ = load_split(data_dir, "test")

    logger.info(f"Number of samples: train={len(train_labels)}  val={len(val_labels)}  test={len(test_labels)}")

    # --- Load model ---
    model = MODELS[args.model]()
    logger.info(f"Model: {model.__class__.__name__}")

    # --- "Training" ---
    logger.info("Fit model")
    model.fit(np.expand_dims(train_data, axis=1), train_labels)

    # save model
    joblib.dump(model, out_dir / f"best_{args.model}.pkl")

    # --- Model evaluation ---
    logger.info("Evaluation")

    val_metrics = evaluate_rocket(model, val_data, val_labels)
    test_metrics = evaluate_rocket(model, test_data, test_labels)

    json.dump(val_metrics, open(out_dir / f"best_{args.model}_val_metrics.json", "w"))
    json.dump(test_metrics, open(out_dir / f"best_{args.model}_test_metrics.json", "w"))

    logger.info(f"{'':>10}  {'exact_match':>11}  {'hamming_acc':>11}  {'f1_macro':>8}")
    logger.info(
        f"{'val':>10}  {val_metrics['exact_match']:>11.4f}  {val_metrics['hamming_acc']:>11.4f}  {val_metrics['f1_macro']:>8.4f}")
    logger.info(
        f"{'test':>10}  {test_metrics['exact_match']:>11.4f}  {test_metrics['hamming_acc']:>11.4f}  {test_metrics['f1_macro']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="cnn", choices=list(MODELS.keys()), help="Model architecture")
    parser.add_argument("--data_dir", type=str, default="./experiments/dataset/", help="Directory containing the dataset splits")
    parser.add_argument("--out_dir", type=str, default="./experiments/models/", help="Directory to save the model checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--delta", type=float, default=0.001, help="Delta for early stopping")

    args = parser.parse_args()

    if "rocket" in args.model:
        train_rocket(args)
    else:
        train_nn(args)
