"""Train the TimeSeriesTransformerForPrediction model on the daily
per-store sales panel, and save the trained checkpoint + metadata.

Run as a script:
    python -m src.train
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import SalesWindowDataset, build_series
from src.model import build_model, count_parameters


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(panel: pd.DataFrame | None = None, verbose: bool = True):
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if panel is None:
        panel = pd.read_parquet(config.DAILY_STORE_SALES_PATH)

    series_list = build_series(panel)
    num_stores = int(panel["store_nbr"].max())

    train_ds = SalesWindowDataset(series_list, mode="train")
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    device = get_device()
    model = build_model(num_stores=num_stores).to(device)
    if verbose:
        print(f"Device: {device}")
        print(f"Trainable parameters: {count_parameters(model):,}")
        print(f"Training windows: {len(train_ds):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    history = {"epoch": [], "train_loss": [], "lr": [], "seconds": []}

    model.train()
    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        epoch_loss = float(np.mean(epoch_losses))
        dt = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["seconds"].append(dt)
        if verbose:
            print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | loss {epoch_loss:.4f} | {dt:.1f}s")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.MODEL_DIR)

    metadata = {
        "num_stores": num_stores,
        "context_length": config.CONTEXT_LENGTH,
        "prediction_length": config.PREDICTION_LENGTH,
        "lags_sequence": config.LAGS_SEQUENCE,
        "num_time_features": config.NUM_TIME_FEATURES,
        "trained_rows": int(len(panel)),
        "last_date": str(panel["date"].max()),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    history_df = pd.DataFrame(history)
    history_df.to_csv(config.TABLES_DIR / "training_history.csv", index=False)

    return model, history_df


if __name__ == "__main__":
    train_model()
