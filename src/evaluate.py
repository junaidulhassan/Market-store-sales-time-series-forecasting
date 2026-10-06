"""Backtest the trained model: for every store, forecast the last
`VALID_DAYS` days (which were held out of training) and score the
forecast against the actual observed sales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src import config
from src.dataset import build_series, make_inference_window
from src.metrics import compute_all_metrics
from src.predict import load_model, summarize_samples


@torch.no_grad()
def backtest(panel: pd.DataFrame, model=None, device=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or load_model(device)

    holdout = config.VALID_DAYS
    rows = []
    forecast_frames = []

    for store_nbr, g in panel.groupby("store_nbr"):
        g = g.sort_values("date")
        train_part = g.iloc[:-holdout]
        actual_part = g.iloc[-holdout : -holdout + config.PREDICTION_LENGTH]
        if len(actual_part) < config.PREDICTION_LENGTH:
            continue

        series = build_series(train_part)[0]
        batch, future_dates = make_inference_window(
            series, context_length=config.CONTEXT_LENGTH, max_lag=max(config.LAGS_SEQUENCE)
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            static_categorical_features=batch["static_categorical_features"],
            future_time_features=batch["future_time_features"],
        )
        samples = np.clip(outputs.sequences.cpu().numpy()[0], 0.0, None)
        summary = summarize_samples(samples)
        summary.insert(0, "date", future_dates)
        summary.insert(0, "store_nbr", store_nbr)
        summary["actual"] = actual_part["sales"].to_numpy()
        forecast_frames.append(summary)

        metrics = compute_all_metrics(
            y_true=summary["actual"].to_numpy(),
            y_pred=summary["p50"].to_numpy(),
            y_train_history=train_part["sales"].to_numpy(),
        )
        metrics["store_nbr"] = store_nbr
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows).set_index("store_nbr").sort_index()
    forecasts_df = pd.concat(forecast_frames, ignore_index=True)
    return metrics_df, forecasts_df


def main():
    panel = pd.read_parquet(config.DAILY_STORE_SALES_PATH)
    metrics_df, forecasts_df = backtest(panel)
    metrics_df.to_csv(config.TABLES_DIR / "backtest_metrics_per_store.csv")
    forecasts_df.to_csv(config.TABLES_DIR / "backtest_forecasts.csv", index=False)

    summary = metrics_df.mean(numeric_only=True).to_frame("value")
    summary.to_csv(config.TABLES_DIR / "backtest_metrics_overall.csv")
    print(summary)


if __name__ == "__main__":
    main()
