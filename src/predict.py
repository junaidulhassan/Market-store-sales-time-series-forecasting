"""Load the trained model and produce probabilistic forecasts for one or
more stores, either for backtesting (known future) or for genuine future
dates beyond the end of the available data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from transformers import TimeSeriesTransformerForPrediction

from src import config
from src.dataset import SeriesData, build_series, make_inference_window


def load_model(device: torch.device | None = None) -> TimeSeriesTransformerForPrediction:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformerForPrediction.from_pretrained(config.MODEL_DIR)
    model.to(device)
    model.eval()
    return model


def load_metadata() -> dict:
    with open(config.METADATA_PATH) as f:
        return json.load(f)


@torch.no_grad()
def forecast_series(model, series: SeriesData, device: torch.device, num_samples: int = config.NUM_SAMPLES_FOR_EVAL):
    """Forecast `prediction_length` days beyond the end of `series`.

    Returns (future_dates, samples) where samples has shape
    (num_samples, prediction_length).
    """
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
    samples = outputs.sequences.cpu().numpy()[0]  # (num_samples_model, prediction_length)
    samples = np.clip(samples, a_min=0.0, a_max=None)  # sales cannot be negative
    return future_dates, samples


def summarize_samples(samples: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": samples.mean(axis=0),
            "p10": np.percentile(samples, 10, axis=0),
            "p50": np.percentile(samples, 50, axis=0),
            "p90": np.percentile(samples, 90, axis=0),
        }
    )


def forecast_for_store(panel: pd.DataFrame, store_nbr: int, model=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or load_model(device)

    store_panel = panel[panel["store_nbr"] == store_nbr]
    series = build_series(store_panel)[0]
    future_dates, samples = forecast_series(model, series, device)
    summary = summarize_samples(samples)
    summary.insert(0, "date", future_dates)
    summary.insert(0, "store_nbr", store_nbr)
    return summary


def forecast_for_all_stores(panel: pd.DataFrame, model=None, device=None) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or load_model(device)

    results = []
    for store_nbr in sorted(panel["store_nbr"].unique()):
        results.append(forecast_for_store(panel, store_nbr, model=model, device=device))
    return pd.concat(results, ignore_index=True)
