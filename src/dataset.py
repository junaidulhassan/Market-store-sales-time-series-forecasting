from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config

TIME_FEATURE_COLUMNS = ["day_of_week", "day_of_month", "day_of_year", "month", "is_weekend"]


def normalize_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["day_of_week"] / 6.0 - 0.5
    out["day_of_month"] = out["day_of_month"] / 30.0 - 0.5
    out["day_of_year"] = out["day_of_year"] / 365.0 - 0.5
    out["month"] = out["month"] / 12.0 - 0.5
    out["is_weekend"] = out["is_weekend"].astype(float) - 0.5
    return out


@dataclass
class SeriesData:
    store_nbr: int
    values: np.ndarray            # (T,) float32 sales
    time_features: np.ndarray     # (T, num_time_features) float32
    dates: np.ndarray             # (T,) datetime64


def build_series(panel: pd.DataFrame) -> list[SeriesData]:
    panel = normalize_time_features(panel)
    series_list = []
    for store_nbr, g in panel.groupby("store_nbr"):
        g = g.sort_values("date")
        series_list.append(
            SeriesData(
                store_nbr=int(store_nbr),
                values=g["sales"].to_numpy(dtype=np.float32),
                time_features=g[TIME_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                dates=g["date"].to_numpy(),
            )
        )
    return series_list


class SalesWindowDataset(Dataset):
    """Produces sliding windows of
    (past_values, past_time_features, past_observed_mask,
     future_values, future_time_features, static_categorical_features)
    for every store series.
    """

    def __init__(
        self,
        series_list: list[SeriesData],
        context_length: int = config.CONTEXT_LENGTH,
        prediction_length: int = config.PREDICTION_LENGTH,
        max_lag: int = max(config.LAGS_SEQUENCE),
        mode: str = "train",
        holdout_days: int = config.VALID_DAYS,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_lag = max_lag
        self.history_length = context_length + max_lag
        self.mode = mode
        self.holdout_days = holdout_days

        self.index: list[tuple[int, int]] = []  # (series_idx, t) t = first index of future window
        self.series_list = series_list

        for s_idx, s in enumerate(series_list):
            T = len(s.values)
            usable_end = T - holdout_days if mode == "train" else T
            earliest_t = self.history_length
            if mode == "train":
                for t in range(earliest_t, usable_end - prediction_length + 1):
                    self.index.append((s_idx, t))
            else:  # one evaluation window per series: last `holdout_days` as future
                t = T - holdout_days
                if t >= earliest_t:
                    self.index.append((s_idx, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, t = self.index[idx]
        s = self.series_list[s_idx]

        past_values = s.values[t - self.history_length : t]
        past_time_feats = s.time_features[t - self.history_length : t]
        future_values = s.values[t : t + self.prediction_length]
        future_time_feats = s.time_features[t : t + self.prediction_length]
        past_observed_mask = np.ones_like(past_values, dtype=np.float32)

        return {
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "past_time_features": torch.tensor(past_time_feats, dtype=torch.float32),
            "past_observed_mask": torch.tensor(past_observed_mask, dtype=torch.float32),
            "future_values": torch.tensor(future_values, dtype=torch.float32),
            "future_time_features": torch.tensor(future_time_feats, dtype=torch.float32),
            "static_categorical_features": torch.tensor([s.store_nbr], dtype=torch.long),
        }


def make_inference_window(series: SeriesData, context_length: int, max_lag: int):
    """Build the single most-recent window used to forecast the next
    `prediction_length` days beyond the end of `series`.
    """
    history_length = context_length + max_lag
    past_values = series.values[-history_length:]
    past_time_feats = series.time_features[-history_length:]
    past_observed_mask = np.ones_like(past_values, dtype=np.float32)

    last_date = pd.Timestamp(series.dates[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=config.PREDICTION_LENGTH, freq="D")
    future_df = pd.DataFrame(
        {
            "day_of_week": future_dates.dayofweek,
            "day_of_month": future_dates.day,
            "day_of_year": future_dates.dayofyear,
            "month": future_dates.month,
            "is_weekend": (future_dates.dayofweek >= 5).astype(int),
        }
    )
    future_time_feats = normalize_time_features(future_df)[TIME_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    batch = {
        "past_values": torch.tensor(past_values, dtype=torch.float32).unsqueeze(0),
        "past_time_features": torch.tensor(past_time_feats, dtype=torch.float32).unsqueeze(0),
        "past_observed_mask": torch.tensor(past_observed_mask, dtype=torch.float32).unsqueeze(0),
        "future_time_features": torch.tensor(future_time_feats, dtype=torch.float32).unsqueeze(0),
        "static_categorical_features": torch.tensor([[series.store_nbr]], dtype=torch.long),
    }
    return batch, future_dates
