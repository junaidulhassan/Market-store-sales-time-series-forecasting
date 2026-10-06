"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train_history: np.ndarray, seasonality: int = 7) -> float:
    """Mean Absolute Scaled Error against a naive seasonal (weekly) baseline
    computed from the training history that precedes the forecast window.
    """
    naive_errors = np.abs(y_train_history[seasonality:] - y_train_history[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_history: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train_history),
    }
