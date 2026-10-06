"""Builds the small HuggingFace `TimeSeriesTransformerForPrediction` model
used throughout this project. The configuration is deliberately tiny
(d_model=32, 2 encoder/decoder layers, 2 attention heads) so that it trains
comfortably on a 4GB GPU such as the RTX A2000.
"""
from __future__ import annotations

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from src import config


def build_model(num_stores: int) -> TimeSeriesTransformerForPrediction:
    model_config = TimeSeriesTransformerConfig(
        prediction_length=config.PREDICTION_LENGTH,
        context_length=config.CONTEXT_LENGTH,
        num_time_features=config.NUM_TIME_FEATURES,
        num_static_categorical_features=1,
        cardinality=[num_stores + 1],  # +1 so 1-indexed store_nbr fits safely
        embedding_dimension=[8],
        lags_sequence=config.LAGS_SEQUENCE,
        **config.MODEL_CONFIG,
    )
    model = TimeSeriesTransformerForPrediction(model_config)
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
