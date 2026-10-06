"""Central configuration for the sales forecasting pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data_raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data_processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
BACKEND_DATA_DIR = ROOT_DIR / "backend_data"

for d in [DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, BACKEND_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DAILY_STORE_SALES_PATH = DATA_PROCESSED_DIR / "daily_store_sales.parquet"
SCALERS_PATH = MODELS_DIR / "scalers.json"
MODEL_DIR = MODELS_DIR / "time_series_transformer"
METADATA_PATH = MODEL_DIR / "metadata.json"

# -- Series / windowing configuration --------------------------------------
FREQ = "D"
CONTEXT_LENGTH = 60          # days of history the encoder sees
PREDICTION_LENGTH = 14       # forecast horizon ("next weeks")
LAGS_SEQUENCE = [1, 2, 3, 7, 14, 21, 28]
NUM_TIME_FEATURES = 5        # day_of_week, day_of_month, day_of_year, month, is_weekend

# -- Model configuration (kept small to fit a 4GB GPU) ----------------------
MODEL_CONFIG = dict(
    d_model=32,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=32,
    decoder_ffn_dim=32,
    dropout=0.1,
    scaling="mean",
)

# -- Training configuration --------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_SAMPLES_FOR_EVAL = 100   # probabilistic samples drawn at inference time
VALID_DAYS = 28              # held-out days at the end of each series for validation/test
RANDOM_SEED = 42
