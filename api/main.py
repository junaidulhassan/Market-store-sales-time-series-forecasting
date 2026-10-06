"""FastAPI backend that serves the React dashboard.

Endpoints
---------
GET  /api/stores               -> store metadata (city, state, type, cluster)
GET  /api/overview             -> headline KPIs for the dashboard
GET  /api/history/{store_nbr}  -> recent daily sales history for a store
GET  /api/forecast/{store_nbr} -> next-N-day probabilistic forecast
POST /api/data                 -> append a new daily sales record
GET  /api/data/log             -> records added through the UI so far

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from threading import Lock

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.predict import load_model, forecast_for_store  # noqa: E402

app = FastAPI(title="Store Sales Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = Lock()
_state = {"panel": None, "model": None, "device": None, "stores": None}

USER_DATA_PATH = config.BACKEND_DATA_DIR / "user_added_data.csv"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def startup():
    panel = pd.read_parquet(config.DAILY_STORE_SALES_PATH)
    if USER_DATA_PATH.exists():
        extra = pd.read_csv(USER_DATA_PATH, parse_dates=["date"])
        panel = _merge_user_rows(panel, extra)
    _state["panel"] = panel
    _state["stores"] = pd.read_csv(config.DATA_RAW_DIR / "stores.csv")
    _state["device"] = _device()
    _state["model"] = load_model(_state["device"])


def _merge_user_rows(panel: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    extra = extra.copy()
    last_oil = panel["dcoilwtico"].iloc[-1]

    store_static_cols = [c for c in ("city", "state", "type", "cluster") if c in panel.columns]
    store_static = panel.drop_duplicates("store_nbr", keep="last").set_index("store_nbr")[store_static_cols]

    for col in store_static_cols:
        extra[col] = extra["store_nbr"].map(store_static[col])

    extra["dcoilwtico"] = last_oil
    extra["transactions"] = extra.get("transactions", 0.0)
    extra["is_holiday"] = extra.get("is_holiday", 0)
    extra["day_of_week"] = extra["date"].dt.dayofweek
    extra["day_of_month"] = extra["date"].dt.day
    extra["day_of_year"] = extra["date"].dt.dayofyear
    extra["month"] = extra["date"].dt.month
    extra["is_weekend"] = (extra["day_of_week"] >= 5).astype("int8")

    for col in panel.columns:
        if col not in extra.columns:
            extra[col] = 0
    extra = extra[panel.columns]

    merged = pd.concat([panel, extra], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "store_nbr"], keep="last")
    merged = merged.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    return merged


class NewSalesRecord(BaseModel):
    date: date
    store_nbr: int
    sales: float = Field(ge=0)
    onpromotion: int = Field(default=0, ge=0)


@app.get("/api/stores")
def get_stores():
    stores = _state["stores"]
    return stores.to_dict(orient="records")


@app.get("/api/overview")
def get_overview():
    panel = _state["panel"]
    last_date = panel["date"].max()
    last_30 = panel[panel["date"] > last_date - pd.Timedelta(days=30)]
    prev_30 = panel[
        (panel["date"] > last_date - pd.Timedelta(days=60))
        & (panel["date"] <= last_date - pd.Timedelta(days=30))
    ]
    total_last_30 = float(last_30["sales"].sum())
    total_prev_30 = float(prev_30["sales"].sum())
    pct_change = ((total_last_30 - total_prev_30) / total_prev_30 * 100.0) if total_prev_30 else 0.0

    return {
        "num_stores": int(panel["store_nbr"].nunique()),
        "last_date": str(last_date.date()),
        "total_sales_last_30d": round(total_last_30, 2),
        "pct_change_vs_prior_30d": round(pct_change, 2),
        "avg_daily_sales_per_store": round(total_last_30 / 30 / panel["store_nbr"].nunique(), 2),
    }


@app.get("/api/history/{store_nbr}")
def get_history(store_nbr: int, days: int = 120):
    panel = _state["panel"]
    g = panel[panel["store_nbr"] == store_nbr].sort_values("date").tail(days)
    if g.empty:
        raise HTTPException(status_code=404, detail="Unknown store_nbr")
    return [
        {"date": str(r.date.date()), "sales": round(float(r.sales), 2)}
        for r in g.itertuples()
    ]


@app.get("/api/forecast/{store_nbr}")
def get_forecast(store_nbr: int):
    panel = _state["panel"]
    if store_nbr not in panel["store_nbr"].unique():
        raise HTTPException(status_code=404, detail="Unknown store_nbr")
    with _lock:
        summary = forecast_for_store(panel, store_nbr, model=_state["model"], device=_state["device"])
    return [
        {
            "date": str(r.date.date()),
            "mean": round(float(r.mean), 2),
            "p10": round(float(r.p10), 2),
            "p50": round(float(r.p50), 2),
            "p90": round(float(r.p90), 2),
        }
        for r in summary.itertuples()
    ]


@app.post("/api/data")
def add_data(record: NewSalesRecord):
    with _lock:
        panel = _state["panel"]
        if record.store_nbr not in panel["store_nbr"].unique():
            raise HTTPException(status_code=404, detail="Unknown store_nbr")

        row = pd.DataFrame([record.model_dump()])
        row["date"] = pd.to_datetime(row["date"])

        config.BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if USER_DATA_PATH.exists():
            row.to_csv(USER_DATA_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(USER_DATA_PATH, mode="w", header=True, index=False)

        _state["panel"] = _merge_user_rows(panel, row)
    return {"status": "ok", "message": "Record added", "record": record.model_dump()}


@app.get("/api/data/log")
def data_log():
    if not USER_DATA_PATH.exists():
        return []
    extra = pd.read_csv(USER_DATA_PATH)
    return extra.to_dict(orient="records")
