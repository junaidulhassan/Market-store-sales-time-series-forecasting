from __future__ import annotations

import pandas as pd
import numpy as np

from src import config


def _load_raw():
    train = pd.read_csv(
        config.DATA_RAW_DIR / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "sales": "float32", "onpromotion": "int32"},
    )
    stores = pd.read_csv(config.DATA_RAW_DIR / "stores.csv")
    oil = pd.read_csv(config.DATA_RAW_DIR / "oil.csv", parse_dates=["date"])
    transactions = pd.read_csv(config.DATA_RAW_DIR / "transactions.csv", parse_dates=["date"])
    holidays = pd.read_csv(config.DATA_RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    return train, stores, oil, transactions, holidays


def _build_holiday_flags(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Return a (date, store_nbr) -> is_holiday frame.

    A day counts as a holiday for a store if there is a non-transferred
    holiday/event that is national, or regional/local and matches the
    store's state/city.
    """
    h = holidays[holidays["transferred"] == False].copy()  # noqa: E712
    national = set(h.loc[h["locale"] == "National", "date"])

    regional = h.loc[h["locale"] == "Regional", ["date", "locale_name"]].rename(
        columns={"locale_name": "state"}
    )
    local = h.loc[h["locale"] == "Local", ["date", "locale_name"]].rename(
        columns={"locale_name": "city"}
    )

    store_days = stores[["store_nbr", "city", "state"]].copy()

    reg_flags = store_days.merge(regional, on="state", how="inner")[["store_nbr", "date"]]
    loc_flags = store_days.merge(local, on="city", how="inner")[["store_nbr", "date"]]

    nat_flags = pd.MultiIndex.from_product(
        [stores["store_nbr"].tolist(), sorted(national)], names=["store_nbr", "date"]
    ).to_frame(index=False)

    flags = pd.concat([reg_flags, loc_flags, nat_flags], ignore_index=True).drop_duplicates()
    flags["is_holiday"] = 1
    return flags


def build_daily_store_panel() -> pd.DataFrame:
    train, stores, oil, transactions, holidays = _load_raw()

    daily = (
        train.groupby(["date", "store_nbr"], as_index=False)
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    )

    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    store_ids = sorted(stores["store_nbr"].unique())
    panel_index = pd.MultiIndex.from_product([full_dates, store_ids], names=["date", "store_nbr"])
    panel = pd.DataFrame(index=panel_index).reset_index()

    panel = panel.merge(daily, on=["date", "store_nbr"], how="left")
    panel["sales"] = panel["sales"].fillna(0.0)
    panel["onpromotion"] = panel["onpromotion"].fillna(0).astype("int32")

    # Oil price: forward/backward fill missing days, broadcast to every store.
    oil = oil.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    panel = panel.merge(oil, on="date", how="left")

    panel = panel.merge(transactions, on=["date", "store_nbr"], how="left")
    panel["transactions"] = panel["transactions"].fillna(0.0)

    holiday_flags = _build_holiday_flags(holidays, stores)
    panel = panel.merge(holiday_flags, on=["date", "store_nbr"], how="left")
    panel["is_holiday"] = panel["is_holiday"].fillna(0).astype("int8")

    panel = panel.merge(stores, on="store_nbr", how="left")

    # Calendar features used as model "time features".
    panel["day_of_week"] = panel["date"].dt.dayofweek
    panel["day_of_month"] = panel["date"].dt.day
    panel["day_of_year"] = panel["date"].dt.dayofyear
    panel["month"] = panel["date"].dt.month
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")

    panel = panel.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    return panel


def main():
    panel = build_daily_store_panel()
    panel.to_parquet(config.DAILY_STORE_SALES_PATH, index=False)
    print(f"Saved daily store panel: {panel.shape} -> {config.DAILY_STORE_SALES_PATH}")
    print(panel.head())


if __name__ == "__main__":
    main()
