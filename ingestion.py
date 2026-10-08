"""
Data ingestion and feature engineering.

fetch_data()       -> one clean DataFrame from a directory of JSON invoice files
convert_to_ts()    -> daily time series (optionally for a single country)
engineer_features()-> X, y, dates where y = revenue over the NEXT 30 days
"""
import os
import re
import json

import numpy as np
import pandas as pd

HORIZON = 30                      # forecast horizon in days
WINDOWS = [7, 14, 28, 56, 84]     # look-back windows (days) used as features

COLUMN_MAP = {
    "total_price": "price",
    "StreamID": "stream_id",
    "TimesViewed": "times_viewed",
    "invoice_id": "invoice",
}


def fetch_data(data_dir):
    """Load every .json file in data_dir into a single clean DataFrame."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError("data directory '{}' not found".format(data_dir))

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError("no .json files found in '{}'".format(data_dir))

    frames = []
    for name in files:
        with open(os.path.join(data_dir, name)) as fh:
            frames.append(pd.DataFrame(json.load(fh)))
    df = pd.concat(frames, ignore_index=True, sort=True)

    df = df.rename(columns=COLUMN_MAP)

    # Invoice ids sometimes contain letters (e.g. 'C' for cancellations / 'A' for adjustments)
    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D+", "", regex=True)
    df["invoice"] = pd.to_numeric(df["invoice"], errors="coerce")

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
        + "-" + df["day"].astype(str).str.zfill(2),
        errors="coerce",
    )

    df = df.dropna(subset=["date", "price"])
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def top_countries(df, n=10):
    """Names of the n countries with the highest total revenue."""
    return df.groupby("country")["price"].sum().sort_values(ascending=False).head(n).index.tolist()


def convert_to_ts(df, country=None):
    """Aggregate transactions into a complete daily time series (missing days filled with 0)."""
    if country is not None and country != "all":
        if country not in set(df["country"].unique()):
            raise KeyError("country '{}' not found in data".format(country))
        df = df[df["country"] == country]

    daily = df.groupby("date").agg(
        purchases=("price", "size"),
        unique_invoices=("invoice", "nunique"),
        unique_streams=("stream_id", "nunique"),
        total_views=("times_viewed", "sum"),
        revenue=("price", "sum"),
    )
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0).rename_axis("date")


def _features_at(ts, date):
    """Features for forecasting the 30 days starting at `date`, using only data BEFORE `date`."""
    feats = {}
    for w in WINDOWS:
        window = ts.loc[date - pd.Timedelta(days=w): date - pd.Timedelta(days=1)]
        feats["revenue_prev_{}".format(w)] = window["revenue"].sum()
    recent = ts.loc[date - pd.Timedelta(days=28): date - pd.Timedelta(days=1)]
    feats["recent_invoices"] = recent["unique_invoices"].sum()
    feats["recent_views"] = recent["total_views"].sum()
    feats["recent_purchases"] = recent["purchases"].sum()
    return feats


def engineer_features(ts, training=True):
    """
    Training: returns X (DataFrame), y (ndarray), dates (DatetimeIndex)
    where every row is a point in time and y is the revenue summed over the following 30 days.
    """
    first_valid = ts.index.min() + pd.Timedelta(days=max(WINDOWS))
    last_valid = ts.index.max() - pd.Timedelta(days=HORIZON - 1)
    dates = pd.date_range(first_valid, last_valid, freq="D")
    if len(dates) < 20:
        raise ValueError("not enough history to engineer features ({} usable rows)".format(len(dates)))

    rows, targets = [], []
    for d in dates:
        rows.append(_features_at(ts, d))
        targets.append(ts.loc[d: d + pd.Timedelta(days=HORIZON - 1), "revenue"].sum())

    X = pd.DataFrame(rows, index=dates)
    y = np.array(targets)
    return X, y, dates


def features_for_date(ts, date):
    """Single-row feature frame used by predict(). Date must have enough history before it."""
    date = pd.Timestamp(date)
    earliest = ts.index.min() + pd.Timedelta(days=max(WINDOWS))
    latest = ts.index.max() + pd.Timedelta(days=1)
    if date < earliest or date > latest:
        raise ValueError(
            "date {} is outside the supported range {} to {}".format(
                date.date(), earliest.date(), latest.date())
        )
    return pd.DataFrame([_features_at(ts, date)], index=[date])


def actual_next_30(ts, date):
    """Actual revenue for the 30 days starting at date, or None if not fully observed yet."""
    date = pd.Timestamp(date)
    end = date + pd.Timedelta(days=HORIZON - 1)
    if end > ts.index.max():
        return None
    return float(ts.loc[date:end, "revenue"].sum())
