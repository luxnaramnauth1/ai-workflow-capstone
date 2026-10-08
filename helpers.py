"""Shared helpers: a small synthetic dataset in the same JSON format as the AAVAIL files."""
import json
import os
import tempfile
import shutil

import numpy as np

COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE"]
WEIGHTS = [0.6, 0.15, 0.15, 0.1]


def make_dataset(out_dir, days=400, start="2018-01-01", seed=0):
    """Write one JSON file per month into out_dir. Returns out_dir."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)
    dates = pd.date_range(start, periods=days, freq="D")
    by_month = {}
    invoice = 100000
    for d in dates:
        season = 1.0 + 0.4 * np.sin(2 * np.pi * d.dayofyear / 365.0)
        for _ in range(rng.poisson(12 * season)):
            invoice += 1
            c = rng.choice(len(COUNTRIES), p=WEIGHTS)
            by_month.setdefault((d.year, d.month), []).append({
                "country": COUNTRIES[c], "customer_id": int(rng.randint(1000, 1100)),
                "day": str(d.day), "invoice": str(invoice), "month": str(d.month),
                "price": round(float(rng.gamma(2.0, 6.0)), 2),
                "stream_id": str(rng.randint(1, 30)), "times_viewed": int(rng.randint(1, 10)),
                "year": str(d.year),
            })
    for (y, mth), rows in by_month.items():
        with open(os.path.join(out_dir, "invoices-{}-{:02d}.json".format(y, mth)), "w") as fh:
            json.dump(rows, fh)
    return out_dir


def make_tmp_dataset(**kwargs):
    tmp = tempfile.mkdtemp(prefix="capstone-data-")
    return make_dataset(tmp, **kwargs)


def remove(path):
    shutil.rmtree(path, ignore_errors=True)


def clean_test_artifacts(model_dir, log_dir):
    """Delete only files created in test mode, never production models or logs."""
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.startswith("test-"):
                os.remove(os.path.join(model_dir, f))
    if os.path.isdir(log_dir):
        for f in os.listdir(log_dir):
            if f.endswith("-test.log"):
                os.remove(os.path.join(log_dir, f))
