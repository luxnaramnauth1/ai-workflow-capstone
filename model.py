"""
Model training, comparison, persistence and prediction.

Task: predict total revenue over the next 30 days from any point in time,
either for a single country or for all countries combined.
"""
import os
import re
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ingestion
from logger import update_train_log, update_predict_log

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_VERSION = "1.0"
DATA_DIRS = [d for d in os.environ.get("DATA_DIRS", "data/cs-train,data/cs-production").split(",") if d]
TOP_N_COUNTRIES = 10
RANDOM_STATE = 42


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Baseline: assume the next 30 days look like the last 28 days (scaled to 30 days)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["revenue_prev_28"]) * (ingestion.HORIZON / 28.0)


def candidate_models(fast=False):
    """name -> (estimator, param_grid). `fast=True` shrinks the grids (used by unit tests)."""
    return {
        "baseline": (NaiveBaseline(), {}),
        "ridge": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0]} if not fast else {"model__alpha": [1.0]},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [None, 5]} if not fast
            else {"n_estimators": [20], "max_depth": [3]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]} if not fast
            else {"n_estimators": [20], "learning_rate": [0.1], "max_depth": [2]},
        ),
    }


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compare_candidates(X, y, fast=False, holdout_frac=0.2):
    """
    Fit every candidate with time-ordered CV, score on a chronological holdout.
    A HORIZON-day gap separates train and holdout so targets don't overlap (no leakage).

    Returns dict with per-model scores, the best model name, holdout predictions and dates.
    """
    n = len(X)
    n_holdout = max(int(n * holdout_frac), 10)
    split = n - n_holdout
    gap = ingestion.HORIZON
    train_end = max(split - gap, 10)

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_hold, y_hold = X.iloc[split:], y[split:]

    n_splits = 3 if len(X_train) >= 40 else 2
    cv = TimeSeriesSplit(n_splits=n_splits)

    results, fitted, hold_preds = {}, {}, {}
    for name, (est, grid) in candidate_models(fast).items():
        if grid:
            search = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
            search.fit(X_train, y_train)
            best, cv_rmse = search.best_estimator_, float(-search.best_score_)
            best_params = search.best_params_
        else:
            best = est.fit(X_train, y_train)
            cv_scores = []
            for tr, va in cv.split(X_train):
                cv_scores.append(_rmse(y_train[va], best.predict(X_train.iloc[va])))
            cv_rmse, best_params = float(np.mean(cv_scores)), {}
        pred = best.predict(X_hold)
        results[name] = {"cv_rmse": cv_rmse, "holdout_rmse": _rmse(y_hold, pred), "params": best_params}
        fitted[name] = best
        hold_preds[name] = pred

    # Choose by CV error among real models (baseline is only a yardstick)
    contenders = {k: v for k, v in results.items() if k != "baseline"}
    best_name = min(contenders, key=lambda k: contenders[k]["cv_rmse"])
    return {"scores": results, "best_name": best_name, "holdout_dates": X_hold.index,
            "y_holdout": y_hold, "holdout_preds": hold_preds}


def _safe(country):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(country)).strip("_").lower()


def _model_path(country, test):
    prefix = "test-" if test else ""
    return os.path.join(MODEL_DIR, "{}sl-{}-v{}.joblib".format(prefix, _safe(country), MODEL_VERSION))


def model_train(data_dir="data/cs-train", test=False, countries=None, n_countries=None):
    """
    Train + compare candidate models for 'all' and the top-N countries by revenue,
    refit the winner on all data, persist it and write the training log.
    Returns a list of summary dicts (one per country).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = ingestion.fetch_data(data_dir)

    if n_countries is None:
        n_countries = 2 if test else TOP_N_COUNTRIES
    targets = ["all"] + (countries if countries is not None else ingestion.top_countries(df, n_countries))

    summaries = []
    for country in targets:
        t0 = time.time()
        ts = ingestion.convert_to_ts(df, None if country == "all" else country)
        try:
            X, y, dates = ingestion.engineer_features(ts)
        except ValueError as err:
            summaries.append({"country": country, "skipped": str(err)})
            continue

        comp = compare_candidates(X, y, fast=test)
        best_name = comp["best_name"]
        estimator = candidate_models(test)[best_name][0]
        estimator.set_params(**comp["scores"][best_name]["params"])
        estimator.fit(X, y)  # refit on ALL data for deployment

        bundle = {
            "model": estimator, "model_name": best_name, "version": MODEL_VERSION,
            "features": list(X.columns), "country": country, "scores": comp["scores"],
            "train_start": str(ts.index.min().date()), "train_end": str(ts.index.max().date()),
        }
        joblib.dump(bundle, _model_path(country, test))

        s = comp["scores"][best_name]
        update_train_log(country, best_name, MODEL_VERSION, s["cv_rmse"], s["holdout_rmse"],
                         comp["scores"]["baseline"]["holdout_rmse"], round(time.time() - t0, 2), test=test)
        summaries.append({"country": country, "model_name": best_name, "scores": comp["scores"]})
    _load_df.cache_clear()
    return summaries


def model_load(test=False):
    """Return {country: bundle} for every saved model of the requested kind."""
    prefix = "test-" if test else ""
    suffix = "-v{}.joblib".format(MODEL_VERSION)
    models = {}
    if not os.path.isdir(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        is_test_file = fname.startswith("test-")
        if fname.endswith(suffix) and is_test_file == test:
            bundle = joblib.load(os.path.join(MODEL_DIR, fname))
            models[bundle["country"]] = bundle
    return models


@lru_cache(maxsize=4)
def _load_df(key):
    """Load (and cache) the combined raw data for a tuple of data directories."""
    frames = [ingestion.fetch_data(d) for d in key if os.path.isdir(d)]
    if not frames:
        raise FileNotFoundError("none of the data directories exist: {}".format(list(key)))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)



def model_predict(country, year, month, day, data_dirs=None, test=False, models=None):
    """
    Forecast revenue for the 30 days starting on year-month-day.
    Uses only data BEFORE that date as input. Raises ValueError/KeyError for bad input.
    """
    t0 = time.time()
    country = "all" if country in (None, "", "all") else country

    try:
        target = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except (TypeError, ValueError) as err:
        raise ValueError("invalid date: {}".format(err))

    if models is None:
        models = model_load(test=test)
    if not models:
        raise RuntimeError("no trained models found; call train first")
    if country not in models:
        raise KeyError("no model for country '{}'. Available: {}".format(country, sorted(models)))

    bundle = models[country]
    key = tuple(data_dirs or DATA_DIRS)
    df = _load_df(key)
    ts = ingestion.convert_to_ts(df, None if country == "all" else country)

    X = ingestion.features_for_date(ts, target)[bundle["features"]]
    y_pred = float(bundle["model"].predict(X)[0])
    actual = ingestion.actual_next_30(ts, target)

    update_predict_log(country, str(target.date()), round(y_pred, 2), bundle["model_name"],
                       bundle["version"], round(time.time() - t0, 4), test=test)
    return {"country": country, "target_date": str(target.date()), "y_pred": y_pred,
            "actual": actual, "model_name": bundle["model_name"], "model_version": bundle["version"]}


if __name__ == "__main__":
    import sys
    for row in model_train(sys.argv[1] if len(sys.argv) > 1 else "data/cs-train"):
        print(row["country"], row.get("model_name", row.get("skipped")))
