"""
Performance monitoring.

performance_report() re-scores every deployed model on the most recent data, compares the
error with the error measured at training time and with the naive baseline, translates it
into the business metric (error as a % of revenue) and runs novelty detection on the inputs.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error

import ingestion
import model as m

DRIFT_RATIO = 1.5          # recent RMSE > 1.5x training holdout RMSE  -> flag drift
BUSINESS_THRESHOLD = 0.25  # assumed: forecast error above 25% of 30-day revenue is not acceptable


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def performance_report(data_dirs=None, test=False, recent_rows=60,
                       drift_ratio=DRIFT_RATIO, business_threshold=BUSINESS_THRESHOLD):
    """Return {country: metrics} for every deployed model."""
    models = m.model_load(test=test)
    if not models:
        raise RuntimeError("no trained models found; call train first")
    df = m._load_df(tuple(data_dirs or m.DATA_DIRS))

    report = {}
    for country, bundle in models.items():
        ts = ingestion.convert_to_ts(df, None if country == "all" else country)
        try:
            X, y, dates = ingestion.engineer_features(ts)
        except ValueError:
            continue
        X = X[bundle["features"]]
        k = min(recent_rows, max(len(X) // 3, 5))
        X_recent, y_recent = X.iloc[-k:], y[-k:]

        pred = bundle["model"].predict(X_recent)
        base_pred = np.asarray(X_recent["revenue_prev_28"]) * (ingestion.HORIZON / 28.0)
        rmse, base_rmse = _rmse(y_recent, pred), _rmse(y_recent, base_pred)
        train_rmse = bundle["scores"][bundle["model_name"]]["holdout_rmse"]
        mean_rev = float(np.mean(y_recent)) or 1.0
        rel_error = rmse / mean_rev

        # Novelty detection: are recent inputs unlike what the model was trained on?
        X_hist = X.iloc[:-k] if len(X) - k >= 20 else X
        iso = IsolationForest(random_state=m.RANDOM_STATE, contamination=0.05).fit(X_hist)
        novelty_rate = float(np.mean(iso.predict(X_recent) == -1))

        report[country] = {
            "model_name": bundle["model_name"],
            "recent_rmse": rmse,
            "baseline_recent_rmse": base_rmse,
            "training_holdout_rmse": train_rmse,
            "rmse_ratio_vs_training": rmse / train_rmse if train_rmse else None,
            "beats_baseline": bool(rmse < base_rmse),
            "mean_30day_revenue": mean_rev,
            "relative_error": rel_error,
            "novelty_rate": novelty_rate,
            "performance_drift": bool(train_rmse and rmse > drift_ratio * train_rmse),
            "meets_business_target": bool(rel_error <= business_threshold),
            "recent_dates": [str(dates[-k].date()), str(dates[-1].date())],
        }
    return report
