"""
Logging for training and prediction.

Every function takes test=True to write into *test* log files so unit tests never
touch production logs. Log directory can be overridden with the LOG_DIR env var.
"""
import os
import csv
import time
from datetime import date

LOG_DIR = os.environ.get("LOG_DIR", "logs")

TRAIN_HEADER = ["unique_id", "timestamp", "country", "model_name", "model_version",
                "cv_rmse", "holdout_rmse", "baseline_holdout_rmse", "runtime"]
PREDICT_HEADER = ["unique_id", "timestamp", "country", "target_date", "y_pred",
                  "model_name", "model_version", "runtime"]


def _log_path(kind, test):
    os.makedirs(LOG_DIR, exist_ok=True)
    if test:
        name = "{}-test.log".format(kind)
    else:
        today = date.today()
        name = "{}-{}-{}.log".format(kind, today.year, today.month)
    return os.path.join(LOG_DIR, name)


def _append(path, header, row):
    is_new = not os.path.exists(path)
    unique_id = 1
    if not is_new:
        with open(path) as fh:
            unique_id = max(sum(1 for _ in fh), 1)  # header counts as 1 line -> next id
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh, delimiter=",")
        if is_new:
            writer.writerow(header)
            unique_id = 1
        writer.writerow([unique_id, time.time()] + list(row))
    return path


def update_train_log(country, model_name, model_version, cv_rmse, holdout_rmse,
                     baseline_holdout_rmse, runtime, test=False):
    """Append one row to the training log."""
    return _append(_log_path("train", test), TRAIN_HEADER,
                   [country, model_name, model_version, cv_rmse, holdout_rmse,
                    baseline_holdout_rmse, runtime])


def update_predict_log(country, target_date, y_pred, model_name, model_version,
                       runtime, test=False):
    """Append one row to the prediction log."""
    return _append(_log_path("predict", test), PREDICT_HEADER,
                   [country, target_date, y_pred, model_name, model_version, runtime])


def list_logs():
    """Names of available log files."""
    if not os.path.isdir(LOG_DIR):
        return []
    return sorted(f for f in os.listdir(LOG_DIR) if f.endswith(".log"))


def read_log(filename):
    """Return a log file as a list of dict rows. Raises FileNotFoundError if missing."""
    safe = os.path.basename(filename)  # block path traversal
    path = os.path.join(LOG_DIR, safe)
    if not os.path.exists(path):
        raise FileNotFoundError(safe)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
