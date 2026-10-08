"""
Flask API.

    GET  /                    health check / usage
    POST /train               {"mode": "test" | "prod"}         -> train + compare models
    POST /predict             {"country": "all", "year": 2019, "month": 5, "day": 1, "mode": "prod"}
    GET  /logs                list log files
    GET  /logs/<filename>     contents of one log file (JSON)
    GET  /monitor?mode=prod   performance / drift / novelty report
"""
import os

from flask import Flask, jsonify, request

import logger
import model
import monitoring

app = Flask(__name__)
TRAIN_DIR = os.environ.get("TRAIN_DIR", "data/cs-train")


def _is_test(payload):
    return str((payload or {}).get("mode", "prod")).lower() == "test"


@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "revenue-forecast", "endpoints": ["/train", "/predict", "/logs", "/monitor"]})


@app.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    test = _is_test(payload)
    data_dir = payload.get("data_dir", TRAIN_DIR)
    try:
        summaries = model.model_train(data_dir, test=test)
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 404
    return jsonify({"trained": [s["country"] for s in summaries if "model_name" in s],
                    "skipped": [s for s in summaries if "skipped" in s],
                    "summary": summaries})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required with country, year, month, day"}), 400
    missing = [k for k in ("year", "month", "day") if k not in payload]
    if missing:
        return jsonify({"error": "missing field(s): {}".format(", ".join(missing))}), 400
    try:
        result = model.model_predict(payload.get("country", "all"), payload["year"], payload["month"],
                                     payload["day"], data_dirs=payload.get("data_dirs"),
                                     test=_is_test(payload))
    except KeyError as err:
        return jsonify({"error": str(err).strip('"')}), 404
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except (RuntimeError, FileNotFoundError) as err:
        return jsonify({"error": str(err)}), 503
    return jsonify(result)


@app.route("/logs", methods=["GET"])
def logs():
    return jsonify({"files": logger.list_logs()})


@app.route("/logs/<filename>", methods=["GET"])
def get_log(filename):
    try:
        return jsonify({"file": filename, "rows": logger.read_log(filename)})
    except FileNotFoundError:
        return jsonify({"error": "log file '{}' not found".format(filename)}), 404


@app.route("/monitor", methods=["GET"])
def monitor():
    test = request.args.get("mode", "prod").lower() == "test"
    data_dirs = request.args.get("data_dirs")
    try:
        report = monitoring.performance_report(data_dirs=data_dirs.split(",") if data_dirs else None, test=test)
    except (RuntimeError, FileNotFoundError) as err:
        return jsonify({"error": str(err)}), 503
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
