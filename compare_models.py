"""
Compare candidate models against the baseline and visualise the result.

    python compare_models.py [data_dir]

Writes reports/model_comparison.png and reports/model_comparison.csv.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import ingestion
import model


def main(data_dir="data/cs-train", out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    df = ingestion.fetch_data(data_dir)
    ts = ingestion.convert_to_ts(df)
    X, y, dates = ingestion.engineer_features(ts)
    comp = model.compare_candidates(X, y)

    table = pd.DataFrame(comp["scores"]).T[["cv_rmse", "holdout_rmse"]]
    table["improvement_vs_baseline_%"] = 100 * (1 - table["holdout_rmse"] / table.loc["baseline", "holdout_rmse"])
    print(table.round(2).to_string())
    print("\nselected model:", comp["best_name"])
    table.to_csv(os.path.join(out_dir, "model_comparison.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    table["holdout_rmse"].plot.bar(ax=axes[0], color=["grey"] + ["steelblue"] * (len(table) - 1))
    axes[0].set_ylabel("holdout RMSE (lower is better)")
    axes[0].set_title("Models vs baseline (grey) on chronological holdout")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(comp["holdout_dates"], comp["y_holdout"], label="actual", color="black")
    axes[1].plot(comp["holdout_dates"], comp["holdout_preds"]["baseline"], label="baseline", linestyle="--", color="grey")
    axes[1].plot(comp["holdout_dates"], comp["holdout_preds"][comp["best_name"]], label=comp["best_name"], color="steelblue")
    axes[1].set_ylabel("revenue, next 30 days")
    axes[1].set_title("Holdout forecasts: selected model vs baseline")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=130)
    print("saved", os.path.join(out_dir, "model_comparison.png"))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
