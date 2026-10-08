"""
Post-production analysis: how does model performance relate to the business metric (revenue)?

    python post_production_analysis.py

Writes reports/post_production_analysis.png and prints a per-country summary table.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import monitoring


def main(out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    report = monitoring.performance_report()
    table = pd.DataFrame(report).T[[
        "model_name", "recent_rmse", "baseline_recent_rmse", "relative_error",
        "novelty_rate", "performance_drift", "meets_business_target"]]
    print(table.to_string())
    table.to_csv(os.path.join(out_dir, "post_production_summary.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rev = [v["mean_30day_revenue"] for v in report.values()]
    rel = [v["relative_error"] for v in report.values()]
    axes[0].scatter(rev, rel)
    for name, x, y in zip(report, rev, rel):
        axes[0].annotate(name, (x, y), fontsize=8)
    axes[0].axhline(monitoring.BUSINESS_THRESHOLD, color="red", linestyle="--", label="business threshold")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("mean 30-day revenue (log scale)")
    axes[0].set_ylabel("RMSE / mean revenue")
    axes[0].set_title("Model error vs business metric")
    axes[0].legend()

    names = list(report)
    axes[1].bar([i - 0.2 for i in range(len(names))], [report[n]["recent_rmse"] for n in names], 0.4, label="model")
    axes[1].bar([i + 0.2 for i in range(len(names))], [report[n]["baseline_recent_rmse"] for n in names], 0.4, label="baseline")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=60, ha="right")
    axes[1].set_ylabel("recent RMSE")
    axes[1].set_title("Model vs baseline on most recent data")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "post_production_analysis.png"), dpi=130)
    print("saved", os.path.join(out_dir, "post_production_analysis.png"))


if __name__ == "__main__":
    main()
