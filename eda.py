"""
Exploratory data analysis with visualizations.

    python eda.py [data_dir]

Saves figures to reports/eda/ and prints data-quality checks.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ingestion


def main(data_dir="data/cs-train", out_dir="reports/eda"):
    os.makedirs(out_dir, exist_ok=True)
    df = ingestion.fetch_data(data_dir)

    print("rows: {}  date range: {} to {}".format(len(df), df["date"].min().date(), df["date"].max().date()))
    print("\nmissing values per column:\n", df.isnull().sum().to_string())
    print("\nprice summary:\n", df["price"].describe().to_string())
    print("\nnegative / zero prices: {}".format(int((df["price"] <= 0).sum())))

    # 1. revenue by country (top 10)
    by_country = df.groupby("country")["price"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    by_country.plot.bar(ax=ax)
    ax.set_ylabel("total revenue")
    ax.set_title("Top 10 countries by total revenue")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "revenue_by_country.png"), dpi=130)
    plt.close(fig)

    # 2. daily revenue / purchases / views time series (all countries)
    ts = ingestion.convert_to_ts(df)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, col in zip(axes, ["revenue", "purchases", "total_views"]):
        ts[col].plot(ax=ax, linewidth=0.8)
        ts[col].rolling(28).mean().plot(ax=ax, color="red", linewidth=1.5, label="28-day mean")
        ax.set_ylabel(col)
    axes[0].legend()
    axes[0].set_title("Daily time series, all countries")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "daily_time_series.png"), dpi=130)
    plt.close(fig)

    # 3. seasonality: revenue per calendar month
    monthly = ts["revenue"].resample("MS").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(monthly)), monthly.values)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in monthly.index], rotation=60, ha="right")
    ax.set_ylabel("monthly revenue")
    ax.set_title("Monthly revenue (look for seasonal peaks)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "monthly_revenue.png"), dpi=130)
    plt.close(fig)

    # 4. correlation between the daily engineered features
    corr = ts.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    ax.set_title("Correlation of daily features")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_correlation.png"), dpi=130)
    plt.close(fig)

    # 5. distribution of order values (log-scaled counts shows the long tail)
    fig, ax = plt.subplots(figsize=(7, 4))
    df["price"].clip(upper=df["price"].quantile(0.99)).hist(bins=50, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("price (clipped at 99th percentile)")
    ax.set_title("Distribution of transaction price")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "price_distribution.png"), dpi=130)
    plt.close(fig)

    print("\nsaved figures to", out_dir)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
