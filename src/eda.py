"""EDA: does the bad-review rate grow with delivery delay, shipping time and freight cost?
Prints the tables behind each plot and saves PNGs to reports/."""
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DB_PATH, ROOT

REPORTS = ROOT / "reports"
BAR, LINE, GRID = "#2a6f97", "#2a6f97", "#dddddd"  # one hue: all plots show one series


def rate_by(df, bucket_col):
    """Bad-review rate and order count per bucket."""
    out = df.groupby(bucket_col, observed=True)["bad_review"].agg(rate="mean", n="size")
    return out.reset_index()


def style(ax, title, xlabel, base):
    ax.axhline(base, color="#888888", lw=1, ls="--", label=f"overall rate {base:.1%}")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bad-review rate (score <= 2)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM features").df()
    base = df["bad_review"].mean()

    # 1) delay buckets (days; negative = delivered earlier than promised)
    edges = [-float("inf"), -10, -5, 0, 3, 7, 14, float("inf")]
    labels = ["early >10d", "early 5-10d", "early 0-5d", "late 0-3d", "late 3-7d", "late 7-14d", "late >14d"]
    df["delay_bucket"] = pd.cut(df["delivery_delay_days"], edges, labels=labels)
    t1 = rate_by(df, "delay_bucket")
    print("\nbad-review rate by delay bucket\n", t1.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(t1["delay_bucket"].astype(str), t1["rate"], color=BAR, width=0.6)
    for x, (r, n) in enumerate(zip(t1["rate"], t1["n"])):
        ax.text(x, r, f"{r:.0%}\nn={n:,}", ha="center", va="bottom", fontsize=8)
    style(ax, "Late deliveries get far more bad reviews", "delivery vs. promised date", base)
    ax.set_ylim(0, t1["rate"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "eda_delay.png", dpi=150)

    # 2) shipping time and freight ratio: deciles (10 equal-sized groups)
    df["ship_decile"] = pd.qcut(df["shipping_days"], 10, duplicates="drop")
    df["freight_decile"] = pd.qcut(df["freight_ratio"], 10, duplicates="drop")
    t2 = rate_by(df, "ship_decile")
    t3 = rate_by(df, "freight_decile")
    print("\nbad-review rate by shipping_days decile\n", t2.round(3).to_string(index=False))
    print("\nbad-review rate by freight_ratio decile\n", t3.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, t, col, title, xl in [
        (axes[0], t2, "ship_decile", "Longer shipping, more bad reviews", "shipping days (decile 1 = fastest)"),
        (axes[1], t3, "freight_decile", "Freight cost matters less", "freight / price (decile 1 = lowest)"),
    ]:
        ax.plot(range(1, len(t) + 1), t["rate"], color=LINE, lw=2, marker="o", ms=5)
        style(ax, title, xl, base)
        ax.set_xticks(range(1, len(t) + 1))
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(REPORTS / "eda_shipping_freight.png", dpi=150)

    # 3) seller track record (only sellers with >= 30 earlier reviews)
    s = df[df["seller_prev_reviews"] >= 30].copy()
    s["seller_bucket"] = pd.cut(s["seller_prev_bad_rate"], [-0.001, 0.05, 0.10, 0.15, 0.20, 1.0],
                                labels=["<=5%", "5-10%", "10-15%", "15-20%", ">20%"])
    t4 = rate_by(s, "seller_bucket")
    print("\nbad-review rate by seller's earlier bad-review rate (>=30 earlier reviews)\n",
          t4.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t4["seller_bucket"].astype(str), t4["rate"], color=BAR, width=0.6)
    for x, (r, n) in enumerate(zip(t4["rate"], t4["n"])):
        ax.text(x, r, f"{r:.0%}\nn={n:,}", ha="center", va="bottom", fontsize=8)
    style(ax, "A seller's past reviews predict this order's review", "seller's earlier bad-review rate", base)
    ax.set_ylim(0, t4["rate"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "eda_seller.png", dpi=150)

    # how much do the features overlap? (late orders are also slow orders)
    print("\ncorrelation with bad_review (Pearson, rough):")
    cols = ["delivery_delay_days", "shipping_days", "freight_ratio", "total_freight", "total_price", "seller_prev_bad_rate"]
    print(df[cols + ["bad_review"]].corr()["bad_review"].drop("bad_review").round(3).to_string())
    print("\nsaved plots to", REPORTS)


if __name__ == "__main__":
    main()
