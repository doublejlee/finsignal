"""Regenerate the 'follow the proven creators' equity curve for the README from the live DB.

    python assets/make_equity_curve.py

Produces assets/equity_curve.png: compounded directional return of acting on every non-neutral
call from proven creators (long bull / short bear, 30-day hold) vs holding SPY over the same
windows. Illustrative — equal weight, no costs, overlapping windows. Requires matplotlib.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from db.storage import get_strategy_performance

def main():
    df, m = get_strategy_performance()
    if df.empty:
        print("No proven-creator trades yet — nothing to plot.")
        return

    x = pd.to_datetime(df["month"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, df["strategy"], color="#1f77b4", lw=2, marker="o", ms=4, label="Follow proven creators")
    ax.plot(x, df["spy"], color="#888", lw=2, ls="--", marker="o", ms=4, label="Hold SPY")
    ax.axhline(1.0, color="#ccc", lw=1, zorder=0)

    ax.set_title("Following the proven creators vs holding SPY (backtest, illustrative)",
                 fontsize=12, weight="bold")
    ax.set_ylabel("Growth of $1 (monthly cohorts, compounded)")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    note = (f"{m['trades']} bullish calls over {m['months']} months  ·  strategy "
            f"{m['strategy_return_pct']:+}%  vs  SPY {m['spy_return_pct']:+}%\n"
            f"win rate {m['win_rate_pct']}%  ·  avg trade {m['avg_trade_pct']:+}%  ·  "
            f"max drawdown {m['max_drawdown_pct']}%")
    ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333")

    fig.tight_layout()
    out = "assets/equity_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    print("metrics:", m)

if __name__ == "__main__":
    main()
