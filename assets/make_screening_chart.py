"""Regenerate the screening figure used in the README from the live DB.

    python assets/make_screening_chart.py

Produces assets/screening.png: per-creator raw beat-SPY rate vs the Wilson lower
bound (the conservative, sample-size-aware estimate the leaderboard actually ranks
on). The gap between the two is the small-sample discount; the dashed gate marks the
30-call eligibility bar. The point of the chart is honesty — it shows that high raw
rates on thin samples collapse toward 50% once you penalize for uncertainty.

Requires matplotlib (reporting-only; not part of the serving stack).
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db.storage import get_screening_leaderboard, MIN_ELIGIBLE_CALLS

def main():
    df = get_screening_leaderboard(min_calls=MIN_ELIGIBLE_CALLS)
    if df.empty:
        print("No benchmark-scored calls in the DB yet — nothing to plot.")
        return
    # Plot worst-to-best so the strongest creator sits on top.
    df = df.sort_values("wilson_lower_pct", ascending=True).reset_index(drop=True)

    elig_color, inelig_color = "#1f77b4", "#b0b0b0"
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(df) + 2))

    for i, r in df.iterrows():
        c = elig_color if r["eligible"] else inelig_color
        # Lollipop: line from Wilson lower bound up to the raw beat rate.
        ax.plot([r["wilson_lower_pct"], r["beat_spy_pct"]], [i, i], color=c, lw=2, zorder=1)
        ax.scatter(r["wilson_lower_pct"], i, color=c, s=40, zorder=2)                 # Wilson LB
        ax.scatter(r["beat_spy_pct"], i, color=c, s=90, zorder=3, edgecolor="white")  # raw rate
        ax.text(max(r["beat_spy_pct"], 50) + 1.5, i,
                f"n={r['calls']}" + ("  ✓ eligible" if r["eligible"] else ""),
                va="center", fontsize=9, color="#333")

    ax.axvline(50, color="#d62728", ls="--", lw=1.2, zorder=0)
    ax.text(50, len(df) - 0.3, "beat SPY 50%", color="#d62728", fontsize=8, ha="center")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["creator"])
    ax.set_xlabel("Beat-SPY rate (%)  ·  dot = raw rate, tail = Wilson lower bound")
    ax.set_title("Creator screening: raw beat-SPY rate vs sample-size-penalized Wilson bound",
                 fontsize=11, weight="bold")
    ax.set_xlim(0, 100)
    # Legend proxies.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=elig_color, lw=2, label=f"eligible (≥{MIN_ELIGIBLE_CALLS} calls)"),
        Line2D([0], [0], marker="o", color=inelig_color, lw=2, label="too few calls to screen"),
    ], loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = "assets/screening.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out} ({len(df)} creators)")

if __name__ == "__main__":
    main()
