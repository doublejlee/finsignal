"""Out-of-sample validation for the creator-screening leaderboard.

The leaderboard (db.storage.get_screening_leaderboard) ranks creators on the Wilson
lower bound of their beat-SPY rate across *all* their calls — an in-sample metric. This
script asks the harder question: does that skill persist on data the ranking never saw?

For each creator we split benchmark-scored calls chronologically (oldest half = in-sample
'train', newest half = out-of-sample 'holdout') and compare beat-SPY performance across
the two. If a creator looks skilled in-sample but their out-of-sample rate falls below
50%, the leaderboard ranking is likely luck or overfit rather than durable edge.

Run after backtest.py (needs populated beat_benchmark). Read-only — no DB writes.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from db.storage import get_out_of_sample_validation

SPLIT_FRACTION = 0.5
MIN_PER_SPLIT = 10

def run():
    df = get_out_of_sample_validation(split_fraction=SPLIT_FRACTION, min_per_split=MIN_PER_SPLIT)

    if df.empty:
        print(f"No creator has at least {MIN_PER_SPLIT} benchmark-scored calls in BOTH the "
              f"in-sample and out-of-sample halves yet.")
        print("Out-of-sample validation needs more evaluated history — re-run after the "
              "backtest picks up more calls past the 30-day horizon.")
        return

    print(f"Out-of-sample validation (split={SPLIT_FRACTION:.0%} oldest=in-sample, "
          f"min {MIN_PER_SPLIT} calls per half)\n")
    print(df.to_string(index=False))

    print("\nReading:")
    for _, r in df.iterrows():
        verdict = "PERSISTED" if r["persisted"] else "DECAYED"
        print(f"  [{verdict}] {r['creator']}: in-sample {r['in_sample_beat_pct']}% beat-SPY "
              f"({r['in_sample_calls']} calls) -> out-of-sample {r['oos_beat_pct']}% "
              f"({r['oos_calls']} calls), delta {r['delta_pct']:+}%")

if __name__ == "__main__":
    run()
