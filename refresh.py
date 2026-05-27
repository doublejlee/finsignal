"""Local ingestion orchestrator — runs the full pipeline in order with VPN prompts.

The stages have opposite network needs (Stage 1 wants the VPN on for YouTube; Stage 2
wants it off because Groq blocks VPN IPs), so this can't run fully unattended without
paid residential proxies. It pauses for the toggle instead. After it finishes, commit
and push db/finsignal.duckdb to refresh the deployed Streamlit dashboard.

    python refresh.py
"""
import subprocess
import sys

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"Stage failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("FinSignal refresh — runs ingestion, reason extraction, and backtest.\n")

    input("1/3  Turn the VPN ON (YouTube needs it), then press Enter...")
    run(["main.py"])

    input("\n2/3  Turn the VPN OFF (Groq blocks VPN IPs), then press Enter...")
    run(["-m", "nlp.backfill_reasons"])

    print("\n3/3  Backtest (no VPN needed)...")
    run(["backtest.py"])

    print("\nDone. To update the live dashboard:")
    print("  git add db/finsignal.duckdb && git commit -m 'data refresh' && git push")
