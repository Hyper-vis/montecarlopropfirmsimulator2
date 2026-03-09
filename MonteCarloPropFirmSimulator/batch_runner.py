"""
batch_runner.py
---------------
Run both simulations (run_until_payout + run_full_period) on every CSV
in this folder and print a ranked comparison table.

v3.1: uses apex_engine_v3_1 — supports uniform, recency_weighted, and
recent_only sampling modes. Set SAMPLING_MODE below to activate.

Usage:
    python batch_runner.py                  # scans all *.csv in this folder
    python batch_runner.py file1.csv file2.csv ...   # explicit list

Output:
    - Summary table ranked by net EV (printed to terminal)
    - batch_results.csv  (machine-readable)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from apex_engine_v3_1 import load_daily_pnl, run_simulations
from strategy_score import score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SIMS       = 20000   # sims per CSV per mode
MAX_DAYS_FP  = 21      # full-period window (trading days)
MAX_DAYS_UTP = 90      # until-payout cap
RESET_COST   = 137     # Apex reset fee ($)

# v3.1 sampling mode applied to every CSV in the batch
# "uniform"          — equal probability (default, identical to v3)
# "recency_weighted" — exponential weight toward recent days
# "recent_only"      — restrict pool to last SAMPLING_RECENT_WINDOW days
SAMPLING_MODE          = "uniform"
SAMPLING_WEIGHT_STR    = 3.0    # weight_strength for recency_weighted
SAMPLING_RECENT_WINDOW = 50     # lookback days for recent_only

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_C = {
    "ELITE":      "\033[95m",
    "STRONG":     "\033[92m",
    "ACCEPTABLE": "\033[93m",
    "MARGINAL":   "\033[33m",
    "REJECT":     "\033[91m",
    "RESET":      "\033[0m",
}


def _col(label): return _C.get(label, "") + label + _C["RESET"]


# ---------------------------------------------------------------------------
# Single-CSV simulation
# ---------------------------------------------------------------------------

def simulate_csv(csv_path: str, n_sims: int) -> dict:
    # Load empirical daily PnL distribution in dollars (v3 engine)
    daily_pnl = load_daily_pnl(csv_path)

    # --- run_until_payout (stop at first payout, up to MAX_DAYS_UTP days) ---
    utp = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = daily_pnl,
        max_days        = MAX_DAYS_UTP,
        stop_at_payout  = True,
        n_paths         = 0,   # no equity paths needed in batch mode
        mode            = SAMPLING_MODE,
        weight_strength = SAMPLING_WEIGHT_STR,
        recent_window   = SAMPLING_RECENT_WINDOW,
    )
    utp_payout_p = utp["pass_rate"]
    utp_mean_pay = utp["average_payout"]
    utp_ev_net   = utp_payout_p * utp_mean_pay - (1 - utp_payout_p) * RESET_COST

    # --- run_full_period (full window, payout tracked but path continues) ---
    fp = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = daily_pnl,
        max_days        = MAX_DAYS_FP,
        stop_at_payout  = False,
        n_paths         = 0,
        mode            = SAMPLING_MODE,
        weight_strength = SAMPLING_WEIGHT_STR,
        recent_window   = SAMPLING_RECENT_WINDOW,
    )
    fp_payout_p  = fp["pass_rate"]
    fp_blow_only = sum(
        1 for o, p in zip(fp["outcomes"], fp["payouts"]) if o == "blow" and p == 0
    ) / n_sims
    fp_timeout_np = sum(
        1 for o, p in zip(fp["outcomes"], fp["payouts"]) if o == "timeout" and p == 0
    ) / n_sims
    fp_mean_pay  = fp["average_payout"]
    fp_mean_bal  = float(np.mean(fp["balances"]))
    fp_ev_net    = fp_payout_p * fp_mean_pay - fp["fail_rate"] * RESET_COST

    return {
        "csv":              os.path.basename(csv_path),
        # until-payout
        "utp_payout_p":    utp_payout_p,
        "utp_mean_pay":    utp_mean_pay,
        "utp_ev_net":      utp_ev_net,
        "utp_rating":      score(utp_payout_p, "until_payout")["label"],
        # full-period
        "fp_payout_p":     fp_payout_p,
        "fp_blow_only_p":  fp_blow_only,
        "fp_timeout_p":    fp_timeout_np,
        "fp_mean_pay":     fp_mean_pay,
        "fp_mean_bal":     fp_mean_bal,
        "fp_ev_net":       fp_ev_net,
        "fp_rating":       score(fp_payout_p, "full_period")["label"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        csv_files = [f for f in sys.argv[1:] if f.endswith(".csv")]
    else:
        csv_files = sorted(f for f in glob.glob("*.csv") if f != "batch_results.csv")

    if not csv_files:
        print("No CSV files found. Pass filenames as arguments or run from the folder containing CSVs.")
        sys.exit(1)

    print(f"\nBatch running {len(csv_files)} CSV(s) with {N_SIMS:,} sims each  [mode={SAMPLING_MODE}]...\n")

    rows = []
    for i, csv in enumerate(csv_files, 1):
        print(f"  [{i}/{len(csv_files)}] {csv} ...", end=" ", flush=True)
        try:
            r = simulate_csv(csv, N_SIMS)
            rows.append(r)
            print(f"UTP {r['utp_payout_p']:.1%} {r['utp_rating']}  |  FP {r['fp_payout_p']:.1%} {r['fp_rating']}  |  EV ${r['fp_ev_net']:,.0f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not rows:
        print("No results to display.")
        return

    # Sort by full-period net EV descending
    rows.sort(key=lambda x: x["fp_ev_net"], reverse=True)

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------

    w = 38
    sep = "=" * (w + 76)

    print(f"\n\n{sep}")
    print(f"  BATCH RESULTS — ranked by Full-Period Net EV  ({N_SIMS:,} sims, {MAX_DAYS_FP}d window, ${RESET_COST} reset)")
    print(sep)
    print(f"  {'CSV':<{w}} {'UTP P%':>7} {'UTP Rating':<12} {'FP P%':>6} {'FP Rating':<12} {'FP EV/cyc':>10} {'FP Mean Pay':>12}")
    print(f"  {'-'*w} {'-'*7} {'-'*12} {'-'*6} {'-'*12} {'-'*10} {'-'*12}")

    for r in rows:
        name = r["csv"]
        if len(name) > w:
            name = name[:w-3] + "..."
        utp_c  = _C.get(r["utp_rating"], "") + r["utp_rating"] + _C["RESET"]
        fp_c   = _C.get(r["fp_rating"],  "") + r["fp_rating"]  + _C["RESET"]
        padded_utp = r["utp_rating"].ljust(12)
        padded_fp  = r["fp_rating"].ljust(12)
        print(
            f"  {name:<{w}} "
            f"{r['utp_payout_p']:>7.2%} "
            f"{_C.get(r['utp_rating'],'')}{padded_utp}{_C['RESET']} "
            f"{r['fp_payout_p']:>6.2%} "
            f"{_C.get(r['fp_rating'],'')}{padded_fp}{_C['RESET']} "
            f"${r['fp_ev_net']:>9,.0f} "
            f"${r['fp_mean_pay']:>11,.0f}"
        )

    print(sep)

    # Detail block per strategy
    print("\n\nDETAIL PER STRATEGY\n")
    for r in rows:
        print(f"  {r['csv']}")
        print(f"    Run-Until-Payout : {r['utp_payout_p']:.2%} payout | mean ${r['utp_mean_pay']:,.0f} | net EV ${r['utp_ev_net']:,.0f} | {_col(r['utp_rating'])}")
        print(f"    Full-Period {MAX_DAYS_FP}d  : {r['fp_payout_p']:.2%} payout | {r['fp_blow_only_p']:.2%} blow | {r['fp_timeout_p']:.2%} timeout | mean pay ${r['fp_mean_pay']:,.0f} | net EV ${r['fp_ev_net']:,.0f} | {_col(r['fp_rating'])}")
        print()

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------

    out_df = pd.DataFrame(rows)
    out_df.to_csv("batch_results.csv", index=False)
    print(f"  Saved: batch_results.csv")


if __name__ == "__main__":
    main()
