"""
run_until_payout_v3.py
======================
Monte Carlo simulation: run each account path until payout or blow-up,
with no artificial time cap (max_days = 90 as a safety ceiling).

Purpose
-------
Answers the question: "Given unlimited time, what fraction of accounts
pay out, and how many trading days does it typically take?"

This is the best-case upper-bound probability — the strategy edge with no
calendar pressure.

Workflow
--------
  1. Load CSV → empirical daily PnL distribution
  2. Print statistical diagnostics
  3. Run Monte Carlo (stop_at_payout = True)
  4. Print results and strategy tier rating
  5. Save equity-path chart

Defaults
--------
  n_sims          = 10,000
  risk_multiplier = 1.0
  max_days        = 90

Usage
-----
  python run_until_payout_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive; saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from apex_engine_v3 import (
    ACCOUNT_SIZE,
    TRAILING_DD,
    TRAIL_STOP_LEVEL,
    PAYOUT_THRESHOLD,
    load_daily_pnl,
    compute_diagnostics,
    run_simulations,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit here or make interactive as needed)
# ─────────────────────────────────────────────────────────────────────────────

N_SIMS          = 10_000
RISK_MULTIPLIER = 1.0
MAX_DAYS        = 90          # hard ceiling; not a target — just stops infinite loops
N_PLOT_PATHS    = 60          # equity curves kept for the chart


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY TIER TABLE
# ─────────────────────────────────────────────────────────────────────────────

# (min_probability, label, description)
_TIERS = [
    (0.75, "ELITE",      "Top-tier edge. Very high payout rate with no time cap."),
    (0.65, "STRONG",     "Solid strategy. Reliable edge, worth scaling."),
    (0.55, "ACCEPTABLE", "Positive EV. Verify mean payout justifies reset risk."),
    (0.45, "MARGINAL",   "Weak edge. Only viable with an unusually large mean payout."),
    (0.00, "REJECT",     "Near-zero or negative EV after reset costs. Do not run."),
]

_ANSI = {
    "ELITE":      "\033[95m",
    "STRONG":     "\033[92m",
    "ACCEPTABLE": "\033[93m",
    "MARGINAL":   "\033[33m",
    "REJECT":     "\033[91m",
    "RESET":      "\033[0m",
}


def _tier_label(pass_rate: float) -> tuple[str, str]:
    for threshold, label, desc in _TIERS:
        if pass_rate >= threshold:
            return label, desc
    return "REJECT", _TIERS[-1][2]


def print_tier_rating(pass_rate: float):
    """Print full tier table, highlighting where this strategy lands."""
    label, desc = _tier_label(pass_rate)
    col   = _ANSI.get(label, "")
    reset = _ANSI["RESET"]

    print()
    print(f"  {'='*54}")
    print(f"  Strategy Rating  [Run-Until-Payout]")
    print(f"  {'='*54}")

    prev_hi = 1.01
    for threshold, tlabel, _ in _TIERS:
        hi_str  = f"{prev_hi*100:.0f}%" if prev_hi < 1.01 else "  -- "
        lo_str  = f"{threshold*100:.0f}%"
        rng     = f"{lo_str}–{hi_str}" if prev_hi < 1.01 else f">= {lo_str}"
        marker  = "  <<< YOU ARE HERE" if tlabel == label else ""
        tc      = _ANSI.get(tlabel, "")
        print(f"  {tc}{tlabel:<12}{reset}  P(payout) >= {lo_str:<5} {marker}")
        prev_hi = threshold

    print(f"  {'─'*54}")
    print(f"  This strategy:  P(payout) = {pass_rate:.2%}")
    print(f"  Rating:  {col}{label}{reset}  —  {desc}")
    print(f"  {'='*54}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict, n_sims: int):
    outcomes = results["outcomes"]
    payouts  = results["payouts"]
    days     = results["days"]

    payout_count = sum(1 for p in payouts if p > 0)
    blow_count   = outcomes.count("blow")
    timeout_count = outcomes.count("timeout")

    pass_rate    = results["pass_rate"]
    fail_rate    = results["fail_rate"]
    timeout_rate = results["timeout_rate"]

    payout_days  = [d for o, d in zip(outcomes, days) if o == "payout"]
    avg_days_all = results["average_days"]
    avg_days_pay = float(np.mean(payout_days)) if payout_days else 0.0
    med_days_pay = int(np.median(payout_days)) if payout_days else 0

    avg_payout   = results["average_payout"]

    w = 54
    print()
    print("═" * w)
    print("  MONTE CARLO RESULTS  [Run-Until-Payout]")
    print("═" * w)
    print(f"  Simulations          : {n_sims:>10,}")
    print(f"  Risk multiplier      : {RISK_MULTIPLIER:>10.2f}×")
    print(f"  Max days (ceiling)   : {MAX_DAYS:>10}")
    print()
    print(f"  Payout probability   : {pass_rate:>9.2%}   ({payout_count:>7,} sims)")
    print(f"  Blow probability     : {fail_rate:>9.2%}   ({blow_count:>7,} sims)")
    print(f"  Timeout probability  : {timeout_rate:>9.2%}   ({timeout_count:>7,} sims)")
    print()
    print(f"  Mean days (all sims) : {avg_days_all:>10.1f}")
    print(f"  Mean days → payout   : {avg_days_pay:>10.1f}")
    print(f"  Median days → payout : {med_days_pay:>10}")
    print()
    print(f"  Avg payout (if paid) : ${avg_payout:>10,.2f}")
    print("═" * w)


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_paths(results: dict, save_path: str = "monte_carlo_until_payout_v3.png"):
    """
    Plot the retained equity paths with key account levels annotated.
    Colours paths by outcome: green = payout, red = blow, grey = timeout.
    """
    paths    = results["equity_paths"]
    outcomes = results["outcomes"]

    color_map = {"payout": "#2ca02c", "blow": "#d62728", "timeout": "#aaaaaa"}

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, path in enumerate(paths):
        outcome = outcomes[i] if i < len(outcomes) else "timeout"
        ax.plot(path, color=color_map.get(outcome, "#aaaaaa"), alpha=0.25, linewidth=0.8)

    # Horizontal reference levels
    ax.axhline(PAYOUT_THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"Payout threshold  ${PAYOUT_THRESHOLD:,.0f}")
    ax.axhline(TRAIL_STOP_LEVEL, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"Trailing stop lock ${TRAIL_STOP_LEVEL:,.0f}")
    ax.axhline(ACCOUNT_SIZE - TRAILING_DD, color="#ff7f0e", linestyle=":", linewidth=1.0,
               label=f"Initial floor      ${ACCOUNT_SIZE - TRAILING_DD:,.0f}")

    # Legend patches for outcome colours
    patches = [
        mpatches.Patch(color=color_map["payout"],  label="Payout"),
        mpatches.Patch(color=color_map["blow"],    label="Blow"),
        mpatches.Patch(color=color_map["timeout"], label="Timeout"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, loc="upper left", fontsize=8)

    ax.set_title("Monte Carlo Equity Paths — Run Until Payout (v3)", fontsize=13)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Account Balance ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    csv_path = input("\nEnter CSV filename (must be in this folder): ").strip()
    if not csv_path.endswith(".csv"):
        csv_path += ".csv"

    daily_pnl = load_daily_pnl(csv_path)

    # ── 2. Diagnostics ───────────────────────────────────────────────────────
    compute_diagnostics(daily_pnl)

    # ── 3. Monte Carlo ───────────────────────────────────────────────────────
    print(f"  Running {N_SIMS:,} simulations (stop_at_payout=True, max_days={MAX_DAYS}) …")
    results = run_simulations(
        n_sims          = N_SIMS,
        daily_pnl       = daily_pnl,
        risk_multiplier = RISK_MULTIPLIER,
        max_days        = MAX_DAYS,
        stop_at_payout  = True,
        n_paths         = N_PLOT_PATHS,
    )

    # ── 4. Print results ─────────────────────────────────────────────────────
    print_results(results, N_SIMS)
    print_tier_rating(results["pass_rate"])

    # ── 5. Chart ─────────────────────────────────────────────────────────────
    plot_equity_paths(results)


if __name__ == "__main__":
    main()
