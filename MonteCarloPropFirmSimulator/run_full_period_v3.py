"""
run_full_period_v3.py
=====================
Monte Carlo simulation: simulate repeated account cycles over a fixed
21-trading-day window (one calendar month of trading).

Purpose
-------
Answers the question: "Over a fixed 21-day window, what fraction of accounts
pay out, what fraction blow up, and what is the long-run monthly expectancy?"

Unlike run_until_payout_v3 (uncapped time), this script applies real calendar
pressure.  Payout is still tracked across the period, but the simulation does
NOT stop at first payout — every path runs the full 21 days so the long-run
distribution is visible.

Workflow
--------
  1. Load CSV → empirical daily PnL distribution
  2. Print statistical diagnostics
  3. Run Monte Carlo (stop_at_payout = False, max_days = 21)
  4. Print outcome breakdown and long-run expectancy
  5. Print strategy tier rating
  6. Save equity-path chart

Defaults
--------
  n_sims          = 10,000
  risk_multiplier = 1.0
  max_days        = 21

Usage
-----
  python run_full_period_v3.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive; writes to file
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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_SIMS          = 10_000
RISK_MULTIPLIER = 1.0
MAX_DAYS        = 21          # one trading month
N_PLOT_PATHS    = 60          # equity curves retained for charting

# Apex account reset cost ($ you spend to re-purchase after a blow)
RESET_COST      = 0.0         # override with actual cost if known (e.g. 167.0)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY TIER TABLE
# ─────────────────────────────────────────────────────────────────────────────

_TIERS = [
    (0.60, "ELITE",      "Majority of accounts pay out within the month. Scale aggressively."),
    (0.50, "STRONG",     "More than half resolve as payouts. Good monthly capital turn."),
    (0.40, "ACCEPTABLE", "Positive EV but meaningful timeout drag. Monitor closely."),
    (0.30, "MARGINAL",   "Too much capital sitting idle. Consider tightening the strategy."),
    (0.00, "REJECT",     "Poor monthly resolution. Capital tied up without return. Do not run."),
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
    """Print tier table for the full-period mode."""
    label, desc = _tier_label(pass_rate)
    col   = _ANSI.get(label, "")
    reset = _ANSI["RESET"]

    print()
    print(f"  {'='*54}")
    print(f"  Strategy Rating  [Full-Period {MAX_DAYS}d]")
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
    balances = results["balances"]
    days     = results["days"]

    # --- categorise outcomes -------------------------------------------------
    payout_count    = sum(1 for p in payouts if p > 0)
    blow_count      = outcomes.count("blow")
    timeout_count   = outcomes.count("timeout")

    # Payout that subsequently blew (payout > 0 AND final outcome == blow)
    payout_then_blew = sum(
        1 for o, p in zip(outcomes, payouts) if o == "blow" and p > 0
    )
    # Payout and survived entire window
    payout_survived = sum(
        1 for o, p in zip(outcomes, payouts) if o != "blow" and p > 0
    )
    # Blow with no prior payout
    blow_no_payout = blow_count - payout_then_blew
    # Timeout with no payout
    timeout_no_pay = sum(
        1 for o, p in zip(outcomes, payouts) if o == "timeout" and p == 0
    )

    pass_rate    = results["pass_rate"]
    fail_rate    = results["fail_rate"]
    timeout_rate = results["timeout_rate"]

    avg_balance = float(np.mean(balances))
    avg_payout  = results["average_payout"]

    winning_payouts = [p for p in payouts if p > 0]
    payout_days     = [d for o, d in zip(outcomes, days) if o == "payout"]

    # ── Long-run monthly expectancy ──────────────────────────────────────────
    # Expected monthly PnL per account-cycle (before reset cost):
    #   E[monthly] = P(payout) × mean_payout  −  P(blow) × ACCOUNT_SIZE
    # This is simplified: it ignores partial balances you get back on timeout.
    # A more conservative estimate uses only loss-on-blow net of ending balance.
    blow_balances   = [b for o, b in zip(outcomes, balances) if o == "blow"]
    avg_blow_loss   = float(np.mean([ACCOUNT_SIZE - b for b in blow_balances])) if blow_balances else 0.0

    # Full expectancy: weighted payouts minus weighted losses minus reset cost
    e_monthly = (
        pass_rate * avg_payout
        - results["fail_rate"] * avg_blow_loss
        - RESET_COST
    )

    w = 54
    print()
    print("═" * w)
    print(f"  OUTCOME BREAKDOWN  ({n_sims:,} simulations, {MAX_DAYS}-day window)")
    print("═" * w)
    print(f"  Payout (any point)         : {pass_rate:>7.2%}   ({payout_count:>6,} sims)")
    print(f"    ↳ Payout + survived       : {payout_survived / n_sims:>7.2%}   ({payout_survived:>6,} sims)")
    print(f"    ↳ Payout + later blew     : {payout_then_blew / n_sims:>7.2%}   ({payout_then_blew:>6,} sims)")
    print(f"  Blow only (no payout)       : {blow_no_payout / n_sims:>7.2%}   ({blow_no_payout:>6,} sims)")
    print(f"  Timeout  (no payout)        : {timeout_no_pay / n_sims:>7.2%}   ({timeout_no_pay:>6,} sims)")
    print(f"  {'─'*w}")
    print(f"  Total                       : 100.00%   ({n_sims:>6,} sims)")
    print("═" * w)
    print()
    print("  ── Payout Distribution ───────────────────────────")
    if winning_payouts:
        arr = np.array(winning_payouts)
        print(f"  Count                  : {len(arr):>10,}")
        print(f"  Mean payout            : ${np.mean(arr):>10,.2f}")
        print(f"  Median payout          : ${np.median(arr):>10,.2f}")
        print(f"  Min payout             : ${np.min(arr):>10,.2f}")
        print(f"  Max payout             : ${np.max(arr):>10,.2f}")
    else:
        print(f"  No payouts recorded.")
    print()
    print("  ── Ending Balance ────────────────────────────────")
    print(f"  Mean ending balance    : ${avg_balance:>10,.2f}")
    print(f"  Mean blow-up loss      : ${avg_blow_loss:>10,.2f}")
    print()
    print("  ── Long-Run Monthly Expectancy ───────────────────")
    print(f"  E[monthly PnL]         : ${e_monthly:>10,.2f}")
    if RESET_COST > 0:
        print(f"  (includes reset cost of ${RESET_COST:,.2f} per cycle)")
    else:
        print(f"  (RESET_COST = $0; set it in the script if applicable)")
    print("═" * w)


# ─────────────────────────────────────────────────────────────────────────────
# PAYOUT DISTRIBUTION HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

def plot_payout_histogram(results: dict, save_path: str = "payout_distribution_v3.png"):
    """Bar chart of payout amounts (winning paths only)."""
    payouts = [p for p in results["payouts"] if p > 0]
    if not payouts:
        print("  [Chart] No payouts to plot in histogram.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(payouts, bins=30, color="#2ca02c", edgecolor="white", linewidth=0.5)
    ax.axvline(float(np.mean(payouts)), color="#d62728", linestyle="--",
               linewidth=1.2, label=f"Mean ${np.mean(payouts):,.2f}")
    ax.set_title(f"Payout Distribution — Full-Period {MAX_DAYS}d (v3)", fontsize=12)
    ax.set_xlabel("Payout Amount ($)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY PATH CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_paths(results: dict, save_path: str = "monte_carlo_full_period_v3.png"):
    """
    Plot retained equity paths, colour-coded by outcome.
    Shades the drawdown danger zone between the initial floor and ACCOUNT_SIZE.
    """
    paths    = results["equity_paths"]
    outcomes = results["outcomes"]

    color_map = {"payout": "#2ca02c", "blow": "#d62728", "timeout": "#aaaaaa"}

    fig, ax = plt.subplots(figsize=(12, 7))

    # Shade the danger zone
    ax.axhspan(
        ACCOUNT_SIZE - TRAILING_DD,
        TRAIL_STOP_LEVEL,
        alpha=0.06,
        color="#d62728",
        label="Drawdown danger zone",
    )

    for i, path in enumerate(paths):
        outcome = outcomes[i] if i < len(outcomes) else "timeout"
        ax.plot(path, color=color_map.get(outcome, "#aaaaaa"), alpha=0.25, linewidth=0.9)

    ax.axhline(PAYOUT_THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.3,
               label=f"Payout threshold  ${PAYOUT_THRESHOLD:,.0f}")
    ax.axhline(TRAIL_STOP_LEVEL, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Trailing stop lock ${TRAIL_STOP_LEVEL:,.0f}")
    ax.axhline(ACCOUNT_SIZE, color="#1f77b4", linestyle=":", linewidth=0.9,
               label=f"Start balance     ${ACCOUNT_SIZE:,.0f}")

    patches = [
        mpatches.Patch(color=color_map["payout"],  label="Payout"),
        mpatches.Patch(color=color_map["blow"],    label="Blow"),
        mpatches.Patch(color=color_map["timeout"], label="Timeout"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, loc="upper left", fontsize=8)

    ax.set_title(f"Monte Carlo Equity Paths — Full {MAX_DAYS}-Day Period (v3)", fontsize=13)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Account Balance ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(0, MAX_DAYS + 1)
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
    print(f"  Running {N_SIMS:,} simulations (stop_at_payout=False, max_days={MAX_DAYS}) …")
    results = run_simulations(
        n_sims          = N_SIMS,
        daily_pnl       = daily_pnl,
        risk_multiplier = RISK_MULTIPLIER,
        max_days        = MAX_DAYS,
        stop_at_payout  = False,
        n_paths         = N_PLOT_PATHS,
    )

    # ── 4. Print results ─────────────────────────────────────────────────────
    print_results(results, N_SIMS)
    print_tier_rating(results["pass_rate"])

    # ── 5. Charts ────────────────────────────────────────────────────────────
    plot_equity_paths(results)
    plot_payout_histogram(results)


if __name__ == "__main__":
    main()
