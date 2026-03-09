"""
run_full_period_v3_1.py
=======================
Monte Carlo simulation: simulate repeated account cycles over a fixed
21-trading-day window (one calendar month of trading).

Version 3.1 adds recency-weighted sampling.  All simulation mechanics,
output format, and chart logic are otherwise identical to v3.

Sampling modes
--------------
Set SAMPLING_MODE to one of:

  "uniform"          — equal probability for all historical days (v3 default)
  "recency_weighted" — recent days sampled more often via exponential weighting
                       governed by WEIGHT_STRENGTH (3 = moderate, 6 = strong)
  "recent_only"      — sample only from the last RECENT_WINDOW trading days

Tuning WEIGHT_STRENGTH (only applies to "recency_weighted")
-----------------------------------------------------------
  weight_i = exp(WEIGHT_STRENGTH × i/N)   then normalised to sum = 1

  WEIGHT_STRENGTH = 0  → uniform (same as mode="uniform")
  WEIGHT_STRENGTH = 3  → moderate bias toward recent performance
  WEIGHT_STRENGTH = 6  → strong bias; last few months dominate
  WEIGHT_STRENGTH = 10 → extreme; effectively only most-recent week matters

Workflow
--------
  1. Load CSV → empirical daily PnL distribution (chronological)
  2. Print statistical diagnostics
  3. Run Monte Carlo (stop_at_payout=False, max_days=21)
  4. Print outcome breakdown and long-run expectancy
  5. Print strategy tier rating
  6. Save equity-path chart and payout histogram

Defaults
--------
  N_SIMS          = 10,000
  RISK_MULTIPLIER = 1.0
  MAX_DAYS        = 21
  SAMPLING_MODE   = "uniform"
  WEIGHT_STRENGTH = 3.0
  RECENT_WINDOW   = 50

Usage
-----
  python run_full_period_v3_1.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from apex_engine_v3_1 import (
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
MAX_DAYS        = 21
N_PLOT_PATHS    = 60
RESET_COST      = 0.0   # override with actual Apex reset fee (e.g. 167.0)

# ── v3.1 sampling mode ───────────────────────────────────────────────────────
# Change SAMPLING_MODE to "recency_weighted" or "recent_only" to activate.
SAMPLING_MODE   = "uniform"     # "uniform" | "recency_weighted" | "recent_only"
WEIGHT_STRENGTH = 3.0           # exponential steepness (recency_weighted only)
RECENT_WINDOW   = 50            # lookback in trading days (recent_only only)


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
    label, desc = _tier_label(pass_rate)
    col   = _ANSI.get(label, "")
    reset = _ANSI["RESET"]

    print()
    print(f"  {'='*54}")
    print(f"  Strategy Rating  [Full-Period {MAX_DAYS}d]")
    print(f"  {'='*54}")

    prev_hi = 1.01
    for threshold, tlabel, _ in _TIERS:
        hi_str = f"{prev_hi*100:.0f}%" if prev_hi < 1.01 else "  -- "
        lo_str = f"{threshold*100:.0f}%"
        marker = "  <<< YOU ARE HERE" if tlabel == label else ""
        tc     = _ANSI.get(tlabel, "")
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

    payout_count     = sum(1 for p in payouts if p > 0)
    blow_count       = outcomes.count("blow")
    timeout_count    = outcomes.count("timeout")
    payout_then_blew = sum(1 for o, p in zip(outcomes, payouts) if o == "blow" and p > 0)
    payout_survived  = sum(1 for o, p in zip(outcomes, payouts) if o != "blow" and p > 0)
    blow_no_payout   = blow_count - payout_then_blew
    timeout_no_pay   = sum(1 for o, p in zip(outcomes, payouts) if o == "timeout" and p == 0)

    pass_rate    = results["pass_rate"]
    fail_rate    = results["fail_rate"]
    avg_balance  = float(np.mean(balances))
    avg_payout   = results["average_payout"]

    winning_payouts = [p for p in payouts if p > 0]

    blow_balances = [b for o, b in zip(outcomes, balances) if o == "blow"]
    avg_blow_loss = float(np.mean([ACCOUNT_SIZE - b for b in blow_balances])) if blow_balances else 0.0

    e_monthly = pass_rate * avg_payout - fail_rate * avg_blow_loss - RESET_COST

    w = 54
    print()
    print("═" * w)
    print(f"  OUTCOME BREAKDOWN  ({n_sims:,} simulations, {MAX_DAYS}-day window)")
    print("═" * w)
    print(f"  Sampling mode              : {SAMPLING_MODE}")
    if SAMPLING_MODE == "recency_weighted":
        print(f"  Weight strength            : {WEIGHT_STRENGTH:.1f}")
    elif SAMPLING_MODE == "recent_only":
        print(f"  Recent window              : {RECENT_WINDOW} days")
    print()
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
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_payout_histogram(results: dict, save_path: str = "payout_distribution_v3_1.png"):
    payouts = [p for p in results["payouts"] if p > 0]
    if not payouts:
        print("  [Chart] No payouts to plot in histogram.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(payouts, bins=30, color="#2ca02c", edgecolor="white", linewidth=0.5)
    ax.axvline(float(np.mean(payouts)), color="#d62728", linestyle="--",
               linewidth=1.2, label=f"Mean ${np.mean(payouts):,.2f}")
    ax.set_title(f"Payout Distribution — Full-Period {MAX_DAYS}d (v3.1 · {SAMPLING_MODE})", fontsize=12)
    ax.set_xlabel("Payout Amount ($)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Chart saved → {save_path}")
    plt.close()


def plot_equity_paths(results: dict, save_path: str = "monte_carlo_full_period_v3_1.png"):
    paths    = results["equity_paths"]
    outcomes = results["outcomes"]

    color_map = {"payout": "#2ca02c", "blow": "#d62728", "timeout": "#aaaaaa"}

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axhspan(
        ACCOUNT_SIZE - TRAILING_DD, TRAIL_STOP_LEVEL,
        alpha=0.06, color="#d62728", label="Drawdown danger zone",
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

    mode_label = (
        f"mode={SAMPLING_MODE}"
        + (f", strength={WEIGHT_STRENGTH}" if SAMPLING_MODE == "recency_weighted" else "")
        + (f", window={RECENT_WINDOW}d"    if SAMPLING_MODE == "recent_only"      else "")
    )
    ax.set_title(
        f"Monte Carlo Equity Paths — Full {MAX_DAYS}-Day Period (v3.1 · {mode_label})",
        fontsize=12,
    )
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
# PURE COMPUTATION ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_full_period_analysis(
    csv_path: str,
    n_sims: int = N_SIMS,
    risk_multiplier: float = RISK_MULTIPLIER,
    max_days: int = MAX_DAYS,
    n_paths: int = N_PLOT_PATHS,
    reset_cost: float = RESET_COST,
    sampling_mode: str = SAMPLING_MODE,
    weight_strength: float = WEIGHT_STRENGTH,
    recent_window: int = RECENT_WINDOW,
) -> dict:
    """Full-period Monte Carlo analysis (stop_at_payout=False).

    Pure programmatic entry-point: no printing, no prompts, no chart rendering.
    All CLI output is handled by main().

    Returns
    -------
    dict with keys:
        metadata               – run parameters
        payout_prob            – fraction of sims that hit payout at any point
        payout_survived_prob   – payout + account survived to end
        payout_then_blew_prob  – payout + account later blew up
        blow_no_payout_prob    – blew up without ever paying out
        timeout_no_pay_prob    – ran to max_days without paying out
        mean_ending_balance    – mean account balance at sim end
        mean_blow_loss         – mean capital lost in blow-up sims
        mean_payout            – mean payout amount among payout sims
        median_payout          – median payout amount among payout sims
        e_monthly              – long-run monthly expectancy ($)
        tier_label             – str tier name (e.g. "ELITE")
        tier_description       – str tier description
        equity_paths           – list[list[float]] (chart-friendly)
        outcomes               – list[str] raw per-sim outcome labels
        payouts                – list[float] raw per-sim payout amounts
        balances               – list[float] raw per-sim ending balances
        _sim_results           – raw dict returned by run_simulations() for CLI helpers
    """
    daily_pnl = load_daily_pnl(csv_path)
    results = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = daily_pnl,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        stop_at_payout  = False,
        n_paths         = n_paths,
        mode            = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
    )

    outcomes = results["outcomes"]
    payouts  = results["payouts"]
    balances = results["balances"]

    payout_count     = sum(1 for p in payouts if p > 0)
    blow_count       = outcomes.count("blow")
    payout_then_blew = sum(1 for o, p in zip(outcomes, payouts) if o == "blow" and p > 0)
    payout_survived  = sum(1 for o, p in zip(outcomes, payouts) if o != "blow" and p > 0)
    blow_no_payout   = blow_count - payout_then_blew
    timeout_no_pay   = sum(1 for o, p in zip(outcomes, payouts) if o == "timeout" and p == 0)

    winning_payouts  = [p for p in payouts if p > 0]
    blow_balances    = [b for o, b in zip(outcomes, balances) if o == "blow"]
    avg_blow_loss    = float(np.mean([ACCOUNT_SIZE - b for b in blow_balances])) if blow_balances else 0.0

    pass_rate  = results["pass_rate"]
    fail_rate  = results["fail_rate"]
    e_monthly  = pass_rate * results["average_payout"] - fail_rate * avg_blow_loss - reset_cost

    tier_label, tier_desc = _tier_label(pass_rate)

    return {
        "metadata": {
            "csv_path":         csv_path,
            "n_sims":           n_sims,
            "risk_multiplier":  risk_multiplier,
            "max_days":         max_days,
            "reset_cost":       reset_cost,
            "sampling_mode":    sampling_mode,
            "weight_strength":  weight_strength,
            "recent_window":    recent_window,
        },
        "payout_prob":            pass_rate,
        "payout_survived_prob":   payout_survived / n_sims,
        "payout_then_blew_prob":  payout_then_blew / n_sims,
        "blow_no_payout_prob":    blow_no_payout / n_sims,
        "timeout_no_pay_prob":    timeout_no_pay / n_sims,
        "mean_ending_balance":    float(np.mean(balances)),
        "mean_blow_loss":         avg_blow_loss,
        "mean_payout":            float(np.mean(winning_payouts)) if winning_payouts else 0.0,
        "median_payout":          float(np.median(winning_payouts)) if winning_payouts else 0.0,
        "e_monthly":              e_monthly,
        "tier_label":             tier_label,
        "tier_description":       tier_desc,
        "equity_paths":           [list(p) for p in results["equity_paths"]],
        "outcomes":               outcomes,
        "payouts":                payouts,
        "balances":               balances,
        "_sim_results":           results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_path = input("\nEnter CSV filename (must be in this folder): ").strip()
    if not csv_path.endswith(".csv"):
        csv_path += ".csv"

    daily_pnl = load_daily_pnl(csv_path)
    compute_diagnostics(daily_pnl)

    print(
        f"  Running {N_SIMS:,} simulations "
        f"(mode={SAMPLING_MODE}, stop_at_payout=False, max_days={MAX_DAYS}) …"
    )
    result = run_full_period_analysis(
        csv_path        = csv_path,
        n_sims          = N_SIMS,
        risk_multiplier = RISK_MULTIPLIER,
        max_days        = MAX_DAYS,
        n_paths         = N_PLOT_PATHS,
        reset_cost      = RESET_COST,
        sampling_mode   = SAMPLING_MODE,
        weight_strength = WEIGHT_STRENGTH,
        recent_window   = RECENT_WINDOW,
    )

    print_results(result["_sim_results"], N_SIMS)
    print_tier_rating(result["payout_prob"])
    plot_equity_paths(result["_sim_results"])
    plot_payout_histogram(result["_sim_results"])


if __name__ == "__main__":
    main()
