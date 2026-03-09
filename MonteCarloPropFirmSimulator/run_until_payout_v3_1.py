"""
run_until_payout_v3_1.py
========================
Monte Carlo simulation: run each account path until payout or blow-up,
with no artificial time cap (max_days = 90 as a safety ceiling).

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
  3. Run Monte Carlo (stop_at_payout=True) with selected sampling mode
  4. Print results and strategy tier rating
  5. Save equity-path chart

Defaults
--------
  N_SIMS          = 10,000
  RISK_MULTIPLIER = 1.0
  MAX_DAYS        = 90
  SAMPLING_MODE   = "uniform"
  WEIGHT_STRENGTH = 3.0
  RECENT_WINDOW   = 50

Usage
-----
  python run_until_payout_v3_1.py
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
MAX_DAYS        = 90
N_PLOT_PATHS    = 60

# ── v3.1 sampling mode ───────────────────────────────────────────────────────
# Change SAMPLING_MODE to "recency_weighted" or "recent_only" to activate.
SAMPLING_MODE   = "uniform"     # "uniform" | "recency_weighted" | "recent_only"
WEIGHT_STRENGTH = 3.0           # exponential steepness (recency_weighted only)
RECENT_WINDOW   = 50            # lookback in trading days (recent_only only)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY TIER TABLE
# ─────────────────────────────────────────────────────────────────────────────

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
    label, desc = _tier_label(pass_rate)
    col   = _ANSI.get(label, "")
    reset = _ANSI["RESET"]

    print()
    print(f"  {'='*54}")
    print(f"  Strategy Rating  [Run-Until-Payout]")
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
    days     = results["days"]

    payout_count  = sum(1 for p in payouts if p > 0)
    blow_count    = outcomes.count("blow")
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
    print(f"  Sampling mode        : {SAMPLING_MODE:>10}")
    if SAMPLING_MODE == "recency_weighted":
        print(f"  Weight strength      : {WEIGHT_STRENGTH:>10.1f}")
    elif SAMPLING_MODE == "recent_only":
        print(f"  Recent window        : {RECENT_WINDOW:>10} days")
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

def plot_equity_paths(results: dict, save_path: str = "monte_carlo_until_payout_v3_1.png"):
    paths    = results["equity_paths"]
    outcomes = results["outcomes"]

    color_map = {"payout": "#2ca02c", "blow": "#d62728", "timeout": "#aaaaaa"}

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, path in enumerate(paths):
        outcome = outcomes[i] if i < len(outcomes) else "timeout"
        ax.plot(path, color=color_map.get(outcome, "#aaaaaa"), alpha=0.25, linewidth=0.8)

    ax.axhline(PAYOUT_THRESHOLD, color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"Payout threshold  ${PAYOUT_THRESHOLD:,.0f}")
    ax.axhline(TRAIL_STOP_LEVEL, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"Trailing stop lock ${TRAIL_STOP_LEVEL:,.0f}")
    ax.axhline(ACCOUNT_SIZE - TRAILING_DD, color="#ff7f0e", linestyle=":", linewidth=1.0,
               label=f"Initial floor      ${ACCOUNT_SIZE - TRAILING_DD:,.0f}")

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
    ax.set_title(f"Monte Carlo Equity Paths — Run Until Payout (v3.1 · {mode_label})", fontsize=12)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Account Balance ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PURE COMPUTATION ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_until_payout_analysis(
    csv_path: str,
    n_sims: int = N_SIMS,
    risk_multiplier: float = RISK_MULTIPLIER,
    max_days: int = MAX_DAYS,
    n_paths: int = N_PLOT_PATHS,
    sampling_mode: str = SAMPLING_MODE,
    weight_strength: float = WEIGHT_STRENGTH,
    recent_window: int = RECENT_WINDOW,
) -> dict:
    """Run-until-payout Monte Carlo analysis.

    Pure programmatic entry-point: no printing, no prompts, no chart rendering.
    All CLI output is handled by main().

    Returns
    -------
    dict with keys:
        metadata               – run parameters
        payout_prob            – fraction of sims that hit the payout threshold
        blow_prob              – fraction of sims that blew up
        timeout_prob           – fraction of sims that ran to max_days
        mean_days_all          – mean days across all sims
        mean_days_to_payout    – mean days among payout sims (0.0 if none)
        median_days_to_payout  – median days among payout sims (0 if none)
        mean_payout            – mean payout amount among payout sims
        tier_label             – str tier name (e.g. "ELITE")
        tier_description       – str tier description
        equity_paths           – list[list[float]] (chart-friendly)
        outcomes               – list[str] raw per-sim outcome labels
        payouts                – list[float] raw per-sim payout amounts
        days                   – list[int] raw per-sim day counts
        _sim_results           – raw dict returned by run_simulations() for CLI helpers
    """
    daily_pnl = load_daily_pnl(csv_path)
    results = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = daily_pnl,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        stop_at_payout  = True,
        n_paths         = n_paths,
        mode            = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
    )

    outcomes    = results["outcomes"]
    payouts     = results["payouts"]
    payout_days = [d for o, d in zip(outcomes, results["days"]) if o == "payout"]
    tier_label, tier_desc = _tier_label(results["pass_rate"])

    return {
        "metadata": {
            "csv_path":         csv_path,
            "n_sims":           n_sims,
            "risk_multiplier":  risk_multiplier,
            "max_days":         max_days,
            "sampling_mode":    sampling_mode,
            "weight_strength":  weight_strength,
            "recent_window":    recent_window,
        },
        "payout_prob":            results["pass_rate"],
        "blow_prob":              results["fail_rate"],
        "timeout_prob":           results["timeout_rate"],
        "mean_days_all":          float(results["average_days"]),
        "mean_days_to_payout":    float(np.mean(payout_days)) if payout_days else 0.0,
        "median_days_to_payout":  int(np.median(payout_days)) if payout_days else 0,
        "mean_payout":            float(results["average_payout"]),
        "tier_label":             tier_label,
        "tier_description":       tier_desc,
        "equity_paths":           [list(p) for p in results["equity_paths"]],
        "outcomes":               outcomes,
        "payouts":                payouts,
        "days":                   results["days"],
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
        f"(mode={SAMPLING_MODE}, stop_at_payout=True, max_days={MAX_DAYS}) …"
    )
    result = run_until_payout_analysis(
        csv_path        = csv_path,
        n_sims          = N_SIMS,
        risk_multiplier = RISK_MULTIPLIER,
        max_days        = MAX_DAYS,
        n_paths         = N_PLOT_PATHS,
        sampling_mode   = SAMPLING_MODE,
        weight_strength = WEIGHT_STRENGTH,
        recent_window   = RECENT_WINDOW,
    )

    print_results(result["_sim_results"], N_SIMS)
    print_tier_rating(result["payout_prob"])
    plot_equity_paths(result["_sim_results"])


if __name__ == "__main__":
    main()
