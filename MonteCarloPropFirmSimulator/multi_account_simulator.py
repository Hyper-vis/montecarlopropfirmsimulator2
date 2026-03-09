"""
multi_account_simulator.py
==========================
Monte Carlo simulation across a portfolio of Apex accounts.

All simulation logic, output format, and chart logic are identical to the
original module-level script.  This version separates concerns so the core
computation can be called programmatically (no stdin prompts, no printing,
no matplotlib side-effects) via run_multi_account_analysis().

Public API
----------
  run_multi_account_analysis(csv_path, n_accounts, ...)   -> dict
      Pure computation function.  Safe to import and call from any context.

  main()
      Interactive CLI entry-point — reproduces the original user experience.

Internal helpers (used by main() only)
---------------------------------------
  print_portfolio_results(result)
  print_cycle_analysis(result)
  plot_payout_histogram(result, save_path)
  plot_equity_paths(result, save_path)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from apex_engine_v3_1 import (
    load_daily_pnl,
    simulate_path,
    ACCOUNT_SIZE,
    TRAILING_DD,
    TRAIL_STOP_LEVEL,
    DAILY_LOSS_LIMIT,
    DAILY_PROFIT_CAP,
    PAYOUT_THRESHOLD,
    MAX_PAYOUT,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS  (identical to original; used by CLI output functions)
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_MONTH = 21   # ~6 days/week × ~3.5 weeks
CYCLE_GATE             = 8    # Apex minimum trading days before payout


# ─────────────────────────────────────────────────────────────────────────────
# INPUT HELPERS  (used by main() only)
# ─────────────────────────────────────────────────────────────────────────────

def ask_int(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a whole number.")


def ask_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a numeric value.")


def ask_yes_no(prompt):
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("yes", "y"):
            return True
        if raw in ("no", "n"):
            return False
        print("  Please answer yes or no.")


# ─────────────────────────────────────────────────────────────────────────────
# PURE COMPUTATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_account_analysis(
    csv_path: str,
    n_accounts: int,
    n_sims: int = 10_000,
    max_days: int = 30,
    stop_at_payout: bool = True,
    risk_multiplier: float = 1.0,
    sampling_mode: str = "uniform",
    weight_strength: float = 3.0,
    recent_window: int = 50,
    n_path_sims: int = 150,
) -> dict:
    """Run multi-account Monte Carlo analysis.

    Pure programmatic entry-point: no printing, no prompts, no chart rendering.
    All CLI output is handled by main().

    Parameters
    ----------
    csv_path        : path to CSV of daily PnL trades
    n_accounts      : number of parallel Apex accounts in the portfolio
    n_sims          : number of Monte Carlo simulations
    max_days        : maximum trading days per account cycle
    stop_at_payout  : if True, each account stops when it hits a payout
    risk_multiplier : scale all daily PnL by this factor
    sampling_mode   : "uniform" | "recency_weighted" | "recent_only"
    weight_strength : exponential steepness (recency_weighted only)
    recent_window   : look-back in trading days (recent_only only)
    n_path_sims     : number of simulation runs from which to collect equity
                      paths (for chart generation by caller)

    Returns
    -------
    dict with keys:
        metadata               -- run parameters
        portfolio              -- aggregate payout / balance statistics
        cycle_efficiency       -- days-to-payout and monthly throughput metrics
                                  (empty dict if no payouts were recorded)
        _portfolio_payouts     -- list[float], per-sim total payout (for charts)
        _account_paths         -- list of per-account equity path samples;
                                  each element is a list of (path, outcome) tuples
                                  collected from the first n_path_sims simulations.
                                  NOT JSON-safe -- for internal / chart use only.
    """
    daily_pnl = load_daily_pnl(csv_path)

    # ── Simulation arrays (identical to original) ─────────────────────────────
    portfolio_payouts    = np.zeros(n_sims)
    portfolio_blowouts   = np.zeros(n_sims, dtype=int)
    account_end_balances = np.zeros((n_sims, n_accounts))
    account_payout_days  = np.full((n_sims, n_accounts), np.nan)
    account_paths        = [[] for _ in range(n_accounts)]

    # ── Simulation loop (identical to original) ───────────────────────────────
    for sim in range(n_sims):
        total_payout = 0.0
        blown_count  = 0

        for acct_idx in range(n_accounts):
            result = simulate_path(
                daily_pnl,
                risk_multiplier = risk_multiplier,
                max_days        = max_days,
                stop_at_payout  = stop_at_payout,
                mode            = sampling_mode,
                weight_strength = weight_strength,
                recent_window   = recent_window,
            )
            total_payout += result["payout"]
            account_end_balances[sim, acct_idx] = result["balance"]
            if result["outcome"] == "blow":
                blown_count += 1
            if result["outcome"] == "payout":
                account_payout_days[sim, acct_idx] = result["days"]
            if sim < n_path_sims:
                account_paths[acct_idx].append((result["equity_path"], result["outcome"]))

        portfolio_payouts[sim]  = total_payout
        portfolio_blowouts[sim] = blown_count

    # ── Aggregate balance metrics ─────────────────────────────────────────────
    mean_end_per_acct  = account_end_balances.mean(axis=0)   # shape (n_accounts,)
    mean_end_portfolio = float(mean_end_per_acct.sum())
    mean_end_single    = float(mean_end_per_acct.mean())

    # ── Cycle efficiency metrics (identical to original) ─────────────────────
    all_payout_days = account_payout_days.flatten()
    all_payout_days = all_payout_days[~np.isnan(all_payout_days)]

    cycle: dict = {}
    if len(all_payout_days) > 0:
        payout_prob_single  = len(all_payout_days) / (n_sims * n_accounts)
        mean_days_payout    = float(all_payout_days.mean())
        median_days_payout  = float(np.median(all_payout_days))
        p5_days             = float(np.percentile(all_payout_days, 5))
        p95_days            = float(np.percentile(all_payout_days, 95))
        pct_within_gate     = float((all_payout_days <= CYCLE_GATE).mean())
        pct_within_10       = float((all_payout_days <= 10).mean())
        pct_within_15       = float((all_payout_days <= 15).mean())
        est_cycles          = TRADING_DAYS_PER_MONTH / mean_days_payout

        # ── Payout-per-account estimate (identical to original calculation) ────
        # The original code computes est_monthly_gross twice and the second
        # assignment (using mean_payout_per_acct_hit) is the one that takes
        # effect.  Replicated verbatim — do NOT simplify.
        mean_payout_per_acct_hit = (
            float(np.mean(portfolio_payouts)) / (n_accounts * payout_prob_single)
            if payout_prob_single > 0 else 0.0
        )
        est_monthly_gross = (
            n_accounts * est_cycles * payout_prob_single * mean_payout_per_acct_hit
        )
        est_monthly_net = (
            est_monthly_gross
            - (n_accounts * est_cycles * (1 - payout_prob_single) * 214.50)
        )

        cycle = {
            "payout_prob_per_cycle":  payout_prob_single,
            "mean_days_to_payout":    mean_days_payout,
            "median_days_to_payout":  median_days_payout,
            "p5_days":                p5_days,
            "p95_days":               p95_days,
            "pct_within_gate":        pct_within_gate,
            "pct_within_10":          pct_within_10,
            "pct_within_15":          pct_within_15,
            "est_cycles_per_month":   est_cycles,
            "est_monthly_gross":      est_monthly_gross,
            "est_monthly_net":        est_monthly_net,
        }

    return {
        "metadata": {
            "csv_path":         csv_path,
            "n_accounts":       n_accounts,
            "n_sims":           n_sims,
            "max_days":         max_days,
            "stop_at_payout":   stop_at_payout,
            "risk_multiplier":  risk_multiplier,
            "sampling_mode":    sampling_mode,
            "weight_strength":  weight_strength,
            "recent_window":    recent_window,
            "n_path_sims":      n_path_sims,
        },
        "portfolio": {
            "mean_total_payout":          float(np.mean(portfolio_payouts)),
            "median_total_payout":        float(np.median(portfolio_payouts)),
            "p5_payout":                  float(np.percentile(portfolio_payouts, 5)),
            "p95_payout":                 float(np.percentile(portfolio_payouts, 95)),
            "prob_any_payout":            float((portfolio_payouts > 0).mean()),
            "mean_blown_accounts":        float(np.mean(portfolio_blowouts)),
            "mean_end_balance_single":    mean_end_single,
            "mean_end_balance_portfolio": mean_end_portfolio,
            "mean_end_balance_per_acct":  mean_end_per_acct.tolist(),
        },
        "cycle_efficiency":   cycle,
        "_portfolio_payouts": portfolio_payouts.tolist(),
        "_account_paths":     account_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT HELPERS  (used by main() only; not part of the programmatic API)
# ─────────────────────────────────────────────────────────────────────────────

def print_portfolio_results(result: dict) -> None:
    """Print the portfolio summary block (identical to original output)."""
    meta  = result["metadata"]
    port  = result["portfolio"]
    n_acc = meta["n_accounts"]
    n_sim = meta["n_sims"]
    mdays = meta["max_days"]

    print("=" * 52)
    print(f"  Portfolio size:            {n_acc} account(s)")
    print(f"  Window:                    {mdays} days (stop at payout)")
    print(f"  Monte Carlo simulations:   {n_sim:,}")
    print("-" * 52)
    print(f"  Mean total payout:         ${port['mean_total_payout']:>12,.2f}")
    print(f"  Median total payout:       ${port['median_total_payout']:>12,.2f}")
    print(f"  5th percentile payout:     ${port['p5_payout']:>12,.2f}")
    print(f"  95th percentile payout:    ${port['p95_payout']:>12,.2f}")
    print(f"  Prob total payout > 0:     {port['prob_any_payout']:>12.2%}")
    print(f"  Mean blown accounts:       {port['mean_blown_accounts']:>12.2f} / {n_acc}")
    print("-" * 52)
    print(f"  Mean ending bal (single):  ${port['mean_end_balance_single']:>12,.2f}")
    print(f"  Mean ending bal (total):   ${port['mean_end_balance_portfolio']:>12,.2f}")
    if n_acc > 1:
        for i, bal in enumerate(port["mean_end_balance_per_acct"]):
            print(f"    Account {i+1}:              ${bal:>12,.2f}")
    print("=" * 52)


def print_cycle_analysis(result: dict) -> None:
    """Print the cycle efficiency block (identical to original output)."""
    cycle = result["cycle_efficiency"]
    if not cycle:
        print("\n  No payouts recorded — cycle analysis unavailable.")
        return

    n_acc = result["metadata"]["n_accounts"]

    print(f"\n{'=' * 52}")
    print(f"  Cycle Efficiency  (Apex min gate = {CYCLE_GATE} trading days)")
    print(f"  Trading days/month assumed: {TRADING_DAYS_PER_MONTH}")
    print(f"{'=' * 52}")
    print(f"  Payout prob per cycle:     {cycle['payout_prob_per_cycle']:>12.2%}")
    print(f"  Mean days to payout:       {cycle['mean_days_to_payout']:>12.1f} trading days")
    print(f"  Median days to payout:     {cycle['median_days_to_payout']:>12.1f} trading days")
    print(f"  5th pct days to payout:    {cycle['p5_days']:>12.1f} trading days")
    print(f"  95th pct days to payout:   {cycle['p95_days']:>12.1f} trading days")
    print(f"{'-' * 52}")
    print(f"  Hit within {CYCLE_GATE} days (min gate): {cycle['pct_within_gate']:>10.2%}")
    print(f"  Hit within 10 days:        {cycle['pct_within_10']:>12.2%}")
    print(f"  Hit within 15 days:        {cycle['pct_within_15']:>12.2%}")
    print(f"{'-' * 52}")
    print(f"  Est. cycles/month/account: {cycle['est_cycles_per_month']:>12.2f}")
    print(f"  Est. monthly gross ({n_acc} accts): ${cycle['est_monthly_gross']:>11,.2f}")
    print(f"  Est. monthly net (fees):   ${cycle['est_monthly_net']:>11,.2f}")
    print(f"{'=' * 52}")


def plot_payout_histogram(
    result: dict,
    save_path: str = "multi_account_histogram.png",
) -> None:
    """Save portfolio payout distribution histogram (identical to original)."""
    portfolio_payouts = result["_portfolio_payouts"]
    meta  = result["metadata"]
    n_acc = meta["n_accounts"]
    n_sim = meta["n_sims"]

    mean_pp   = float(np.mean(portfolio_payouts))
    median_pp = float(np.median(portfolio_payouts))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_payouts, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(mean_pp,   color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean ${mean_pp:,.0f}")
    ax.axvline(median_pp, color="green",  linestyle="--", linewidth=1.5,
               label=f"Median ${median_pp:,.0f}")
    ax.set_title(
        f"Portfolio Payout Distribution — {n_acc} Account(s), {n_sim:,} Simulations",
        fontsize=13,
    )
    ax.set_xlabel("Total Portfolio Payout ($)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to: {save_path}")


def plot_equity_paths(
    result: dict,
    save_path: str = "multi_account_paths.png",
) -> None:
    """Save equity path grid (identical to original; up to 9 accounts shown)."""
    meta          = result["metadata"]
    account_paths = result["_account_paths"]
    n_accounts    = meta["n_accounts"]
    n_path_sims   = meta["n_path_sims"]

    OUTCOME_STYLE = {
        "payout":  {"color": "#2ecc71", "alpha": 0.30, "zorder": 2},
        "blow":    {"color": "#e74c3c", "alpha": 0.30, "zorder": 2},
        "timeout": {"color": "#95a5a6", "alpha": 0.20, "zorder": 1},
    }

    n_shown = min(n_accounts, 9)
    n_cols  = min(n_shown, 3)
    n_rows  = (n_shown + n_cols - 1) // n_cols
    fig_w   = max(6, n_cols * 4)
    fig_h   = max(4, n_rows * 3.2)

    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for plot_idx in range(n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        ax2 = axes[row][col]

        if plot_idx >= n_shown:
            ax2.set_visible(False)
            continue

        paths_for_acct = account_paths[plot_idx]
        outcome_counts = {k: 0 for k in OUTCOME_STYLE}

        for path, outcome in paths_for_acct:
            style = OUTCOME_STYLE.get(outcome, OUTCOME_STYLE["timeout"])
            ax2.plot(path, linewidth=0.7, **style)
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        ax2.axhline(PAYOUT_THRESHOLD,          color="#27ae60", linestyle="--", linewidth=0.9, alpha=0.8)
        ax2.axhline(ACCOUNT_SIZE,              color="#2980b9", linestyle="--", linewidth=0.9, alpha=0.8)
        ax2.axhline(ACCOUNT_SIZE - TRAILING_DD, color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.8)

        n_p = outcome_counts.get("payout", 0)
        n_b = outcome_counts.get("blow", 0)
        n_t = outcome_counts.get("timeout", 0)
        ax2.set_title(
            f"Account {plot_idx + 1}\n"
            f"Pay {n_p}  Blow {n_b}  Time {n_t}",
            fontsize=8, pad=3,
        )
        ax2.tick_params(labelsize=7)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax2.grid(alpha=0.2)

    legend_elements = [
        Line2D([0], [0], color="#2ecc71", linewidth=1.5, label="Payout"),
        Line2D([0], [0], color="#e74c3c", linewidth=1.5, label="Blow"),
        Line2D([0], [0], color="#95a5a6", linewidth=1.5, label="Timeout"),
        Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--", label="Payout threshold"),
        Line2D([0], [0], color="#2980b9", linewidth=1.2, linestyle="--", label="Account size"),
        Line2D([0], [0], color="#c0392b", linewidth=1.2, linestyle="--", label="Max draw floor"),
    ]
    fig2.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
                bbox_to_anchor=(0.5, 0), framealpha=0.9)

    if n_shown < n_accounts:
        fig2.suptitle(
            f"Equity Paths — First {n_shown} of {n_accounts} Accounts  "
            f"(sample of {n_path_sims} simulations each)",
            fontsize=11, y=1.01,
        )
    else:
        fig2.suptitle(
            f"Equity Paths — {n_accounts} Account(s)  "
            f"(sample of {n_path_sims} simulations each)",
            fontsize=11, y=1.01,
        )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Equity path grid saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  (interactive CLI — identical behavior to original)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Inputs ───────────────────────────────────────────────────────────────
    csv_path = input("Enter CSV filename (must be in this folder): ").strip()
    if not csv_path.endswith(".csv"):
        csv_path += ".csv"

    n_accounts = ask_int("Number of accounts: ")

    use_apex = ask_yes_no("Use default Apex settings? (yes/no): ")

    if use_apex:
        max_days        = 30
        stop_at_payout  = True
        risk_multiplier = 1.0
        n_sims          = 10_000
        sampling_mode   = "uniform"
        weight_strength = 3.0
        recent_window   = 50
    else:
        print()
        print("  NOTE: Account geometry (size, drawdown limits, thresholds) is fixed")
        print("        by Apex constants in apex_engine_v3_1. Configurable below:")
        print()
        max_days        = ask_int("Max days per account cycle: ", default=30)
        stop_at_payout  = ask_yes_no("Stop each account at first payout? (yes/no): ")
        risk_multiplier = ask_float("Risk multiplier (1.0 = no change): ", default=1.0)
        n_sims          = ask_int("Number of Monte Carlo simulations: ", default=10_000)
        print("  Sampling mode: uniform / recency_weighted / recent_only")
        sampling_mode   = input("  Sampling mode [uniform]: ").strip() or "uniform"
        weight_strength = 3.0
        recent_window   = 50
        if sampling_mode == "recency_weighted":
            weight_strength = ask_float("  Weight strength (3=moderate, 6=strong): ", default=3.0)
        elif sampling_mode == "recent_only":
            recent_window   = ask_int("  Recent window (trading days): ", default=50)

    # ── Run pure computation ──────────────────────────────────────────────────
    print(f"\nRunning {n_sims:,} simulations across {n_accounts} account(s)...\n")

    result = run_multi_account_analysis(
        csv_path        = csv_path,
        n_accounts      = n_accounts,
        n_sims          = n_sims,
        max_days        = max_days,
        stop_at_payout  = stop_at_payout,
        risk_multiplier = risk_multiplier,
        sampling_mode   = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
    )

    # ── Display results ───────────────────────────────────────────────────────
    print_portfolio_results(result)
    print_cycle_analysis(result)
    plot_payout_histogram(result)
    plot_equity_paths(result)


if __name__ == "__main__":
    main()
