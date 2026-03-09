"""
apex_engine_v3.py
=================
Version 3 — Daily-PnL Monte Carlo engine for Apex prop firm simulation.

Key change from v2: samples are drawn from the empirical **daily PnL
distribution (USD dollars)** instead of percentage returns, eliminating the
artificial compounding that arose when multiplying a random return by a
balance that itself drifted over time.

All Apex rule mechanics are preserved exactly:
  - Trailing drawdown with lock-out at TRAIL_STOP_LEVEL
  - Daily profit cap and daily loss limit
  - Minimum trading days (MIN_DAYS = 8)
  - Green-day count requirement (MIN_GREEN_DAYS = 5, GREEN_DAY_MIN = $50)
  - 30 % single-day concentration rule
  - Payout floor and MAX_PAYOUT cap

Public API
----------
  load_daily_pnl(csv_path)          → np.ndarray of daily PnL in USD
  compute_diagnostics(daily_pnl)    → diagnostic dict (also prints report)
  simulate_path(daily_pnl, ...)     → single-path result dict
  run_simulations(n_sims, ...)      → aggregate statistics dict
"""

import math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# APEX ACCOUNT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ACCOUNT_SIZE       = 50_000.0   # Starting balance ($)
TRAILING_DD        = 2_500.0    # Max trailing drawdown ($)
TRAIL_STOP_LEVEL   = 50_100.0   # Floor locks here once peak crosses threshold
PAYOUT_THRESHOLD   = 52_600.0   # Balance required to trigger payout eligibility

DAILY_LOSS_LIMIT   = -700.0     # Worst allowed day PnL ($ — applied as a floor)
DAILY_PROFIT_CAP   =  1_050.0   # Best allowed day PnL ($ — applied as a ceiling)

MIN_DAYS           = 8          # Minimum calendar trading days before payout
MIN_GREEN_DAYS     = 5          # Minimum green days required for payout
GREEN_DAY_MIN      = 50.0       # A day qualifies as 'green' if PnL >= this ($)

MAX_PAYOUT         = 2_000.0    # Maximum single payout withdrawal ($)

# Derived geometry (used in diagnostics)
_D = TRAILING_DD                           # Distance to failure from start
_T = PAYOUT_THRESHOLD - ACCOUNT_SIZE      # Distance to target from start


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_pnl(csv_path: str) -> np.ndarray:
    """
    Load a trade history CSV and return an array of **daily** PnL totals in USD.

    Expected columns (whitespace-stripped):
      Type            — filter to rows where Type == "Trade"
      Date and time   — trade timestamp
      Net P&L USD     — per-trade profit/loss in USD

    If no "Trade" rows are found, falls back to rows where Type starts with
    "Exit" (TradingView-style CSVs where closed-trade PnL sits on exit rows).

    Returns
    -------
    np.ndarray
        1-D array of daily aggregate PnL values.  Each element represents
        the net USD result for one trading day.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # --- primary filter: Type == "Trade" -----------------------------------
    mask = df["Type"].str.strip() == "Trade"
    if mask.sum() == 0:
        # Fallback: TradingView export uses "Exit long" / "Exit short"
        mask = df["Type"].str.strip().str.startswith("Exit")
        if mask.sum() == 0:
            raise ValueError(
                f"No 'Trade' or 'Exit' rows found in '{csv_path}'.\n"
                "Ensure the CSV contains a 'Type' column with trade rows."
            )
        print(f"[load_daily_pnl] NOTE: No 'Trade' rows found; using 'Exit' rows instead.")

    df = df[mask].copy()

    df["Date and time"] = pd.to_datetime(df["Date and time"])
    df["Date"]          = df["Date and time"].dt.date

    daily = df.groupby("Date")["Net P&L USD"].sum()
    daily_pnl = daily.values.astype(float)

    print(f"\n[Data] Loaded from : {csv_path}")
    print(f"       Trading days  : {len(daily_pnl)}")
    print(f"       Mean day PnL  : ${np.mean(daily_pnl):>10,.2f}")
    print(f"       Std dev       : ${np.std(daily_pnl):>10,.2f}")
    print(f"       Best day      : ${np.max(daily_pnl):>10,.2f}")
    print(f"       Worst day     : ${np.min(daily_pnl):>10,.2f}")
    print(f"       Win-day rate  : {(daily_pnl > 0).mean():.1%}")

    return daily_pnl


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(daily_pnl: np.ndarray) -> dict:
    """
    Compute and print statistical diagnostics derived from the empirical daily
    PnL distribution.  Monte Carlo remains the primary evaluation method;
    these are informational approximations only.

    Metrics computed
    ----------------
    Drift / variance diagnostics
      mu                  — mean daily PnL ($)
      sigma               — std dev of daily PnL ($)
      variance            — sigma²
      drift_variance_ratio — mu / variance  (Kelly-related quantity)

    Prop firm geometry
      distance_to_failure — TRAILING_DD ($)
      distance_to_target  — PAYOUT_THRESHOLD − ACCOUNT_SIZE ($)

    Composite score
      PropScore           — (mu/sigma) * sqrt(D/T)
                            Sharpe-like ratio scaled by account geometry

    Kelly sizing
      f_star              — theoretical Kelly fraction (mu / variance)
      recommended_f       — conservative suggestion (20 % of f_star)

    Closed-form pass probability (drift-diffusion approximation)
      P_pass              — analytical gambler's-ruin approximation
                            (diagnostic only; ignores caps and rule constraints)

    Returns
    -------
    dict with all computed values.
    """
    mu       = float(np.mean(daily_pnl))
    sigma    = float(np.std(daily_pnl, ddof=1))
    variance = sigma ** 2

    D = _D   # distance to failure
    T = _T   # distance to target

    drift_variance_ratio = mu / variance if variance > 0 else float("nan")

    prop_score = (mu / sigma) * math.sqrt(D / T) if sigma > 0 else float("nan")

    f_star        = mu / variance if variance > 0 else float("nan")
    recommended_f = 0.20 * f_star if not math.isnan(f_star) else float("nan")

    # Closed-form gambler's-ruin approximation (drift-diffusion)
    # P_pass = (1 - e^{-2μD/σ²}) / (1 - e^{-2μ(D+T)/σ²})
    # Undefined / degenerate when mu = 0; returns nan in that case.
    if mu != 0 and variance > 0:
        arg_num = -2.0 * mu * D / variance
        arg_den = -2.0 * mu * (D + T) / variance
        # Guard against overflow in exp
        try:
            p_pass = (1.0 - math.exp(arg_num)) / (1.0 - math.exp(arg_den))
        except OverflowError:
            p_pass = float("nan")
    else:
        p_pass = float("nan")

    diag = {
        "mu":                    mu,
        "sigma":                 sigma,
        "variance":              variance,
        "drift_variance_ratio":  drift_variance_ratio,
        "distance_to_failure":   D,
        "distance_to_target":    T,
        "prop_score":            prop_score,
        "f_star":                f_star,
        "recommended_f":         recommended_f,
        "p_pass_closed_form":    p_pass,
    }

    # ── Print diagnostic report ──────────────────────────────────────────────
    w = 54
    print()
    print("═" * w)
    print("  STATISTICAL DIAGNOSTICS  (empirical daily PnL)")
    print("═" * w)

    print()
    print("  ── Drift / Variance ──────────────────────────────")
    print(f"  Mean daily PnL          : ${mu:>10,.2f}")
    print(f"  Std deviation           : ${sigma:>10,.2f}")
    print(f"  Variance (σ²)           : {variance:>12,.2f}")
    print(f"  Drift / variance ratio  : {drift_variance_ratio:>12.6f}")

    print()
    print("  ── Prop Firm Geometry ────────────────────────────")
    print(f"  Distance to failure (D) : ${D:>10,.2f}   [TRAILING_DD]")
    print(f"  Distance to target  (T) : ${T:>10,.2f}   [PAYOUT_THRESHOLD − START]")

    print()
    print("  ── Composite Score ───────────────────────────────")
    print(f"  PropScore (μ/σ)·√(D/T) : {prop_score:>12.4f}")
    _score_label(prop_score)

    print()
    print("  ── Kelly Sizing (theoretical) ────────────────────")
    print(f"  f*  = μ / σ²            : {f_star:>12.6f}")
    print(f"  Recommended (0.2 × f*)  : {recommended_f:>12.6f}")
    print(f"  NOTE: Kelly here is a daily-PnL ratio, not a")
    print(f"        position-size fraction. Use with caution.")

    print()
    print("  ── Closed-Form Pass Probability (diagnostic) ────")
    if not math.isnan(p_pass):
        print(f"  P(pass) ≈ {p_pass:.4%}   [drift-diffusion; ignores rules]")
    else:
        print(f"  P(pass) ≈ N/A  (μ = 0 or zero variance)")
    print(f"  Monte Carlo simulation is the primary estimate.")

    print("═" * w)
    print()

    return diag


def _score_label(prop_score: float):
    """Print a qualitative label for the PropScore."""
    if math.isnan(prop_score):
        print("  Rating                  :  N/A")
        return
    if prop_score >= 0.50:
        label = "STRONG EDGE"
    elif prop_score >= 0.25:
        label = "MODERATE EDGE"
    elif prop_score >= 0.10:
        label = "WEAK EDGE"
    elif prop_score > 0:
        label = "MARGINAL"
    else:
        label = "NEGATIVE DRIFT — review strategy"
    print(f"  Rating                  :  {label}")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PATH SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_path(
    daily_pnl: np.ndarray,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
) -> dict:
    """
    Simulate a single Apex account equity path at the **daily** level.

    Each simulated trading day draws one value from the empirical daily_pnl
    distribution (bootstrap resampling with replacement), scales it by
    risk_multiplier, applies Apex daily caps, and then applies the trailing
    drawdown and payout eligibility checks.

    Parameters
    ----------
    daily_pnl       : np.ndarray — empirical daily PnL values ($) from history
    risk_multiplier : float      — scale all PnL samples (1.0 = no change)
    max_days        : int        — hard ceiling on trading days
    stop_at_payout  : bool       — if True, return immediately on first payout

    Returns
    -------
    dict with keys:
      outcome     : "payout" | "blow" | "timeout"
      balance     : final account balance ($)
      days        : total trading days elapsed
      payout      : payout amount ($); 0 if no payout
      equity_path : list[float] — balance after each day (index 0 = start)
    """
    balance      = ACCOUNT_SIZE
    peak         = ACCOUNT_SIZE
    trading_days = 0
    green_days   = 0
    max_day_pnl  = 0.0       # largest single-day profit seen so far
    total_profit = 0.0       # cumulative profit (for 30 % rule)
    payout_amount = 0.0

    equity_path = [balance]

    for _day in range(max_days):

        # ── Sample and cap daily PnL ──────────────────────────────────────
        raw_pnl = float(np.random.choice(daily_pnl)) * risk_multiplier
        pnl     = max(DAILY_LOSS_LIMIT, min(DAILY_PROFIT_CAP, raw_pnl))

        balance += pnl
        equity_path.append(balance)
        trading_days += 1

        # ── Update peak ────────────────────────────────────────────────────
        if balance > peak:
            peak = balance

        # ── Apex trailing floor ────────────────────────────────────────────
        # The floor rises with peak_balance but locks at TRAIL_STOP_LEVEL once
        # peak_balance − TRAILING_DD reaches that level.
        if peak - TRAILING_DD < TRAIL_STOP_LEVEL:
            trailing_floor = peak - TRAILING_DD
        else:
            trailing_floor = TRAIL_STOP_LEVEL

        # ── Green-day accounting ───────────────────────────────────────────
        if pnl >= GREEN_DAY_MIN:
            green_days += 1

        # ── 30 % concentration tracking ────────────────────────────────────
        if pnl > max_day_pnl:
            max_day_pnl = pnl

        # ── Blow-up check ──────────────────────────────────────────────────
        if balance < trailing_floor:
            return {
                "outcome":     "blow",
                "balance":     balance,
                "days":        trading_days,
                "payout":      payout_amount,   # preserve any prior payout
                "equity_path": equity_path,
            }

        # ── Payout eligibility ─────────────────────────────────────────────
        if balance >= PAYOUT_THRESHOLD:
            total_profit = balance - ACCOUNT_SIZE

            eligible = (
                trading_days >= MIN_DAYS
                and green_days >= MIN_GREEN_DAYS
                and total_profit > 0
                and max_day_pnl <= 0.30 * total_profit
            )

            if eligible:
                # Withdrawable amount above the safety floor
                payout_floor = PAYOUT_THRESHOLD - 500.0
                withdrawable = balance - payout_floor

                if withdrawable >= 500.0:
                    payout_amount = min(withdrawable, MAX_PAYOUT)

                    if stop_at_payout:
                        return {
                            "outcome":     "payout",
                            "balance":     balance,
                            "days":        trading_days,
                            "payout":      payout_amount,
                            "equity_path": equity_path,
                        }

    # ── Max days reached without resolution ────────────────────────────────
    return {
        "outcome":     "timeout",
        "balance":     balance,
        "days":        trading_days,
        "payout":      payout_amount,
        "equity_path": equity_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulations(
    n_sims: int,
    daily_pnl: np.ndarray,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
    seed: int | None = None,
    n_paths: int = 50,
) -> dict:
    """
    Run n_sims Monte Carlo paths and return aggregate statistics.

    Parameters
    ----------
    n_sims          : int        — number of simulation paths
    daily_pnl       : np.ndarray — empirical daily PnL distribution ($)
    risk_multiplier : float      — scale factor applied to each sample
    max_days        : int        — hard day ceiling per path
    stop_at_payout  : bool       — stop each path on first payout
    seed            : int|None   — fix NumPy RNG for reproducibility
    n_paths         : int        — number of equity paths to retain for plotting

    Returns
    -------
    dict with keys:
      pass_rate         : float  — fraction of sims ending in payout
      fail_rate         : float  — fraction of sims that blew up
      timeout_rate      : float  — fraction of sims that hit max_days
      average_days      : float  — mean days across all paths
      average_payout    : float  — mean payout across paths that paid out
      outcomes          : list[str]
      balances          : list[float]
      payouts           : list[float]
      days              : list[int]
      equity_paths      : list[list[float]]  — up to n_paths sample paths
    """
    if seed is not None:
        np.random.seed(seed)

    outcomes     = []
    balances     = []
    payouts      = []
    days_list    = []
    equity_paths = []

    for i in range(n_sims):
        result = simulate_path(daily_pnl, risk_multiplier, max_days, stop_at_payout)
        outcomes.append(result["outcome"])
        balances.append(result["balance"])
        payouts.append(result["payout"])
        days_list.append(result["days"])
        if i < n_paths:
            equity_paths.append(result["equity_path"])

        if (i + 1) % 5_000 == 0:
            print(f"  ... {i + 1:,} / {n_sims:,} simulations complete")

    winning_payouts = [p for p in payouts if p > 0]

    return {
        "pass_rate":      sum(1 for p in payouts if p > 0) / n_sims,
        "fail_rate":      outcomes.count("blow")    / n_sims,
        "timeout_rate":   outcomes.count("timeout") / n_sims,
        "average_days":   float(np.mean(days_list)),
        "average_payout": float(np.mean(winning_payouts)) if winning_payouts else 0.0,
        "outcomes":       outcomes,
        "balances":       balances,
        "payouts":        payouts,
        "days":           days_list,
        "equity_paths":   equity_paths,
    }
