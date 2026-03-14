"""
recency_analysis.py
===================
Strategy Recency Analysis Layer for PassPlan — Prop Firm Probability Engine.

Computes a *second* Monte Carlo run over only the most-recent N trades and
compares it with the overall (full-history) result.  The delta and an
auto-generated comment allow users to see whether strategy edge is stable,
improving, or weakening.

Auto-selected window sizes
--------------------------
  >= 50 trades  → use last 50  (primary window)
  >= 30 trades  → use last 30  (reduced window)
  <  30 trades  → use all trades (window = full dataset)

Public API
----------
  get_recent_trade_subset(trades, window_size)        → np.ndarray
  compute_probability_delta(overall_prob, recent_prob) → float
  generate_recency_comment(delta)                      → dict
  select_window_size(n_trades)                         → int
  run_recency_simulation(csv_path, overall_payout_prob, n_sims, ...)  → dict

This module is intentionally standalone: no imports from api_server, no
modification of any existing engine file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apex_engine_v3_1 import (
    load_daily_pnl,
    run_simulations,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_WINDOW   = 50    # preferred recent window (trading days, legacy fallback)
FALLBACK_WINDOW  = 30    # used when fewer than PRIMARY_WINDOW days exist (legacy)
RECENCY_LOOKBACK_DAYS = 30   # calendar-day window for date-anchored recency
MIN_TRADES_WARNING    = 10   # warn when fewer than this many trades are in window

# Thresholds for status classification
IMPROVING_THRESHOLD  =  0.10   # delta >= +0.10  → Improving
WEAKENING_THRESHOLD  = -0.10   # delta <= -0.10  → Weakening


# ─────────────────────────────────────────────────────────────────────────────
# Core utility functions
# ─────────────────────────────────────────────────────────────────────────────

def select_window_size(n_trades: int) -> int:
    """Choose the appropriate recent-window size for a given trade count.

    Returns
    -------
    int — number of trades to use as the recent window
        PRIMARY_WINDOW (50)  if n_trades >= 50
        FALLBACK_WINDOW (30) if 30 <= n_trades < 50
        n_trades             if n_trades < 30 (use full dataset)
    """
    if n_trades >= PRIMARY_WINDOW:
        return PRIMARY_WINDOW
    if n_trades >= FALLBACK_WINDOW:
        return FALLBACK_WINDOW
    return n_trades   # full dataset — window equals the entire history


def load_trades_df(csv_path: str) -> pd.DataFrame:
    """Load all individual trade rows from a CSV, sorted chronologically.

    Returns a DataFrame with at least 'Date and time' and 'Net P&L USD' columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    mask = df["Type"].str.strip() == "Trade"
    if mask.sum() == 0:
        mask = df["Type"].str.strip().str.startswith("Exit")
    df = df[mask].copy()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    return df.sort_values("Date and time").reset_index(drop=True)


def get_recent_trades_as_daily_pnl(
    csv_path: str,
    lookback_days: int = RECENCY_LOOKBACK_DAYS,
) -> tuple:
    """Load trades from the last *lookback_days* calendar days and aggregate
    to daily PnL.

    The window is anchored to the most-recent trade date in the CSV, so
    strategies with different backtest end-dates are handled correctly.

    Returns
    -------
    (daily_pnl, n_recent_trades, n_active_days, full_n_trades, full_n_days,
     low_sample_warning)
        daily_pnl         : np.ndarray — aggregated PnL per calendar day
        n_recent_trades   : int        — individual trade rows in window
        n_active_days     : int        — distinct trading days in window
        full_n_trades     : int        — total individual trades in CSV
        full_n_days       : int        — distinct calendar days in full history
        low_sample_warning: bool       — True when n_recent_trades < MIN_TRADES_WARNING
    """
    df             = load_trades_df(csv_path)
    full_n_trades  = len(df)
    df["Date"]     = df["Date and time"].dt.date
    full_n_days    = df["Date"].nunique()

    latest         = df["Date and time"].max()
    cutoff         = latest - pd.Timedelta(days=lookback_days)
    recent         = df[df["Date and time"] > cutoff].copy()

    # Fallback: if the window is empty (e.g. very sparse CSV), use the
    # most-recent MIN_TRADES_WARNING trades so we always return something.
    if len(recent) == 0:
        recent = df.tail(MIN_TRADES_WARNING).copy()

    n_recent_trades    = len(recent)
    daily              = recent.groupby("Date")["Net P&L USD"].sum()
    n_active_days      = len(daily)
    low_sample_warning = n_recent_trades < MIN_TRADES_WARNING

    return (
        daily.values.astype(float),
        n_recent_trades,
        n_active_days,
        full_n_trades,
        full_n_days,
        low_sample_warning,
    )


def get_recent_trade_subset(
    trades: np.ndarray,
    window_size: int | None = None,
) -> np.ndarray:
    """Return the most-recent *window_size* trades from *trades*.

    Parameters
    ----------
    trades      : 1-D numpy array of daily PnL values (chronological)
    window_size : number of trades to include.
                  If None, ``select_window_size(len(trades))`` is used.
                  If window_size >= len(trades) the full array is returned.

    Returns
    -------
    np.ndarray — slice of the last window_size elements (or all elements)
    """
    if window_size is None:
        window_size = select_window_size(len(trades))
    if window_size >= len(trades):
        return trades
    return trades[-window_size:]


def compute_probability_delta(
    overall_prob: float,
    recent_prob: float,
) -> float:
    """Compute the signed probability delta: recent − overall.

    Positive → recent performance is *better* than historical average.
    Negative → recent performance is *worse* than historical average.

    Returns
    -------
    float, rounded to 4 decimal places
    """
    return round(recent_prob - overall_prob, 4)


def generate_recency_comment(delta: float) -> dict:
    """Map a probability delta to a human-readable status and comment.

    Parameters
    ----------
    delta : float — output of compute_probability_delta()

    Returns
    -------
    dict with keys:
        status  — "Improving" | "Stable" | "Weakening"
        comment — plain-English explanation
    """
    if delta >= IMPROVING_THRESHOLD:
        return {
            "status":  "Improving",
            "comment": "Recent trades indicate improving strategy conditions.",
        }
    if delta <= WEAKENING_THRESHOLD:
        return {
            "status":  "Weakening",
            "comment": "Recent trades indicate weakening strategy conditions.",
        }
    return {
        "status":  "Stable",
        "comment": "Recent performance is broadly consistent with historical results.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_recency_simulation(
    csv_path: str,
    overall_payout_prob: float,
    n_sims: int = 5_000,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
    seed: int | None = None,
) -> dict:
    """Run a secondary Monte Carlo on the recent trade window.

    Loads the same CSV, extracts the auto-selected recent window, runs
    *n_sims* simulations (with stop_at_payout=True by default so it matches
    the primary UTP simulation), and returns a complete recency block.

    Parameters
    ----------
    csv_path            : path to the strategy CSV
    overall_payout_prob : pass probability from the primary (full) simulation
    n_sims              : simulations to run on the recent window
                          (default 5 000 — half of a typical 10 000-sim run)
    risk_multiplier     : passed through to run_simulations unchanged
    max_days            : hard cap per path (should match the primary run)
    stop_at_payout      : whether to stop each path at first payout
    seed                : optional RNG seed

    Returns
    -------
    dict with keys:
        overall_pass_probability : float  — from caller (echoed back)
        recent_pass_probability  : float  — from the recent-window run
        probability_delta        : float  — recent − overall
        recency_status           : str    — "Improving" | "Stable" | "Weakening"
        recency_comment          : str    — plain-English interpretation
        recent_window_size       : int    — actual recent window used
        recent_n_days            : int    — number of trading days in subset
        full_n_days              : int    — total trading days available in CSV
    """
    # Load full trade history
    full_pnl = load_daily_pnl(csv_path)
    n_full   = len(full_pnl)

    window   = select_window_size(n_full)
    recent   = get_recent_trade_subset(full_pnl, window)

    # Run recent simulation
    if seed is not None:
        np.random.seed(seed)

    recent_results = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = recent,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        stop_at_payout  = stop_at_payout,
        n_paths         = 0,       # no equity-path capture needed
        mode            = "uniform",
    )

    recent_prob = recent_results["pass_rate"]
    delta       = compute_probability_delta(overall_payout_prob, recent_prob)
    comment     = generate_recency_comment(delta)

    return {
        "overall_pass_probability": round(float(overall_payout_prob), 4),
        "recent_pass_probability":  round(float(recent_prob), 4),
        "probability_delta":        delta,
        "recency_status":           comment["status"],
        "recency_comment":          comment["comment"],
        "recent_window_size":       window,
        "recent_n_days":            len(recent),
        "full_n_days":              n_full,
    }


def run_recency_simulation_from_trades(
    csv_path: str,
    overall_payout_prob: float,
    lookback_days: int = RECENCY_LOOKBACK_DAYS,
    n_sims: int = 5_000,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
    seed: int | None = None,
) -> dict:
    """Run a recency simulation using trades from the last *lookback_days*
    calendar days.

    The window is calendar-anchored so that strategies with different trading
    frequencies all reflect the same market regime in their recency window.
    A ``low_sample_warning`` flag is set when fewer than MIN_TRADES_WARNING
    trades fall inside the window.

    Returns
    -------
    dict with keys:
        overall_pass_probability : float
        recent_pass_probability  : float
        probability_delta        : float
        recency_status           : str
        recency_comment          : str
        lookback_days            : int   — calendar days requested
        recent_n_trades          : int   — individual trade rows sampled
        n_active_days            : int   — distinct trading days in window
        full_n_trades            : int   — total rows in CSV
        full_n_days              : int   — total trading days in CSV
        low_sample_warning       : bool  — True when < MIN_TRADES_WARNING trades
    """
    daily_pnl, n_recent_trades, n_active_days, full_n_trades, full_n_days, low_warn = \
        get_recent_trades_as_daily_pnl(csv_path, lookback_days)

    if seed is not None:
        np.random.seed(seed)

    recent_results = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = daily_pnl,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        stop_at_payout  = stop_at_payout,
        n_paths         = 0,
        mode            = "uniform",
    )

    recent_prob = recent_results["pass_rate"]
    delta       = compute_probability_delta(overall_payout_prob, recent_prob)
    comment     = generate_recency_comment(delta)

    return {
        "overall_pass_probability": round(float(overall_payout_prob), 4),
        "recent_pass_probability":  round(float(recent_prob), 4),
        "probability_delta":        delta,
        "recency_status":           comment["status"],
        "recency_comment":          comment["comment"],
        "lookback_days":            lookback_days,
        "recent_n_trades":          n_recent_trades,
        "n_active_days":            n_active_days,
        "full_n_trades":            full_n_trades,
        "full_n_days":              full_n_days,
        "low_sample_warning":       low_warn,
    }


def run_recency_simulation_from_pnl(
    full_pnl: np.ndarray,
    overall_payout_prob: float,
    n_sims: int = 5_000,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
    seed: int | None = None,
) -> dict:
    """Identical to run_recency_simulation() but accepts a pre-loaded PnL array.

    Use this variant when the caller has already loaded the CSV to avoid
    a redundant file read.

    Parameters
    ----------
    full_pnl            : 1-D numpy array of full daily PnL (chronological)
    overall_payout_prob : pass probability from the primary simulation
    (remaining parameters identical to run_recency_simulation)

    Returns
    -------
    Same dict shape as run_recency_simulation().
    """
    n_full   = len(full_pnl)
    window   = select_window_size(n_full)
    recent   = get_recent_trade_subset(full_pnl, window)

    if seed is not None:
        np.random.seed(seed)

    recent_results = run_simulations(
        n_sims          = n_sims,
        daily_pnl       = recent,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        stop_at_payout  = stop_at_payout,
        n_paths         = 0,
        mode            = "uniform",
    )

    recent_prob = recent_results["pass_rate"]
    delta       = compute_probability_delta(overall_payout_prob, recent_prob)
    comment     = generate_recency_comment(delta)

    return {
        "overall_pass_probability": round(float(overall_payout_prob), 4),
        "recent_pass_probability":  round(float(recent_prob), 4),
        "probability_delta":        delta,
        "recency_status":           comment["status"],
        "recency_comment":          comment["comment"],
        "recent_window_size":       window,
        "recent_n_days":            len(recent),
        "full_n_days":              n_full,
    }
