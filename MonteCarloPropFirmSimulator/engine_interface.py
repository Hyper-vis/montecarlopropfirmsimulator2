"""
engine_interface.py
===================
Single programmatic entry-point between the simulation backend and any
higher-level caller (FastAPI, UI, tests, notebooks).

All public functions are pure callables — no stdin prompts, no stdout
printing, no matplotlib side-effects.  They accept typed parameters and
return structured dicts (JSON-serialisable except where noted).

Public API
----------
  analyze_until_payout(csv_path, ...)             → dict
  analyze_full_period(csv_path, ...)              → dict
  run_batch(csv_paths, ..., progress_cb)          → list[dict]
  analyze_correlation(named_csvs, ...)            → dict
  run_multi_account(csv_path, n_accounts, ...)    → dict

JSON-safety note
----------------
All public return dicts are fully JSON-serialisable.  Keys prefixed with
"_" (e.g. "_account_paths", "_sim_results") are intended for internal /
CLI use and should be stripped before passing to json.dumps() if the
values are non-serialisable (e.g. large equity_path lists or tuples).

Version history
---------------
  1.0  — initial release (Subtask 2.1 of API architecture)
  1.1  — added run_multi_account(); fixed numpy/pandas JSON serialisation
         in analyze_correlation() via _json_safe() helper (Phase 2 Task 2-3)
  1.2  — standardised result shapes {metadata/probabilities/metrics/distributions};
         added seed: int | None = None to all five public functions;
         added _corr_dict_to_matrix() helper for correlation output;
         applied _json_safe() to run_batch and run_multi_account returns
         (Phase 2 Task 1-3 verification pass)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ── refactored runner entry-points (pure functions, no print/input/chart) ────
from run_until_payout_v3_1 import (
    run_until_payout_analysis,
    N_SIMS          as _UTP_N_SIMS,
    RISK_MULTIPLIER as _UTP_RISK,
    MAX_DAYS        as _UTP_MAX_DAYS,
    N_PLOT_PATHS    as _UTP_N_PATHS,
    SAMPLING_MODE   as _UTP_MODE,
    WEIGHT_STRENGTH as _UTP_WEIGHT,
    RECENT_WINDOW   as _UTP_WINDOW,
)
from run_full_period_v3_1 import (
    run_full_period_analysis,
    N_SIMS          as _FP_N_SIMS,
    RISK_MULTIPLIER as _FP_RISK,
    MAX_DAYS        as _FP_MAX_DAYS,
    N_PLOT_PATHS    as _FP_N_PATHS,
    RESET_COST      as _FP_RESET_COST,
    SAMPLING_MODE   as _FP_MODE,
    WEIGHT_STRENGTH as _FP_WEIGHT,
    RECENT_WINDOW   as _FP_WINDOW,
)

# ── batch runner (simulate_csv is already a pure function) ───────────────────
from batch_runner import (
    simulate_csv,
    N_SIMS          as _BATCH_N_SIMS,
    SAMPLING_MODE   as _BATCH_MODE,
    SAMPLING_WEIGHT_STR    as _BATCH_WEIGHT,
    SAMPLING_RECENT_WINDOW as _BATCH_WINDOW,
    RESET_COST      as _BATCH_RESET_COST,
)

# ── multi-account simulator (pure function, no module-level side-effects) ────
from multi_account_simulator import (
    run_multi_account_analysis,
    TRADING_DAYS_PER_MONTH as _MA_DAYS_PER_MONTH,
    CYCLE_GATE             as _MA_CYCLE_GATE,
)

# ── correlation analysis helpers (all pure functions) ────────────────────────
from strategy_correlation_analyzer import (
    load_daily_pnl_series,
    build_aligned_frame,
    return_correlation,
    rolling_correlation_pairs,
    simultaneous_drawdown_table,
    worst_overlap_periods,
    mc_blow_correlation,
    equity_curves,
    drawdown_curves,
)
from apex_engine_v3_1 import (
    ACCOUNT_SIZE, TRAILING_DD, TRAIL_STOP_LEVEL, PAYOUT_THRESHOLD,
    DAILY_LOSS_LIMIT, DAILY_PROFIT_CAP, MIN_DAYS, MIN_GREEN_DAYS,
    GREEN_DAY_MIN, MAX_PAYOUT,
    load_daily_pnl as _load_daily_pnl,
)
import recency_analysis as _ra


# ─────────────────────────────────────────────────────────────────────────────
# JSON-SAFETY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to JSON-serialisable Python types.

    Handles numpy scalars, numpy arrays, pandas Timestamps, and containers
    (dict / list / tuple) by recursing into their elements.  All other types
    are returned unchanged.

    This is applied at the boundary of analyze_correlation() where pandas
    DataFrame.to_dict() returns numpy float64 values and pandas Timestamps.
    """
    # bool must be checked before int (bool is a subclass of int in Python)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {_json_safe(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# RECENCY ANALYSIS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _inject_recency(
    full_pnl: np.ndarray,
    overall_prob: float,
    n_sims: int,
    max_days: int,
    stop_at_payout: bool,
    seed: Optional[int],
) -> dict:
    """Run a secondary Monte Carlo on the recent trade window and return
    the recency block to be merged into a simulation result dict.

    Uses at most half of *n_sims* (minimum 1 000) for the recent run so that
    overall response time stays well below 2× the primary run.
    """
    n_recent = max(1_000, n_sims // 2)
    try:
        return _ra.run_recency_simulation_from_pnl(
            full_pnl            = full_pnl,
            overall_payout_prob = overall_prob,
            n_sims              = n_recent,
            max_days            = max_days,
            stop_at_payout      = stop_at_payout,
            seed                = seed,
        )
    except Exception:
        # Never let recency errors crash the primary result.
        interp = _ra.generate_recency_comment(0.0)
        return {
            "overall_pass_probability": round(float(overall_prob), 4),
            "recent_pass_probability":  None,
            "probability_delta":        None,
            "recency_status":           interp["status"],
            "recency_comment":          "Recency analysis unavailable.",
            "recent_window_size":       None,
            "recent_n_trades":          None,
            "full_n_trades":            len(full_pnl),
        }


def _corr_dict_to_matrix(corr_dict: dict) -> dict:
    """Convert a dict-of-dict correlation matrix to {labels, matrix} format.

    Parameters
    ----------
    corr_dict : dict[str, dict[str, float]]
        Output of pandas DataFrame.corr().to_dict() — outer key = row label,
        inner key = column label.

    Returns
    -------
    dict with keys:
        labels : list[str]         – strategy names in matrix order
        matrix : list[list[float]] – row-major 2-D correlation matrix
    """
    labels = list(corr_dict.keys())
    matrix = [
        [corr_dict[row][col] for col in labels]
        for row in labels
    ]
    return {"labels": labels, "matrix": matrix}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Run-Until-Payout analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_until_payout(
    csv_path: str,
    n_sims: int = _UTP_N_SIMS,
    risk_multiplier: float = _UTP_RISK,
    max_days: int = _UTP_MAX_DAYS,
    n_paths: int = _UTP_N_PATHS,
    sampling_mode: str = _UTP_MODE,
    weight_strength: float = _UTP_WEIGHT,
    recent_window: int = _UTP_WINDOW,
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo analysis: simulate runs until payout threshold or blow/timeout.

    Parameters
    ----------
    csv_path        : path to CSV of daily PnL trades
    n_sims          : number of Monte Carlo simulations
    risk_multiplier : scale all daily PnL by this factor
    max_days        : hard cap on days per simulation run
    n_paths         : number of equity paths to capture for charts
    sampling_mode   : "uniform" | "recency_weighted" | "recent_only"
    weight_strength : exponential steepness (recency_weighted only)
    recent_window   : look-back in trading days (recent_only only)
    seed            : optional RNG seed for reproducible results

    Returns
    -------
    dict with keys:
        metadata       – run parameters
        probabilities  – payout_prob, blow_prob, timeout_prob
        metrics        – mean_days_all, mean_days_to_payout,
                         median_days_to_payout, mean_payout,
                         tier_label, tier_description
        distributions  – equity_paths (list[list[float]])
    """
    if seed is not None:
        np.random.seed(seed)

    raw = run_until_payout_analysis(
        csv_path        = csv_path,
        n_sims          = n_sims,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        n_paths         = n_paths,
        sampling_mode   = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
    )

    from apex_engine_v3_1 import load_daily_pnl as _load_pnl
    full_pnl = _load_pnl(csv_path)
    recency  = _inject_recency(full_pnl, raw["payout_prob"], n_sims, max_days, True, seed)

    return {
        "metadata": raw["metadata"],
        "probabilities": {
            "payout_prob":   raw["payout_prob"],
            "blow_prob":     raw["blow_prob"],
            "timeout_prob":  raw["timeout_prob"],
        },
        "metrics": {
            "mean_days_all":           raw["mean_days_all"],
            "mean_days_to_payout":     raw["mean_days_to_payout"],
            "median_days_to_payout":   raw["median_days_to_payout"],
            "mean_payout":             raw["mean_payout"],
            "tier_label":              raw["tier_label"],
            "tier_description":        raw["tier_description"],
        },
        "distributions": {
            "equity_paths": raw["equity_paths"],
        },
        "recency": recency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Full-period analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_full_period(
    csv_path: str,
    n_sims: int = _FP_N_SIMS,
    risk_multiplier: float = _FP_RISK,
    max_days: int = _FP_MAX_DAYS,
    n_paths: int = _FP_N_PATHS,
    reset_cost: float = _FP_RESET_COST,
    sampling_mode: str = _FP_MODE,
    weight_strength: float = _FP_WEIGHT,
    recent_window: int = _FP_WINDOW,
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo analysis: simulate a fixed-duration window (stop_at_payout=False).

    Parameters
    ----------
    csv_path        : path to CSV of daily PnL trades
    n_sims          : number of Monte Carlo simulations
    risk_multiplier : scale all daily PnL by this factor
    max_days        : trading-day window length (default 21 ≈ 1 calendar month)
    n_paths         : number of equity paths to capture for charts
    reset_cost      : cost to reset a blown account ($) used in EV calculation
    sampling_mode   : "uniform" | "recency_weighted" | "recent_only"
    weight_strength : exponential steepness (recency_weighted only)
    recent_window   : look-back in trading days (recent_only only)
    seed            : optional RNG seed for reproducible results

    Returns
    -------
    dict with keys:
        metadata       – run parameters (including reset_cost)
        probabilities  – payout_prob, payout_survived_prob,
                         payout_then_blew_prob, blow_no_payout_prob,
                         timeout_no_pay_prob
        metrics        – mean_ending_balance, mean_blow_loss,
                         mean_payout, median_payout, e_monthly,
                         tier_label, tier_description
        distributions  – equity_paths (list[list[float]])
    """
    if seed is not None:
        np.random.seed(seed)

    raw = run_full_period_analysis(
        csv_path        = csv_path,
        n_sims          = n_sims,
        risk_multiplier = risk_multiplier,
        max_days        = max_days,
        n_paths         = n_paths,
        reset_cost      = reset_cost,
        sampling_mode   = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
    )

    from apex_engine_v3_1 import load_daily_pnl as _load_pnl
    full_pnl = _load_pnl(csv_path)
    recency  = _inject_recency(full_pnl, raw["payout_prob"], n_sims, max_days, False, seed)

    return {
        "metadata": raw["metadata"],
        "probabilities": {
            "payout_prob":              raw["payout_prob"],
            "payout_survived_prob":     raw["payout_survived_prob"],
            "payout_then_blew_prob":    raw["payout_then_blew_prob"],
            "blow_no_payout_prob":      raw["blow_no_payout_prob"],
            "timeout_no_pay_prob":      raw["timeout_no_pay_prob"],
        },
        "metrics": {
            "mean_ending_balance":  raw["mean_ending_balance"],
            "mean_blow_loss":       raw["mean_blow_loss"],
            "mean_payout":         raw["mean_payout"],
            "median_payout":       raw["median_payout"],
            "e_monthly":           raw["e_monthly"],
            "tier_label":          raw["tier_label"],
            "tier_description":    raw["tier_description"],
        },
        "distributions": {
            "equity_paths": raw["equity_paths"],
        },
        "recency": recency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Batch analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    csv_paths: list[str],
    n_sims: int = _BATCH_N_SIMS,
    sampling_mode: str = _BATCH_MODE,
    weight_strength: float = _BATCH_WEIGHT,
    recent_window: int = _BATCH_WINDOW,
    reset_cost: float = _BATCH_RESET_COST,
    progress_cb: Callable[[int, int, str], None] | None = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Run both simulations (UTP + full-period) on every CSV in csv_paths.

    Parameters
    ----------
    csv_paths     : list of CSV file paths to analyse
    n_sims        : simulations per CSV per mode
    sampling_mode : "uniform" | "recency_weighted" | "recent_only"
    weight_strength, recent_window : sampling tuning parameters
    reset_cost    : account reset fee used in EV calculation ($)
    progress_cb   : optional callback invoked after each CSV completes.
                    Signature: progress_cb(completed: int, total: int, csv_path: str)
    seed          : optional RNG seed for reproducible results

    Returns
    -------
    list[dict] — one result dict per CSV, sorted by full-period net EV desc.
    Each dict matches the schema returned by batch_runner.simulate_csv().
    All numeric values are JSON-serialisable Python floats.
    Failures are recorded as dicts with an "error" key instead of results.
    """
    if seed is not None:
        np.random.seed(seed)
    # Temporarily override the module-level config in batch_runner so that
    # simulate_csv() uses the caller's parameters.  This is safe for single-
    # threaded use; for concurrent calls a per-call parameter pass-through
    # would require a refactor of simulate_csv() itself.
    import batch_runner as _br
    _orig = (
        _br.N_SIMS, _br.SAMPLING_MODE, _br.SAMPLING_WEIGHT_STR,
        _br.SAMPLING_RECENT_WINDOW, _br.RESET_COST,
    )
    _br.N_SIMS                  = n_sims
    _br.SAMPLING_MODE           = sampling_mode
    _br.SAMPLING_WEIGHT_STR     = weight_strength
    _br.SAMPLING_RECENT_WINDOW  = recent_window
    _br.RESET_COST              = reset_cost

    results: list[dict] = []
    total = len(csv_paths)

    try:
        for i, csv_path in enumerate(csv_paths, 1):
            try:
                row = simulate_csv(csv_path, n_sims)
                # Inject recency metrics using UTP pass-rate as the overall prob
                try:
                    from apex_engine_v3_1 import load_daily_pnl as _lpnl
                    _pnl = _lpnl(csv_path)
                    _rec = _inject_recency(
                        _pnl,
                        overall_prob   = row.get("utp_payout_p", 0.0),
                        n_sims         = max(1_000, n_sims // 4),
                        max_days       = 90,
                        stop_at_payout = True,
                        seed           = None,
                    )
                    row["recent_pass_probability"] = _rec["recent_pass_probability"]
                    row["probability_delta"]       = _rec["probability_delta"]
                    row["recency_status"]          = _rec["recency_status"]
                    row["recency_comment"]         = _rec["recency_comment"]
                    row["recent_window_size"]      = _rec["recent_window_size"]
                except Exception:
                    pass  # recency failure must never break batch
                results.append(row)
            except Exception as exc:
                results.append({"csv": csv_path, "error": str(exc)})
            if progress_cb is not None:
                progress_cb(i, total, csv_path)
    finally:
        # Always restore original config even if an exception occurs.
        (
            _br.N_SIMS, _br.SAMPLING_MODE, _br.SAMPLING_WEIGHT_STR,
            _br.SAMPLING_RECENT_WINDOW, _br.RESET_COST,
        ) = _orig

    # Sort successful rows by full-period net EV descending; errors go last.
    results.sort(key=lambda r: r.get("fp_ev_net", float("-inf")), reverse=True)
    # Ensure all numeric values from strategy_score and batch_runner are JSON-safe
    return [_json_safe(r) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_correlation(
    named_csvs: dict[str, str],
    rolling_window: int = 30,
    dd_threshold: float = -500.0,
    mc_n_sims: int = 5000,
    mc_max_days: int = 30,
    sampling_mode: str = "uniform",
    weight_strength: float = 3.0,
    recent_window: int = 50,
    seed: Optional[int] = None,
) -> dict:
    """Compute pairwise correlation and simultaneous drawdown metrics.

    Parameters
    ----------
    named_csvs      : mapping of {strategy_name: csv_path} — minimum 2 entries
    rolling_window  : rolling correlation window in trading days
    dd_threshold    : drawdown level ($) considered a "danger" event
    mc_n_sims       : Monte Carlo sims for blow-up correlation matrix
    mc_max_days     : days per Monte Carlo run for blow-up correlation
    sampling_mode   : "uniform" | "recency_weighted" | "recent_only"
    weight_strength : exponential steepness (recency_weighted only)
    recent_window   : look-back in trading days (recent_only only)
    seed            : optional RNG seed for reproducible MC blow-correlations

    Returns
    -------
    dict with keys:
        labels              – list of strategy names (shared across all matrices)
        return_correlation  – {labels: list[str], matrix: list[list[float]]}
        blow_correlation    – {labels: list[str], matrix: list[list[float]]}
        rolling_pairs       – {pair_label: list[float]} rolling correlations
        simultaneous_dd     – list of dicts (date, count, strategy flags)
        worst_overlap       – list of dicts (top worst simultaneous DD periods)
        equity_curves       – {strategy_name: list[float]} cumulative equity
        drawdown_curves     – {strategy_name: list[float]} drawdown series
    """
    if seed is not None:
        np.random.seed(seed)
    if len(named_csvs) < 2:
        raise ValueError("analyze_correlation requires at least 2 strategies.")

    # Load all strategy PnL series
    strategies: dict = {}
    for name, csv_path in named_csvs.items():
        strategies[name] = load_daily_pnl_series(csv_path, name)

    pnl_df  = build_aligned_frame(strategies)
    eq_df   = equity_curves(pnl_df, ACCOUNT_SIZE)
    dd_df   = drawdown_curves(eq_df)

    ret_corr     = return_correlation(pnl_df)
    rolling      = rolling_correlation_pairs(pnl_df, window=rolling_window)
    simult_dd    = simultaneous_drawdown_table(dd_df, dd_threshold=dd_threshold)
    worst        = worst_overlap_periods(dd_df, dd_threshold=dd_threshold)
    blow_df      = mc_blow_correlation(
        strategies,
        n_sims          = mc_n_sims,
        max_days        = mc_max_days,
        mode            = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
    )
    blow_corr = blow_df.corr()

    labels = list(named_csvs.keys())
    return _json_safe({
        "labels":             labels,
        "return_correlation": _corr_dict_to_matrix(ret_corr.to_dict()),
        "blow_correlation":   _corr_dict_to_matrix(blow_corr.to_dict()),
        "rolling_pairs":      {k: v.dropna().tolist() for k, v in rolling.items()},
        "simultaneous_dd":    simult_dd.reset_index().to_dict(orient="records"),
        "worst_overlap":      worst.reset_index().to_dict(orient="records"),
        "equity_curves":      eq_df.to_dict(orient="list"),
        "drawdown_curves":    dd_df.to_dict(orient="list"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Multi-account analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_account(
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
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo analysis across a portfolio of concurrent Apex accounts.

    Parameters
    ----------
    csv_path        : path to CSV of daily PnL trades
    n_accounts      : number of parallel Apex accounts in the portfolio
    n_sims          : number of Monte Carlo simulations
    max_days        : maximum trading days per account cycle
    stop_at_payout  : if True, each account stops at first payout
    risk_multiplier : scale all daily PnL by this factor
    sampling_mode   : "uniform" | "recency_weighted" | "recent_only"
    weight_strength : exponential steepness (recency_weighted only)
    recent_window   : look-back in trading days (recent_only only)
    n_path_sims     : equity path samples to capture (for chart generation)

    Returns
    -------
    dict — JSON-serialisable subset of run_multi_account_analysis() output:

        metadata               – run parameters
        portfolio              – aggregate payout / balance statistics
            mean_total_payout       float  – mean portfolio payout per sim
            median_total_payout     float
            p5_payout               float  – 5th-percentile payout
            p95_payout              float  – 95th-percentile payout
            prob_any_payout         float  – P(portfolio payout > 0)
            mean_blown_accounts     float  – mean blown accounts per sim
            mean_end_balance_single     float  – mean ending balance per account
            mean_end_balance_portfolio  float  – mean total ending balance
            mean_end_balance_per_acct   list[float]
        cycle_efficiency       – monthly throughput metrics (empty dict if no payouts)
            payout_prob_per_cycle   float
            mean_days_to_payout     float
            median_days_to_payout   float
            p5_days / p95_days      float
            pct_within_gate         float  – fraction hitting within CYCLE_GATE days
            pct_within_10           float
            pct_within_15           float
            est_cycles_per_month    float
            est_monthly_gross       float
            est_monthly_net         float

    Note: "_portfolio_payouts" and "_account_paths" are NOT included in this
    return value as they are large internal arrays unsuitable for API responses.
    """
    if seed is not None:
        np.random.seed(seed)

    raw = run_multi_account_analysis(
        csv_path        = csv_path,
        n_accounts      = n_accounts,
        n_sims          = n_sims,
        max_days        = max_days,
        stop_at_payout  = stop_at_payout,
        risk_multiplier = risk_multiplier,
        sampling_mode   = sampling_mode,
        weight_strength = weight_strength,
        recent_window   = recent_window,
        n_path_sims     = n_path_sims,
    )
    # Build chart_data: sampled account paths + portfolio payout histogram data
    # Limit to first 9 accounts and first 150 path-sims to keep payload small.
    MAX_CHART_ACCTS = 9
    raw_paths = raw["_account_paths"]  # list[acct_idx -> list[(equity_path, outcome)]]
    chart_account_paths = []
    for acct_paths in raw_paths[:MAX_CHART_ACCTS]:
        chart_account_paths.append([
            {"path": p, "outcome": o}
            for p, o in acct_paths
        ])

    return _json_safe({
        "metadata":         raw["metadata"],
        "portfolio":        raw["portfolio"],
        "cycle_efficiency": raw["cycle_efficiency"],
        "chart_data": {
            "portfolio_payouts": raw["_portfolio_payouts"],
            "account_paths":     chart_account_paths,
            "thresholds": {
                "start":         50_000.0,
                "payout":        52_600.0,
                "blow_floor":    47_500.0,   # ACCOUNT_SIZE − TRAILING_DD at start
            },
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# RESCUE / WHAT-IF ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _rescue_sim_vec(
    pool: np.ndarray,
    n_sims: int,
    current_balance: float,
    current_peak: float,
    max_days: int,
):
    """Vectorised Monte-Carlo from a mid-account state.

    Returns
    -------
    payout_amts : np.ndarray shape (n_sims,)   — payout amount, 0 if no payout
    is_blow     : np.ndarray shape (n_sims,)   — True if blown before payout
    """
    raw = np.random.choice(pool, size=(n_sims, max_days), replace=True)
    pnl = np.clip(raw, DAILY_LOSS_LIMIT, DAILY_PROFIT_CAP)

    bal = np.empty((n_sims, max_days + 1), dtype=np.float64)
    bal[:, 0] = current_balance
    bal[:, 1:] = current_balance + np.cumsum(pnl, axis=1)

    init_peak = np.full((n_sims, 1), current_peak, dtype=np.float64)
    running_peak = np.maximum.accumulate(
        np.concatenate([init_peak, bal[:, 1:]], axis=1), axis=1
    )
    trailing_floor = np.minimum(running_peak - TRAILING_DD, TRAIL_STOP_LEVEL)

    blown = bal[:, 1:] < trailing_floor[:, 1:]
    any_blown = blown.any(axis=1)
    blow_day = np.where(any_blown, np.argmax(blown, axis=1), max_days)

    # green-day and eligibility tracking
    green_cum = np.cumsum(pnl >= GREEN_DAY_MIN, axis=1)
    day_counts = np.arange(1, max_days + 1, dtype=np.int32)

    # concentration: max single-day profit ≤ 30% of total profit
    max_day_pnl_cum = np.maximum.accumulate(np.maximum(pnl, 0.0), axis=1)
    total_profit = bal[:, 1:] - ACCOUNT_SIZE  # measured from original $50k base
    conc_ok = (total_profit <= 0) | (
        max_day_pnl_cum <= 0.30 * np.maximum(total_profit, 1e-9)
    )

    eligible = (
        (day_counts[np.newaxis, :] >= MIN_DAYS)
        & (green_cum >= MIN_GREEN_DAYS)
        & (total_profit > 0)
        & conc_ok
        & (bal[:, 1:] >= PAYOUT_THRESHOLD)
    )
    any_eligible = eligible.any(axis=1)
    payout_day = np.where(any_eligible, np.argmax(eligible, axis=1), max_days)
    has_payout = any_eligible & (payout_day < blow_day)

    # payout amount: withdrawable above floor = PAYOUT_THRESHOLD − 500, capped at MAX_PAYOUT
    payout_floor = PAYOUT_THRESHOLD - 500.0
    sim_idx = np.arange(n_sims)
    safe_pd = np.clip(payout_day, 0, max_days - 1)
    bal_at_pay = bal[sim_idx, safe_pd + 1]
    withdrawable = bal_at_pay - payout_floor
    payout_amt = np.clip(withdrawable, 0.0, MAX_PAYOUT)
    # require at least $500 withdrawable
    has_payout &= withdrawable >= 500.0

    return np.where(has_payout, payout_amt, 0.0), any_blown & ~has_payout


def analyze_rescue(
    csv_paths: list[str],
    current_balance: float,
    trailing_stop_floor: float,
    n_sims: int = 10_000,
    max_days: int = 90,
    seed: Optional[int] = None,
) -> list[dict]:
    """What-If / Rescue Analysis.

    For each strategy CSV, runs ``n_sims`` Monte-Carlo paths starting from the
    user's current mid-account state (``current_balance`` and
    ``trailing_stop_floor``) and returns the pass/blow/timeout probabilities
    plus expected payout, sorted best-first by ``pass_probability``.

    Parameters
    ----------
    csv_paths            : list of absolute paths to strategy CSVs
    current_balance      : current account equity (e.g. 49_800)
    trailing_stop_floor  : current trailing-stop floor (e.g. 47_800)
    n_sims               : number of Monte-Carlo simulations per strategy
    max_days             : day budget per simulation
    seed                 : optional RNG seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Derive current peak from floor position
    if trailing_stop_floor < TRAIL_STOP_LEVEL:
        current_peak = trailing_stop_floor + TRAILING_DD
    else:
        current_peak = TRAIL_STOP_LEVEL + TRAILING_DD

    distance_to_stop = round(float(current_balance) - float(trailing_stop_floor), 2)

    results: list[dict] = []
    for csv_path in csv_paths:
        try:
            pool = _load_daily_pnl(csv_path)
            payout_amts, is_blow = _rescue_sim_vec(
                pool, n_sims,
                float(current_balance), float(current_peak),
                max_days,
            )
            n_pay = int((payout_amts > 0).sum())
            n_blow = int(is_blow.sum())
            n_time = n_sims - n_pay - n_blow
            results.append({
                "csv":                  csv_path,
                "n_sims":              n_sims,
                "pass_probability":    round(n_pay / n_sims, 4),
                "blow_probability":    round(n_blow / n_sims, 4),
                "timeout_probability": round(n_time / n_sims, 4),
                "mean_payout":         round(
                    float(payout_amts[payout_amts > 0].mean()) if n_pay > 0 else 0.0, 2
                ),
                "distance_to_stop":    distance_to_stop,
                "current_balance":     float(current_balance),
                "trailing_stop_floor": float(trailing_stop_floor),
            })
        except Exception as exc:
            results.append({"csv": csv_path, "error": str(exc)})

    results.sort(key=lambda r: r.get("pass_probability", -1), reverse=True)
    return results
