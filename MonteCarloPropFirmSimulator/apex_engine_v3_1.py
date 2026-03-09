"""
apex_engine_v3_1.py
===================
Version 3.1 — adds recency-weighted Monte Carlo sampling on top of v3.

All existing simulation mechanics are completely unchanged:
  - Daily PnL sampling in dollars (no balance-multiplication compounding)
  - Trailing drawdown with lock-out at TRAIL_STOP_LEVEL
  - Daily profit cap and daily loss limit
  - MIN_DAYS, MIN_GREEN_DAYS, GREEN_DAY_MIN requirements
  - 30 % single-day concentration rule
  - Payout floor and MAX_PAYOUT cap

New in v3.1
-----------
Sampling mode is now configurable via the `mode` parameter:

  "uniform"          — identical to v3 (np.random.choice with equal probability)
  "recency_weighted" — recent trading days are more likely to be sampled,
                       using exponential weights over the chronological array
  "recent_only"      — sample only from the last `recent_window` trading days
                       with uniform probability within that window

Helper
------
  compute_sampling_weights(daily_pnl, mode, weight_strength, recent_window)
      Returns (sample_pool, weights_or_None) pre-computed before each path.
      Called once at the start of simulate_path, before the day-loop,
      so the weights are constant across all days within one simulation run.

Tuning guide for weight_strength
---------------------------------
  weight_i = exp(weight_strength × (i / N))

  weight_strength =  0  → uniform (identical to mode="uniform")
  weight_strength =  3  → moderate recency bias (~20× more weight on last day
                           vs first day over a 100-day history)
  weight_strength =  6  → strong recency bias (~400× more weight on last vs first)
  weight_strength = 10  → extreme recency bias; effectively only the last few
                           days matter

  Normalised weights always sum to 1, so np.random.choice p= is always valid.

Public API (unchanged from v3 except new optional parameters)
--------------------------------------------------------------
  load_daily_pnl(csv_path)                          → np.ndarray (chronological)
  compute_sampling_weights(daily_pnl, mode, ...)    → (pool, weights|None)
  compute_diagnostics(daily_pnl)                    → diagnostic dict
  simulate_path(daily_pnl, ..., mode, weight_strength, recent_window)
  run_simulations(n_sims, daily_pnl, ..., mode, weight_strength, recent_window)
"""

import math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# APEX ACCOUNT CONSTANTS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

ACCOUNT_SIZE       = 50_000.0
TRAILING_DD        = 2_500.0
TRAIL_STOP_LEVEL   = 50_100.0
PAYOUT_THRESHOLD   = 52_600.0

DAILY_LOSS_LIMIT   = -700.0
DAILY_PROFIT_CAP   =  1_050.0

MIN_DAYS           = 8
MIN_GREEN_DAYS     = 5
GREEN_DAY_MIN      = 50.0

MAX_PAYOUT         = 2_000.0

_D = TRAILING_DD
_T = PAYOUT_THRESHOLD - ACCOUNT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_pnl(csv_path: str) -> np.ndarray:
    """
    Load a trade-history CSV and return a **chronologically ordered** 1-D array
    of daily aggregate PnL values in USD.

    Order matters for recency weighting: index 0 = oldest day, index -1 = newest.

    Expected columns (whitespace-stripped):
      Type            — "Trade" (primary) or "Exit …" (TradingView fallback)
      Date and time   — trade timestamp
      Net P&L USD     — per-trade profit/loss

    groupby("Date") sorts dates ascending by default, preserving chronology.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    mask = df["Type"].str.strip() == "Trade"
    if mask.sum() == 0:
        mask = df["Type"].str.strip().str.startswith("Exit")
        if mask.sum() == 0:
            raise ValueError(
                f"No 'Trade' or 'Exit' rows found in '{csv_path}'.\n"
                "Ensure the CSV contains a 'Type' column with trade rows."
            )
        print("[load_daily_pnl] NOTE: No 'Trade' rows found; using 'Exit' rows instead.")

    df = df[mask].copy()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    df["Date"]          = df["Date and time"].dt.date

    daily     = df.groupby("Date")["Net P&L USD"].sum()   # ascending date order
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
# RECENCY WEIGHTING HELPER  (new in v3.1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sampling_weights(
    daily_pnl: np.ndarray,
    mode: str = "uniform",
    weight_strength: float = 3.0,
    recent_window: int = 50,
) -> tuple:
    """
    Pre-compute the (sample_pool, weights) pair used by simulate_path.

    This is called ONCE per simulation path, before the day-loop, so weights
    are constant across all simulated days within one path.

    Parameters
    ----------
    daily_pnl       : np.ndarray  — full chronological history (oldest → newest)
    mode            : str
        "uniform"          — equal probability across all days (v3 default)
        "recency_weighted" — exponential weight favouring recent days
        "recent_only"      — uniform sample from last `recent_window` days only
    weight_strength : float
        Controls how steeply the exponential rises toward recent days.
        Only used when mode == "recency_weighted".
        Recommended range: 3 (moderate) … 6 (strong).
        weight_i = exp(weight_strength × i/N), then normalised to sum = 1.
        At weight_strength=0 this degenerates to uniform.
    recent_window   : int
        Number of most-recent trading days to restrict sampling to.
        Only used when mode == "recent_only".

    Returns
    -------
    (pool, weights)
        pool    : np.ndarray — the array to pass to np.random.choice
        weights : np.ndarray | None — probability weights; None = uniform
    """
    if mode == "uniform":
        # No weighting — standard v3 behaviour
        return daily_pnl, None

    elif mode == "recency_weighted":
        # Exponential weighting over the full history.
        # Normalised index: 0.0 for oldest day, 1.0 for most recent day.
        # weight_i = exp(weight_strength × normalised_index)
        # Higher weight_strength → more probability mass on recent days.
        N       = len(daily_pnl)
        indices = np.arange(N, dtype=float)
        weights = np.exp(weight_strength * (indices / (N - 1)) if N > 1 else indices)
        weights = weights / weights.sum()   # normalise to a valid probability vector
        return daily_pnl, weights

    elif mode == "recent_only":
        # Restrict the sample pool to the last `recent_window` trading days.
        # Uniform probability within that window.
        window = min(recent_window, len(daily_pnl))
        pool   = daily_pnl[-window:]        # most recent `window` days
        return pool, None

    else:
        raise ValueError(
            f"Unknown sampling mode '{mode}'. "
            "Choose 'uniform', 'recency_weighted', or 'recent_only'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(daily_pnl: np.ndarray) -> dict:
    """
    Compute and print statistical diagnostics from the empirical daily PnL.
    Monte Carlo remains the primary evaluation method; these are informational.

    Metrics: μ, σ, variance, drift/variance ratio, PropScore, Kelly f*,
    closed-form gambler's-ruin pass probability approximation.
    """
    mu       = float(np.mean(daily_pnl))
    sigma    = float(np.std(daily_pnl, ddof=1))
    variance = sigma ** 2

    D = _D
    T = _T

    drift_variance_ratio = mu / variance if variance > 0 else float("nan")
    prop_score           = (mu / sigma) * math.sqrt(D / T) if sigma > 0 else float("nan")
    f_star               = mu / variance if variance > 0 else float("nan")
    recommended_f        = 0.20 * f_star if not math.isnan(f_star) else float("nan")

    if mu != 0 and variance > 0:
        arg_num = -2.0 * mu * D / variance
        arg_den = -2.0 * mu * (D + T) / variance
        try:
            p_pass = (1.0 - math.exp(arg_num)) / (1.0 - math.exp(arg_den))
        except OverflowError:
            p_pass = float("nan")
    else:
        p_pass = float("nan")

    diag = {
        "mu": mu, "sigma": sigma, "variance": variance,
        "drift_variance_ratio": drift_variance_ratio,
        "distance_to_failure": D, "distance_to_target": T,
        "prop_score": prop_score, "f_star": f_star,
        "recommended_f": recommended_f, "p_pass_closed_form": p_pass,
    }

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
    if math.isnan(prop_score):
        print("  Rating                  :  N/A"); return
    if prop_score >= 0.50:   label = "STRONG EDGE"
    elif prop_score >= 0.25: label = "MODERATE EDGE"
    elif prop_score >= 0.10: label = "WEAK EDGE"
    elif prop_score > 0:     label = "MARGINAL"
    else:                    label = "NEGATIVE DRIFT — review strategy"
    print(f"  Rating                  :  {label}")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PATH SIMULATION  (new: mode / weight_strength / recent_window params)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_path(
    daily_pnl: np.ndarray,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
    # ── v3.1 sampling mode ───────────────────────────────────────────────────
    mode: str = "uniform",
    weight_strength: float = 3.0,
    recent_window: int = 50,
) -> dict:
    """
    Simulate a single Apex account equity path at the daily level.

    Sampling modes (new in v3.1)
    ----------------------------
    mode="uniform"          identical to v3 — np.random.choice(daily_pnl)
    mode="recency_weighted" recent days have exponentially higher probability
                            governed by weight_strength (see compute_sampling_weights)
    mode="recent_only"      sample only from the last `recent_window` days

    All other mechanics (trailing DD, caps, payout rules) are unchanged.

    Returns
    -------
    dict: outcome, balance, days, payout, equity_path
    """
    # ── Pre-compute sampling pool and weights ONCE per path ──────────────────
    # This is done before the day-loop so the same weights apply throughout
    # the entire path (they're resampled fresh for each new simulation run).
    pool, weights = compute_sampling_weights(
        daily_pnl, mode=mode, weight_strength=weight_strength,
        recent_window=recent_window,
    )

    # ── Account state ─────────────────────────────────────────────────────────
    balance       = ACCOUNT_SIZE
    peak          = ACCOUNT_SIZE
    trading_days  = 0
    green_days    = 0
    max_day_pnl   = 0.0
    total_profit  = 0.0
    payout_amount = 0.0

    equity_path = [balance]

    for _day in range(max_days):

        # ── Sample daily PnL (mode-aware) ─────────────────────────────────────
        # weights is None for uniform modes → np.random.choice uses equal probability
        raw_pnl = float(np.random.choice(pool, p=weights)) * risk_multiplier

        # ── Apply Apex daily caps (unchanged) ─────────────────────────────────
        pnl = max(DAILY_LOSS_LIMIT, min(DAILY_PROFIT_CAP, raw_pnl))

        balance += pnl
        equity_path.append(balance)
        trading_days += 1

        # ── Update peak ────────────────────────────────────────────────────────
        if balance > peak:
            peak = balance

        # ── Apex trailing drawdown floor (unchanged) ───────────────────────────
        if peak - TRAILING_DD < TRAIL_STOP_LEVEL:
            trailing_floor = peak - TRAILING_DD
        else:
            trailing_floor = TRAIL_STOP_LEVEL

        # ── Green-day tracking (unchanged) ────────────────────────────────────
        if pnl >= GREEN_DAY_MIN:
            green_days += 1

        # ── 30 % concentration tracking (unchanged) ────────────────────────────
        if pnl > max_day_pnl:
            max_day_pnl = pnl

        # ── Blow-up check (unchanged) ──────────────────────────────────────────
        if balance < trailing_floor:
            return {
                "outcome":     "blow",
                "balance":     balance,
                "days":        trading_days,
                "payout":      payout_amount,
                "equity_path": equity_path,
            }

        # ── Payout eligibility (unchanged) ────────────────────────────────────
        if balance >= PAYOUT_THRESHOLD:
            total_profit = balance - ACCOUNT_SIZE
            eligible = (
                trading_days >= MIN_DAYS
                and green_days >= MIN_GREEN_DAYS
                and total_profit > 0
                and max_day_pnl <= 0.30 * total_profit
            )
            if eligible:
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

    return {
        "outcome":     "timeout",
        "balance":     balance,
        "days":        trading_days,
        "payout":      payout_amount,
        "equity_path": equity_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO RUNNER  (new: mode / weight_strength / recent_window params)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulations(
    n_sims: int,
    daily_pnl: np.ndarray,
    risk_multiplier: float = 1.0,
    max_days: int = 90,
    stop_at_payout: bool = True,
    seed: int | None = None,
    n_paths: int = 50,
    # ── v3.1 sampling mode ───────────────────────────────────────────────────
    mode: str = "uniform",
    weight_strength: float = 3.0,
    recent_window: int = 50,
) -> dict:
    """
    Run n_sims Monte Carlo paths and return aggregate statistics.

    Sampling mode parameters (new in v3.1)
    ---------------------------------------
    mode            : "uniform" | "recency_weighted" | "recent_only"
    weight_strength : float — exponential steepness for "recency_weighted"
    recent_window   : int   — lookback window size for "recent_only"

    All other parameters and returned keys are identical to v3.
    """
    if seed is not None:
        np.random.seed(seed)

    outcomes     = []
    balances     = []
    payouts      = []
    days_list    = []
    equity_paths = []

    for i in range(n_sims):
        result = simulate_path(
            daily_pnl,
            risk_multiplier = risk_multiplier,
            max_days        = max_days,
            stop_at_payout  = stop_at_payout,
            mode            = mode,
            weight_strength = weight_strength,
            recent_window   = recent_window,
        )
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
