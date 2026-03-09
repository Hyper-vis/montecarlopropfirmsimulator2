"""
portfolio_optimizer.py
======================
Find the optimal allocation of N independent Apex accounts across a set of
trading strategies.

Model
-----
Each strategy runs on its own dedicated Apex account — no blending.
A "portfolio" is an allocation of a fixed total number of accounts across
the available strategies.

Example (10 accounts, 2 strategies):
  allocation (7, 3) → Strategy A gets 7 accounts, Strategy B gets 3.
Each of the 7 + 3 = 10 accounts is simulated independently using the
`simulate_path` function from apex_engine_v3_1.  Per-trial metrics:
  • number of accounts that paid out this cycle
  • total dollar payout across all accounts

Workflow
--------
  1.  Load CSVs → empirical daily PnL arrays
  2.  Per-strategy trade stats + individual Monte Carlo
  3.  Pairwise correlation matrix (shown for reference)
  4.  Enumerate all strategy subsets (size 1 … max_strategies)
       × all account allocations for each subset
       (random sample when the space is too large to enumerate exactly)
  5.  Run independent-account Monte Carlo for every candidate allocation
  6.  Score and rank by composite metric; print top N portfolios

Portfolio metrics
-----------------
  • Expected number of payouts per cycle
  • Expected total payout ($)
  • P(≥ k payouts) for user-specified thresholds, e.g. 1, 5, 10

Composite score
---------------
  score = 0.60 × (expected_payouts / n_accounts)
        + 0.25 × (expected_total_payout / (MAX_PAYOUT × n_accounts))
        − 0.15 × avg_pairwise_correlation

Search-space management
-----------------------
  Small spaces  → exact enumeration of all compositions (ordered partitions)
  Large spaces  → random sample of MAX_RANDOM_ALLOCS allocations per subset

Dependencies
------------
  numpy, pandas  — standard
  apex_engine_v3_1  — project module (not modified)

Usage
-----
  python portfolio_optimizer.py
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd

from apex_engine_v3_1 import (
    MAX_PAYOUT,
    ACCOUNT_SIZE,
    TRAILING_DD,
    TRAIL_STOP_LEVEL,
    PAYOUT_THRESHOLD,
    DAILY_LOSS_LIMIT,
    DAILY_PROFIT_CAP,
    MIN_DAYS,
    MIN_GREEN_DAYS,
    GREEN_DAY_MIN,
    run_simulations,
    simulate_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

N_SIMS_INDIVIDUAL  = 5_000   # MC sims per strategy (individual solo analysis)
N_SIMS_PORTFOLIO   = 2_000   # MC trials per portfolio candidate
MAX_DAYS           = 21      # per-account path ceiling (trading days) — one calendar month
STOP_AT_PAYOUT     = True    # stop each account path at first qualifying payout
TOP_N_PORTFOLIOS   = 5       # ranked portfolios shown in summary

# Exact enumeration up to this many allocations per strategy-subset.
# Beyond this, MAX_RANDOM_ALLOCS random allocations are drawn instead.
# Increase these for more thorough search with 4-5+ strategies.
# Trade-off: total runtime = candidates × N_SIMS_PORTFOLIO × n_accounts.
# If too slow, lower N_SIMS_PORTFOLIO at the prompt instead of lowering this.
MAX_EXACT_ALLOCS   = 1_000
MAX_RANDOM_ALLOCS  = 1_000

# Default payout-count thresholds for P(≥ k) output
DEFAULT_K_THRESHOLDS = [1, 2, 3, 5, 10]

# Multi-stage tournament: (sims_per_candidate, survivors)
# survivors = float → keep that fraction of the field
# survivors = int   → keep that absolute number
# Last stage survivors should equal or exceed TOP_N_PORTFOLIOS.
# Total sim budget ≈ sum across stages of (survivors_in × sims × n_accounts).
DEFAULT_STAGES = [
    (100,   0.20),   # Stage 1 : quick filter  — keep top 20 %
    (500,   0.25),   # Stage 2 : mid filter    — keep top 25 % of survivors
    (2_000, 20),     # Stage 3 : deep filter   — keep top 20 absolute
    (5_000, 5),      # Stage 4 : stress test   — keep final top 5
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_strategy_pnl(csv_path: str) -> np.ndarray:
    """
    Load a strategy CSV and return a **chronologically ordered** 1-D array of
    daily aggregate PnL values in USD.

    Column detection (case-insensitive, whitespace-stripped):
      Primary   : 'pnl'
      Fallback  : 'net p&l usd'  (TradingView MultiBot export format)

    Row filtering (applied before aggregation when the column is present):
      'type' == 'Trade'  (identical logic to apex_engine_v3_1.load_daily_pnl)

    Date aggregation:
      If a 'date and time' column is present, PnL is summed per calendar day
      so the returned array represents daily PnL (consistent with the MC engine).
      If no date column is found, all rows are returned as-is.

    Parameters
    ----------
    csv_path : str — path to the CSV file

    Returns
    -------
    np.ndarray of float, shape (n_days,)  — chronological order

    Raises
    ------
    ValueError if no recognisable PnL column exists or the array is empty.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # ── Detect PnL column ─────────────────────────────────────────────────────
    if "pnl" in df.columns:
        pnl_col = "pnl"
    elif "net p&l usd" in df.columns:
        pnl_col = "net p&l usd"
    else:
        raise ValueError(
            f"No 'pnl' column found in '{csv_path}'. "
            f"Available columns: {list(df.columns)}"
        )

    # ── Filter to Trade rows only (if 'type' column exists) ──────────────────
    if "type" in df.columns:
        mask = df["type"].str.strip().str.lower() == "trade"
        if mask.sum() == 0:
            # TradingView fallback: rows beginning with "Exit"
            mask = df["type"].str.strip().str.startswith("Exit")
        if mask.sum() > 0:
            df = df[mask].copy()

    df = df.dropna(subset=[pnl_col])

    # ── Aggregate by calendar date if timestamp column is present ─────────────
    if "date and time" in df.columns:
        df["date and time"] = pd.to_datetime(df["date and time"])
        df["_date"]         = df["date and time"].dt.date
        daily               = df.groupby("_date")[pnl_col].sum().sort_index()
        pnl                 = daily.values.astype(float)
    else:
        pnl = df[pnl_col].values.astype(float)

    if len(pnl) == 0:
        raise ValueError(f"No valid PnL data found in '{csv_path}'.")

    return pnl


def discover_csv_files(folder: str = ".") -> list[str]:
    """Return sorted list of all CSV paths in the given folder (non-recursive)."""
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def _strategy_name(csv_path: str) -> str:
    """Extract a clean display name from a CSV file path (stem only)."""
    return os.path.splitext(os.path.basename(csv_path))[0]


# ─────────────────────────────────────────────────────────────────────────────
# TRADE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_trade_stats(pnl: np.ndarray) -> dict:
    """
    Compute descriptive trade statistics from a PnL array.

    Parameters
    ----------
    pnl : np.ndarray — per-trade PnL values (may contain wins and losses)

    Returns
    -------
    dict with keys:
        n_trades      : int
        win_rate      : float  [0, 1]
        avg_win       : float  (mean of positive trades; 0.0 if none)
        avg_loss      : float  (mean of negative trades, negative number; 0.0 if none)
        profit_factor : float  (gross wins / |gross losses|; inf if no losses)
        expectancy    : float  (mean PnL per trade)
    """
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    n          = len(pnl)
    win_rate   = len(wins)   / n if n > 0 else 0.0
    avg_win    = float(np.mean(wins))   if len(wins)   > 0 else 0.0
    avg_loss   = float(np.mean(losses)) if len(losses) > 0 else 0.0

    gross_wins   = float(np.sum(wins))
    gross_losses = float(np.sum(losses))

    if gross_losses < 0:
        profit_factor = gross_wins / abs(gross_losses)
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    expectancy = float(np.mean(pnl)) if n > 0 else 0.0

    return {
        "n_trades":      n,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "profit_factor": profit_factor,
        "expectancy":    expectancy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL MONTE CARLO (delegates to apex_engine_v3_1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mc_metrics(
    pnl: np.ndarray,
    n_sims: int = N_SIMS_INDIVIDUAL,
    max_days: int = MAX_DAYS,
    stop_at_payout: bool = STOP_AT_PAYOUT,
    *,
    label_days: int | None = None,
) -> dict:
    """
    Estimate payout probability, risk of ruin, and expected payout via Monte
    Carlo, using run_simulations from apex_engine_v3_1 (mode='uniform').

    label_days is stored in the returned dict for display purposes only;
    defaults to max_days when not supplied.

    Returns
    -------
    dict: payout_prob, risk_of_ruin, expected_payout, label_days
    """
    results = run_simulations(
        n_sims         = n_sims,
        daily_pnl      = pnl,
        max_days       = max_days,
        stop_at_payout = stop_at_payout,
        n_paths        = 0,     # equity curves not needed here
    )
    return {
        "payout_prob":     results["pass_rate"],
        "risk_of_ruin":    results["fail_rate"],
        "expected_payout": results["average_payout"],
        "label_days":      label_days if label_days is not None else max_days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation_matrix(
    strategy_pnl: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute the pairwise Pearson correlation matrix between all strategies.

    When two strategies have different numbers of trades, the comparison uses
    the most-recent N trades where N = min(len(a), len(b)), preserving the
    recency convention used elsewhere in the system.

    Pearson's r is computed directly with numpy (no scipy dependency).
    Pairs where either series has zero variance receive correlation = 0.0.

    Parameters
    ----------
    strategy_pnl : dict mapping strategy name → 1-D float np.ndarray

    Returns
    -------
    pd.DataFrame : symmetric N×N correlation matrix indexed by strategy name
    """
    names  = list(strategy_pnl.keys())
    n      = len(names)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            a      = strategy_pnl[names[i]]
            b      = strategy_pnl[names[j]]
            length = min(len(a), len(b))

            if length < 2:
                corr = 0.0
            else:
                va   = a[-length:]
                vb   = b[-length:]
                sa   = np.std(va)
                sb   = np.std(vb)
                if sa == 0.0 or sb == 0.0:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(va, vb)[0, 1])

            matrix[i, j] = corr
            matrix[j, i] = corr

    return pd.DataFrame(matrix, index=names, columns=names)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _compositions(n: int, k: int):
    """
    Generate all ordered k-tuples of positive integers that sum to n.
    Each strategy gets at least 1 account.
    """
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest


def _count_compositions(n: int, k: int) -> int:
    """C(n-1, k-1) — number of ordered k-tuples of positive integers summing to n."""
    if n < k:
        return 0
    num, den = 1, 1
    for i in range(k - 1):
        num *= (n - 1 - i)
        den *= (i + 1)
    return num // den


def _sample_compositions(
    n: int, k: int, count: int, rng: np.random.Generator
) -> list[tuple]:
    """
    Draw `count` distinct random ordered k-tuples of positive integers summing to n
    using the random cut-point (stars-and-bars) method.
    """
    seen, result = set(), []
    max_attempts = count * 20
    for _ in range(max_attempts):
        if len(result) >= count:
            break
        if n - 1 < k - 1:
            break
        cuts = tuple(sorted(rng.choice(n - 1, size=k - 1, replace=False) + 1))
        tup  = tuple(
            cuts[0] if i == 0
            else (cuts[i] - cuts[i - 1] if i < k - 1 else n - cuts[-1])
            for i in range(k)
        )
        if tup not in seen and all(x >= 1 for x in tup):
            seen.add(tup)
            result.append(tup)
    return result


def get_allocations(
    n_accounts: int,
    k_strategies: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """
    Return allocations (ordered k-tuples summing to n_accounts) to evaluate.
    Uses exact enumeration for small spaces; random sampling for large ones.
    """
    if n_accounts < k_strategies:
        return []
    total = _count_compositions(n_accounts, k_strategies)
    if total <= MAX_EXACT_ALLOCS:
        return list(_compositions(n_accounts, k_strategies))
    return _sample_compositions(n_accounts, k_strategies, MAX_RANDOM_ALLOCS, rng)


# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENT PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_portfolio_trial(
    pools: list[np.ndarray],
    allocation: tuple[int, ...],
    max_days: int,
    stop_at_payout: bool,
) -> tuple[int, float]:
    """
    Run one portfolio trial: simulate each account independently.

    Parameters
    ----------
    pools      : empirical PnL pools, one per strategy
    allocation : number of accounts assigned to each strategy
    max_days   : path ceiling per account
    stop_at_payout : stop each account on first qualifying payout

    Returns
    -------
    (n_payouts, total_payout_amount)
    """
    n_payouts    = 0
    total_payout = 0.0
    for pool, count in zip(pools, allocation):
        for _ in range(count):
            result = simulate_path(
                pool,
                max_days       = max_days,
                stop_at_payout = stop_at_payout,
            )
            if result["payout"] > 0:
                n_payouts    += 1
                total_payout += result["payout"]
    return n_payouts, total_payout


def simulate_portfolio_mc(
    pools: list[np.ndarray],
    allocation: tuple[int, ...],
    n_sims: int = N_SIMS_PORTFOLIO,
    max_days: int = MAX_DAYS,
    stop_at_payout: bool = STOP_AT_PAYOUT,
    k_thresholds: list[int] | None = None,
) -> dict:
    """
    Run n_sims independent portfolio trials and return aggregate statistics.

    Returns
    -------
    dict:
        n_accounts            : total accounts in this portfolio
        expected_payouts      : mean payout-count per cycle
        expected_total_payout : mean total payout $ per cycle
        payout_count_dist     : {k: P(payouts >= k)} for each k in k_thresholds
    """
    if k_thresholds is None:
        k_thresholds = DEFAULT_K_THRESHOLDS
    n_accounts  = sum(allocation)
    pay_counts  = np.empty(n_sims, dtype=int)
    pay_amounts = np.empty(n_sims)

    for i in range(n_sims):
        c, a            = _simulate_portfolio_trial(pools, allocation, max_days, stop_at_payout)
        pay_counts[i]  = c
        pay_amounts[i] = a

    dist = {k: float(np.mean(pay_counts >= k)) for k in k_thresholds if k <= n_accounts}
    if 1 not in dist:
        dist[1] = float(np.mean(pay_counts >= 1))

    return {
        "n_accounts":            n_accounts,
        "expected_payouts":      float(np.mean(pay_counts)),
        "expected_total_payout": float(np.mean(pay_amounts)),
        "payout_count_dist":     dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED MC  (Option C — 50-200× faster than simulate_portfolio_mc)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_account_batch_vec(
    pool: np.ndarray,
    n_sims: int,
    max_days: int,
    stop_at_payout: bool,
) -> np.ndarray:
    """
    Vectorised simulation of n_sims independent paths for a single account.
    All Apex mechanics are reproduced in numpy: trailing drawdown, daily caps,
    green-day minimum, 30 % concentration rule, payout floor, MAX_PAYOUT cap.

    Assumes stop_at_payout=True (first eligible payout ends the path).
    Falls back to the sequential simulate_path when stop_at_payout=False.

    Returns
    -------
    np.ndarray of shape (n_sims,) — payout amount per sim (0.0 if no payout).
    """
    if not stop_at_payout:
        # Vectorised path assumes single-payout logic; fall back for safety.
        out = np.empty(n_sims, dtype=np.float64)
        for i in range(n_sims):
            out[i] = simulate_path(pool, max_days=max_days, stop_at_payout=False)["payout"]
        return out

    # ── Pre-sample all day PnLs: shape (n_sims, max_days) ────────────────────
    raw = np.random.choice(pool, size=(n_sims, max_days), replace=True)
    # Apply Apex daily caps
    pnl = np.clip(raw, DAILY_LOSS_LIMIT, DAILY_PROFIT_CAP)       # (n_sims, max_days)

    # ── Running balance: shape (n_sims, max_days+1) ───────────────────────────
    # balance[:, 0] = ACCOUNT_SIZE (before any trading day)
    # balance[:, t] = balance after t trading days
    balance = np.empty((n_sims, max_days + 1), dtype=np.float64)
    balance[:, 0]  = ACCOUNT_SIZE
    balance[:, 1:] = ACCOUNT_SIZE + np.cumsum(pnl, axis=1)

    # ── Running peak (for trailing drawdown) ──────────────────────────────────
    peak = np.maximum.accumulate(balance, axis=1)                 # (n_sims, max_days+1)

    # ── Trailing drawdown floor ────────────────────────────────────────────────
    # Sequential logic: if peak - TRAILING_DD < TRAIL_STOP_LEVEL → floor = peak - TRAILING_DD
    #                   else                                      → floor = TRAIL_STOP_LEVEL
    # This is equivalent to: floor = min(peak - TRAILING_DD, TRAIL_STOP_LEVEL)
    trailing_floor = np.minimum(peak - TRAILING_DD, TRAIL_STOP_LEVEL)  # (n_sims, max_days+1)

    # ── Blow detection: balance < floor, for trading days 1..max_days ─────────
    blown    = balance[:, 1:] < trailing_floor[:, 1:]             # (n_sims, max_days)
    any_blown = blown.any(axis=1)                                 # (n_sims,)
    # First blow day (0-indexed in pnl array; trading day = idx+1)
    # If never blown, sentinel value = max_days (one past last valid index).
    blow_day = np.where(any_blown, np.argmax(blown, axis=1), max_days)

    # ── Green-day count (cumulative per sim) ──────────────────────────────────
    green_cum = np.cumsum(pnl >= GREEN_DAY_MIN, axis=1)          # (n_sims, max_days)

    # ── Max single-day PnL seen so far (rolling max, starts at 0.0) ──────────
    # Sequential code initialises max_day_pnl = 0.0 and only raises it on positive days.
    max_day_pnl_cum = np.maximum.accumulate(np.maximum(pnl, 0.0), axis=1)  # (n_sims, max_days)

    # ── Total profit at each day ───────────────────────────────────────────────
    total_profit = balance[:, 1:] - ACCOUNT_SIZE                 # (n_sims, max_days)

    # ── Payout eligibility (all five Apex conditions reproduced) ─────────────
    day_counts = np.arange(1, max_days + 1, dtype=np.int32)      # trading day number (1-indexed)
    # 30 % concentration rule: max single-day PnL ≤ 30 % of total profit.
    # Rule is vacuous when total_profit ≤ 0 (payout can't trigger anyway).
    conc_ok = (
        (total_profit <= 0) |
        (max_day_pnl_cum <= 0.30 * np.maximum(total_profit, 1e-9))
    )
    eligible = (
        (day_counts[np.newaxis, :] >= MIN_DAYS) &    # ≥ 8 trading days
        (green_cum        >= MIN_GREEN_DAYS)    &    # ≥ 5 green days
        (total_profit     >  0)                 &    # net positive
        conc_ok                                 &    # 30 % concentration rule
        (balance[:, 1:]   >= PAYOUT_THRESHOLD)       # hit the payout threshold
    )  # (n_sims, max_days)

    # ── First eligible payout day ─────────────────────────────────────────────
    any_eligible = eligible.any(axis=1)               # (n_sims,)
    # 0-indexed in pnl array; trading day = payout_day + 1.
    # Sentinel = max_days if no eligible day exists.
    payout_day = np.where(any_eligible, np.argmax(eligible, axis=1), max_days)

    # Blow is checked before payout within each day (sequential code order),
    # so blow wins on a tie.  Payout only wins if it comes strictly before blow.
    has_payout = any_eligible & (payout_day < blow_day)

    # ── Payout amount ─────────────────────────────────────────────────────────
    payout_floor = PAYOUT_THRESHOLD - 500.0
    sim_idx      = np.arange(n_sims)
    safe_pd      = np.clip(payout_day, 0, max_days - 1)          # guard against sentinel
    bal_at_pay   = balance[sim_idx, safe_pd + 1]                  # balance after that day
    withdrawable = bal_at_pay - payout_floor
    payout_amt   = np.clip(withdrawable, 0.0, MAX_PAYOUT)
    has_payout  &= (withdrawable >= 500.0)                        # minimum $500 withdrawal

    return np.where(has_payout, payout_amt, 0.0)


def simulate_portfolio_mc_vec(
    pools: list[np.ndarray],
    allocation: tuple[int, ...],
    n_sims: int = N_SIMS_PORTFOLIO,
    max_days: int = MAX_DAYS,
    stop_at_payout: bool = STOP_AT_PAYOUT,
    k_thresholds: list[int] | None = None,
) -> dict:
    """
    Vectorised drop-in replacement for simulate_portfolio_mc.

    Uses batch numpy operations instead of a Python-level day-loop per sim,
    giving 50–200× speedup on typical n_sims values.  The return dict has
    the same keys and semantics as simulate_portfolio_mc.
    """
    if k_thresholds is None:
        k_thresholds = DEFAULT_K_THRESHOLDS

    n_accounts  = sum(allocation)
    pay_counts  = np.zeros(n_sims, dtype=np.int32)
    pay_amounts = np.zeros(n_sims, dtype=np.float64)

    for pool, count in zip(pools, allocation):
        for _ in range(count):
            payouts      = _simulate_account_batch_vec(pool, n_sims, max_days, stop_at_payout)
            pay_counts  += (payouts > 0).astype(np.int32)
            pay_amounts += payouts

    dist = {k: float(np.mean(pay_counts >= k)) for k in k_thresholds if k <= n_accounts}
    if 1 not in dist:
        dist[1] = float(np.mean(pay_counts >= 1))

    return {
        "n_accounts":            n_accounts,
        "expected_payouts":      float(np.mean(pay_counts)),
        "expected_total_payout": float(np.mean(pay_amounts)),
        "payout_count_dist":     dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pairwise_correlation(
    names: list[str],
    corr_matrix: pd.DataFrame,
) -> float:
    """Mean of upper-triangle off-diagonal correlations for the given name subset."""
    if len(names) < 2:
        return 0.0
    sub   = corr_matrix.loc[names, names].values
    n     = len(names)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([sub[i, j] for i, j in pairs]))


def _score_portfolio(mc: dict, avg_corr: float) -> float:
    """
    Composite score (higher = better):

      score = 0.60 × (expected_payouts / n_accounts)
            + 0.25 × (expected_total_payout / (MAX_PAYOUT × n_accounts))
            − 0.15 × avg_pairwise_correlation
    """
    n  = mc["n_accounts"]
    hr = mc["expected_payouts"]      / n               if n > 0 else 0.0
    dr = mc["expected_total_payout"] / (MAX_PAYOUT * n) if n > 0 else 0.0
    return 0.60 * hr + 0.25 * dr - 0.15 * avg_corr


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL PRE-FILTER + ADAPTIVE STAGES  (Options A & B)
# ─────────────────────────────────────────────────────────────────────────────

def _analytical_score(
    pools: list[np.ndarray],
    allocation: tuple[int, ...],
    avg_corr: float,
) -> float:
    """
    Instant proxy score for pre-filtering tournament candidates (no simulation).

    Computes a per-account weighted Sharpe-like metric, penalised by the
    average pairwise correlation.  Runs in microseconds per candidate —
    used to discard the bottom 50 % of the field before any MC simulation.

    Higher is better; negative values indicate negative-drift strategies.
    """
    n_total = sum(allocation)
    if n_total == 0:
        return -999.0
    weighted_mu    = sum(a * float(np.mean(p))  for a, p in zip(allocation, pools)) / n_total
    weighted_sigma = sum(a * float(np.std(p))   for a, p in zip(allocation, pools)) / n_total
    sharpe = weighted_mu / (weighted_sigma + 1e-9)
    return sharpe * (1.0 - 0.3 * max(0.0, avg_corr))


def _auto_stages(
    n_candidates: int,
    n_accounts: int,
    top_n: int = TOP_N_PORTFOLIOS,
    target_budget: int = 8_000_000,
) -> list[tuple]:
    """
    Auto-generate tournament stages scaled to the candidate count and a
    soft overall account-path-simulation budget, so the tournament is fast
    regardless of how large the search space is.

    Sim counts grow geometrically across stages while survival fractions
    shrink the field, keeping total cost ≤ target_budget without manual tuning.

    Parameters
    ----------
    n_candidates  : total candidates entering the tournament
    n_accounts    : accounts per portfolio (cost scales linearly with this)
    top_n         : minimum survivors guaranteed through every stage
    target_budget : soft cap on total account-path simulations

    Returns
    -------
    list of (n_sims, survivors) tuples compatible with find_optimal_portfolios.
    Falls back to DEFAULT_STAGES if the generated list would be empty.
    """
    # Geometric sim-count ladder and matching keep fractions / absolute counts.
    sim_ladder  = [50,   200,  800,  3_000, 10_000]
    keep_ladder = [0.20, 0.20, 0.20, top_n, top_n]

    stages: list[tuple] = []
    alive     = n_candidates
    remaining = target_budget

    for sims, keep in zip(sim_ladder, keep_ladder):
        if alive <= max(top_n, 2):
            break
        # Scale sims down to respect the remaining budget (never below 25).
        stage_cost = alive * sims * max(1, n_accounts)
        if len(stages) > 0 and stage_cost > remaining:
            sims = max(25, int(remaining // max(1, alive * n_accounts)))
        sims = int(sims)
        if sims < 20:
            break
        stages.append((sims, keep))
        # Advance field-size estimate for next iteration.
        if isinstance(keep, float):
            alive = max(top_n, int(alive * keep))
        else:
            alive = max(top_n, min(keep, alive))
        remaining -= alive * sims * max(1, n_accounts)   # rough running subtraction

    return stages if stages else DEFAULT_STAGES


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# STAGED TOURNAMENT SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_stage_budget(
    stages: list[tuple],
    n_candidates: int,
    n_accounts: int,
) -> list[tuple]:
    """
    Return (survivors_in, sims, survivors_out, path_sims) for each stage,
    given the initial candidate count.
    """
    rows = []
    alive = n_candidates
    for sims, keep in stages:
        path_sims = alive * sims * n_accounts
        if isinstance(keep, float):
            out = max(1, int(alive * keep))
        else:
            out = max(1, min(keep, alive))
        rows.append((alive, sims, out, path_sims))
        alive = out
        if alive <= 1:
            break
    return rows


def find_optimal_portfolios(
    strategy_pnl: dict[str, np.ndarray],
    corr_matrix: pd.DataFrame,
    n_accounts: int,
    max_strategies: int = 3,
    stages: list[tuple] | None = None,
    max_days: int = MAX_DAYS,
    stop_at_payout: bool = STOP_AT_PAYOUT,
    k_thresholds: list[int] | None = None,
    top_n: int = TOP_N_PORTFOLIOS,
    seed: int = 42,
    progress_cb=None,
) -> list[dict]:
    """
    Multi-stage tournament search over all strategy-subset × allocation candidates.

    Stage flow
    ----------
    Each stage runs MC simulations on the surviving candidates and eliminates
    the bottom performers, passing a smaller pool to the next stage.  The final
    stage stress-tests the remaining candidates with the highest sim count and
    returns the top_n by expected total payout.

    stages : list of (n_sims, survivors)
        n_sims    : int   — MC trials per candidate in this stage
        survivors : float → keep that fraction of the field (e.g. 0.20 = top 20 %)
                    int   → keep that absolute number (e.g. 20)
    """
    import time

    _stages_override = stages   # None → auto-generate after candidate count is known

    names = list(strategy_pnl.keys())
    rng   = np.random.default_rng(seed)

    # ── Build full candidate pool ─────────────────────────────────────────────
    candidates = []
    for k in range(1, min(max_strategies, len(names)) + 1):
        for combo in itertools.combinations(names, k):
            for alloc in get_allocations(n_accounts, k, rng):
                candidates.append((list(combo), alloc))

    total = len(candidates)
    if _stages_override is None:
        stages = _auto_stages(total, n_accounts, top_n)   # Option B: adaptive stages
    else:
        stages = _stages_override

    # ── Print budget estimate ─────────────────────────────────────────────────
    budget = _estimate_stage_budget(stages, total, n_accounts)
    total_paths = sum(r[3] for r in budget)
    print(f"\n  {'─'*56}")
    print(f"  TOURNAMENT SEARCH  —  {total:,} candidates")
    print(f"  {'─'*56}")
    print(f"  {'Stage':<8} {'Field':>8} {'Sims':>8} {'Survivors':>10} {'Path-sims':>12}")
    print(f"  {'─'*56}")
    for i, (alive, sims, out, ps) in enumerate(budget, 1):
        print(f"  Stage {i:<3} {alive:>8,} {sims:>8,} {out:>10,} {ps:>12,}")
    print(f"  {'─'*56}")
    print(f"  Total account-path sims : {total_paths:>12,}")
    print(f"  {'─'*56}\n")

    # ── Pre-compute static per-candidate fields ───────────────────────────────
    pool_cache = {
        name: strategy_pnl[name] for name in names
    }
    candidate_meta = [
        {
            "strategies":      combo_names,
            "allocation":      alloc,
            "avg_correlation": _mean_pairwise_correlation(combo_names, corr_matrix),
            "pools":           [pool_cache[n] for n in combo_names],
        }
        for combo_names, alloc in candidates
    ]

    # ── Option A: Analytical pre-filter — discard bottom 50 % before any MC ──
    if len(candidate_meta) > 100:
        for cand in candidate_meta:
            cand["_ascore"] = _analytical_score(
                cand["pools"], cand["allocation"], cand["avg_correlation"]
            )
        candidate_meta.sort(key=lambda c: c["_ascore"], reverse=True)
        n_pre = max(top_n, len(candidate_meta) // 2)
        candidate_meta = candidate_meta[:n_pre]
        print(f"  Analytical pre-filter : {total:,} → {n_pre:,} candidates (top 50 %)\n")

    # ── Run stages ────────────────────────────────────────────────────────────
    survivors = candidate_meta
    mc_results: dict[int, dict] = {}   # idx → latest mc dict

    for stage_num, (n_sims, keep) in enumerate(stages, 1):
        if len(survivors) <= max(1, top_n):
            print(f"  Stage {stage_num}: only {len(survivors)} candidate(s) remain — skipping.")
            break

        t0 = time.perf_counter()
        print(f"  Stage {stage_num}: {len(survivors):,} candidates × {n_sims:,} sims …")

        if progress_cb is not None:
            progress_cb({
                "type":       "stage",
                "stage":      stage_num,
                "n_stages":   len(stages),
                "candidates": len(survivors),
                "sims":       n_sims,
            })

        for i, cand in enumerate(survivors):
            mc = simulate_portfolio_mc_vec(
                cand["pools"], cand["allocation"],
                n_sims, max_days, stop_at_payout, k_thresholds,
            )
            mc_results[id(cand)] = mc
            if progress_cb is not None:
                progress_cb({
                    "type":     "progress",
                    "stage":    stage_num,
                    "n_stages": len(stages),
                    "completed": i + 1,
                    "total":    len(survivors),
                })
            if (i + 1) % max(1, len(survivors) // 20) == 0 or i + 1 == len(survivors):
                elapsed = time.perf_counter() - t0
                rate    = (i + 1) / elapsed if elapsed > 0 else 1
                remain  = (len(survivors) - i - 1) / rate
                print(
                    f"    {i+1:>5}/{len(survivors)}  "
                    f"elapsed {elapsed:>5.0f}s  "
                    f"ETA {remain:>5.0f}s",
                    end="\r",
                )

        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.0f}s{' ' * 30}")

        # Score and sort survivors by expected_total_payout, then composite score
        survivors.sort(
            key=lambda c: (
                mc_results[id(c)]["expected_total_payout"],
                _score_portfolio(mc_results[id(c)], c["avg_correlation"]),
            ),
            reverse=True,
        )

        # Eliminate bottom of field
        if isinstance(keep, float):
            n_keep = max(top_n, int(len(survivors) * keep))
        else:
            n_keep = max(top_n, keep)
        survivors = survivors[:n_keep]
        print(f"  → {len(survivors)} candidates advanced to next stage.\n")

    # ── Build final result list ───────────────────────────────────────────────
    # If all stages were skipped (candidate pool ≤ top_n from the start),
    # mc_results is empty — run one evaluation pass on the full set now.
    if not mc_results and survivors:
        final_sims = stages[-1][0] if stages else N_SIMS_PORTFOLIO
        print(
            f"  All stages skipped (small pool) — "
            f"evaluating {len(survivors)} candidate(s) with {final_sims:,} sims …"
        )
        for cand in survivors:
            mc_results[id(cand)] = simulate_portfolio_mc_vec(
                cand["pools"], cand["allocation"],
                final_sims, max_days, stop_at_payout, k_thresholds,
            )

    portfolios = []
    for cand in survivors:
        mc    = mc_results[id(cand)]
        score = _score_portfolio(mc, cand["avg_correlation"])
        portfolios.append({
            "strategies":      cand["strategies"],
            "allocation":      cand["allocation"],
            "avg_correlation": cand["avg_correlation"],
            "score":           score,
            **mc,
        })

    portfolios.sort(
        key=lambda x: (x["expected_total_payout"], x["score"]),
        reverse=True,
    )
    return portfolios[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _div(w: int = 56) -> str:
    return "═" * w


def print_strategy_analysis(name: str, stats: dict, mc: dict, width: int = 56):
    label_days = mc.get("label_days", MAX_DAYS)
    print(f"  {'─' * width}")
    print(f"  Strategy       : {name}")
    print(f"  Trading days   : {stats['n_trades']:>7,}")
    print(f"  Win rate       : {stats['win_rate']:>8.1%}")
    print(f"  Avg win day    : ${stats['avg_win']:>9,.2f}")
    print(f"  Avg loss day   : ${stats['avg_loss']:>9,.2f}")
    print(f"  Profit factor  : {stats['profit_factor']:>9.3f}")
    print(f"  Expectancy/day : ${stats['expectancy']:>9,.2f}")
    print(f"  P(payout)      : {mc['payout_prob']:>8.2%}   [1 account, {label_days}d cycle]")
    print(f"  Risk of ruin   : {mc['risk_of_ruin']:>8.2%}")
    print(f"  Avg payout     : ${mc['expected_payout']:>9,.2f}   [when paid]")


def print_portfolio_results(
    portfolios: list[dict],
    k_thresholds: list[int],
    cycle_days: int = MAX_DAYS,
    width: int = 56,
):
    print()
    print(_div(width))
    print(f"  RANKED PORTFOLIOS  —  {cycle_days}-day cycle  (ranked by expected total payout)")
    print(_div(width))
    print()
    print("  Score = 0.60×(hit rate) + 0.25×($ rate) − 0.15×(avg corr)")
    print()

    for rank, p in enumerate(portfolios, 1):
        alloc_str = "  +  ".join(
            f"{count}× {name}"
            for name, count in zip(p["strategies"], p["allocation"])
        )
        print(_div(width))
        print(f"  Portfolio #{rank}")
        print()
        print(f"  Allocation  :  {alloc_str}")
        print(f"  Total accts :  {p['n_accounts']}")
        print(f"  Cycle length:  {cycle_days} trading days")
        print(f"  Avg corr    :  {p['avg_correlation']:.3f}")
        print(f"  Risk score  :  {p['score']:+.4f}")
        print()
        print(f"  ★ Expected total payout / cycle : ${p['expected_total_payout']:>10,.2f}")
        print(f"    Expected payouts (# accounts)  : {p['expected_payouts']:>6.2f}")
        print()
        print("  Payout-count probabilities:")
        dist = p["payout_count_dist"]
        for k in sorted(dist.keys()):
            if k <= p["n_accounts"]:
                bar = "█" * int(dist[k] * 30)
                pad = "s" if k > 1 else " "
                print(f"    P(≥ {k:>2} payout{pad}) : {dist[k]:>6.2%}  {bar}")
        print()

    print(_div(width))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _ask_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def main():
    w = 56
    print()
    print(_div(w))
    print("  PORTFOLIO OPTIMIZER  —  Independent Account Allocation")
    print(_div(w))

    # ── 1. Discover CSVs ──────────────────────────────────────────────────────
    folder    = input("\n  Folder containing strategy CSVs [.]: ").strip() or "."
    csv_files = discover_csv_files(folder)

    if not csv_files:
        print(f"\n  No CSV files found in '{folder}'. Exiting.")
        return

    print(f"\n  Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"    {os.path.basename(f)}")

    use_all = input("\n  Include all files? (yes/no) [yes]: ").strip().lower()
    if use_all in ("no", "n"):
        selected = []
        for f in csv_files:
            ans = input(f"    Include '{os.path.basename(f)}'? (yes/no): ").strip().lower()
            if ans not in ("no", "n"):
                selected.append(f)
        csv_files = selected

    if not csv_files:
        print("\n  No files selected. Exiting.")
        return

    # ── 2. Parameters ─────────────────────────────────────────────────────────
    print()
    n_accounts   = _ask_int("  Total Apex accounts in portfolio [10]: ", 10)
    max_strats   = _ask_int("  Max strategies per portfolio (1–10) [3]: ", 3)
    max_strats   = max(1, min(max_strats, min(10, len(csv_files))))
    max_days     = _ask_int(f"  Cycle length in trading days [{MAX_DAYS}]: ", MAX_DAYS)
    top_n        = _ask_int(f"  Top N portfolios to display       [{TOP_N_PORTFOLIOS}]: ", TOP_N_PORTFOLIOS)

    n_sims_indiv = _ask_int(f"  MC sims per strategy (individual) [{N_SIMS_INDIVIDUAL}]: ", N_SIMS_INDIVIDUAL)

    raw_thresh = input(
        f"  Payout-count thresholds, comma-separated [{','.join(str(k) for k in DEFAULT_K_THRESHOLDS)}]: "
    ).strip()
    try:
        k_thresholds = [int(x) for x in raw_thresh.split(",") if x.strip()] or DEFAULT_K_THRESHOLDS
    except ValueError:
        k_thresholds = DEFAULT_K_THRESHOLDS
    k_thresholds = sorted(set(k_thresholds))

    # ── 3. Load and analyse each strategy ─────────────────────────────────────
    strategy_pnl:   dict[str, np.ndarray] = {}
    strategy_stats: dict[str, dict]       = {}
    individual_mc:  dict[str, dict]       = {}

    print()
    print(_div(w))
    print("  INDIVIDUAL STRATEGY ANALYSIS")
    print(_div(w))

    for path in csv_files:
        name = _strategy_name(path)
        try:
            pnl = load_strategy_pnl(path)
        except ValueError as exc:
            print(f"\n  [SKIP] {name}: {exc}")
            continue

        print(f"\n  Loading '{name}' … {len(pnl)} trading days")
        stats = compute_trade_stats(pnl)
        mc    = compute_mc_metrics(pnl, n_sims=n_sims_indiv, max_days=max_days)

        strategy_pnl[name]   = pnl
        strategy_stats[name] = stats
        individual_mc[name]  = mc

        print_strategy_analysis(name, stats, mc, width=w)

    if not strategy_pnl:
        print("\n  No valid strategies loaded. Exiting.")
        return

    # ── 4. Correlation matrix ─────────────────────────────────────────────────
    print()
    print(_div(w))
    print("  PAIRWISE CORRELATION  (daily PnL series)")
    print(_div(w))
    corr_df = compute_correlation_matrix(strategy_pnl)
    print()
    print(corr_df.round(3).to_string())
    print()

    # ── 5. Portfolio search ───────────────────────────────────────────────────
    print(_div(w))
    print("  PORTFOLIO OPTIMIZATION")
    print(f"  {n_accounts} total accounts  ·  up to {max_strats} strategies  ·  independent accounts")
    print(_div(w))

    ranked = find_optimal_portfolios(
        strategy_pnl    = strategy_pnl,
        corr_matrix     = corr_df,
        n_accounts      = n_accounts,
        max_strategies  = max_strats,
        max_days        = max_days,
        k_thresholds    = k_thresholds,
        top_n           = top_n,
    )

    # ── 6. Results ────────────────────────────────────────────────────────────
    print_portfolio_results(ranked, k_thresholds, cycle_days=max_days, width=w)


if __name__ == "__main__":
    main()
