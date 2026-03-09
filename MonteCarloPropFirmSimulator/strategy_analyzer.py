"""
strategy_analyzer.py
====================
Strategy Intelligence Layer for PassPlan — Prop Firm Probability Engine.

Extracts statistical features from an uploaded CSV trading strategy and
returns them as a plain JSON-serialisable dict.

Public API
----------
  analyze_strategy(csv_path: str) -> dict

    Read *csv_path*, compute all metrics, and return a feature dict.
    Returns a dict of safe default values (NaN → None) on any error so the
    caller can always store the result without crashing.

Required CSV format
-------------------
  Must contain at minimum a column named ``pnl`` (case-insensitive).
  All other columns are ignored.

  trade_id,pnl
  1,50
  2,-25
  3,40

Returned keys
-------------
  num_trades, win_rate, avg_win, avg_loss, rr_ratio, expectancy,
  profit_factor, std_dev, variance, skew, kurtosis,
  max_drawdown, max_win_streak, max_loss_streak
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Safe numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: Any) -> Optional[float]:
    """Return *value* as a Python float, or None if it is NaN / Inf / None."""
    try:
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Return *value* as a Python int, or None on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Streak helper
# ─────────────────────────────────────────────────────────────────────────────

def _streaks(flags: list[bool]) -> tuple[int, int]:
    """
    Given a list of booleans (True = win, False = loss), return
    (max_win_streak, max_loss_streak).
    """
    max_win = max_loss = cur_win = cur_loss = 0
    for f in flags:
        if f:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        if cur_win  > max_win:  max_win  = cur_win
        if cur_loss > max_loss: max_loss = cur_loss
    return max_win, max_loss


# ─────────────────────────────────────────────────────────────────────────────
# Max drawdown
# ─────────────────────────────────────────────────────────────────────────────

def _max_drawdown(pnl_series: list[float]) -> float:
    """
    Build a cumulative equity curve and compute the maximum peak-to-trough
    drawdown (returned as a **negative** float, e.g. -1500.0).

    Returns 0.0 for an empty series.
    """
    if not pnl_series:
        return 0.0

    equity = 0.0
    peak   = 0.0
    max_dd = 0.0

    for pnl in pnl_series:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = equity - peak          # always ≤ 0
        if dd < max_dd:
            max_dd = dd

    return max_dd


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FEATURES: Dict[str, Any] = {
    "num_trades":     0,
    "win_rate":       None,
    "avg_win":        None,
    "avg_loss":       None,
    "rr_ratio":       None,
    "expectancy":     None,
    "profit_factor":  None,
    "std_dev":        None,
    "variance":       None,
    "skew":           None,
    "kurtosis":       None,
    "max_drawdown":   None,
    "max_win_streak": 0,
    "max_loss_streak": 0,
}


def analyze_strategy(csv_path: str) -> Dict[str, Any]:
    """
    Parse *csv_path*, compute statistical features, and return a dict.

    All values are JSON-serialisable (float or int, never NaN/Inf).
    On any error a safe default dict is returned so the caller never crashes.
    """
    # ── import pandas lazily so the module is importable even without it ──
    try:
        import pandas as pd
        from scipy import stats as sp_stats
    except ImportError as exc:
        log.error("strategy_analyzer: missing dependency — %s", exc)
        return dict(_DEFAULT_FEATURES)

    path = Path(csv_path)
    if not path.exists():
        log.warning("strategy_analyzer: file not found — %s", csv_path)
        return dict(_DEFAULT_FEATURES)

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log.warning("strategy_analyzer: could not parse CSV %s — %s", csv_path, exc)
        return dict(_DEFAULT_FEATURES)

    # Normalise column names; require 'pnl'
    df.columns = [c.strip().lower() for c in df.columns]
    if "pnl" not in df.columns:
        log.warning("strategy_analyzer: 'pnl' column not found in %s", csv_path)
        return dict(_DEFAULT_FEATURES)

    pnl_raw = pd.to_numeric(df["pnl"], errors="coerce").dropna()
    if pnl_raw.empty:
        log.warning("strategy_analyzer: no numeric pnl values in %s", csv_path)
        return dict(_DEFAULT_FEATURES)

    pnl: list[float] = pnl_raw.tolist()
    n = len(pnl)

    # ── Trade statistics ──────────────────────────────────────────────────────
    wins   = [v for v in pnl if v > 0]
    losses = [v for v in pnl if v < 0]

    win_rate  = len(wins) / n if n else None
    avg_win   = sum(wins)   / len(wins)   if wins   else None
    avg_loss  = sum(losses) / len(losses) if losses else None   # negative

    # Risk/Reward
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        rr_ratio = avg_win / abs(avg_loss)
    else:
        rr_ratio = None

    # Expectancy
    if avg_win is not None and avg_loss is not None and win_rate is not None:
        expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    else:
        expectancy = None

    # Profit factor
    total_wins   = sum(wins)
    total_losses = abs(sum(losses))
    profit_factor = (total_wins / total_losses) if total_losses else None

    # ── Distribution statistics ───────────────────────────────────────────────
    if n >= 2:
        series = pd.Series(pnl)
        std_dev    = _safe(series.std(ddof=1))
        variance   = _safe(series.var(ddof=1))
        skew_val   = _safe(sp_stats.skew(pnl, bias=False))
        kurt_val   = _safe(sp_stats.kurtosis(pnl, fisher=True, bias=False))  # excess k
    else:
        std_dev = variance = skew_val = kurt_val = None

    # ── Drawdown ──────────────────────────────────────────────────────────────
    max_dd = _safe(_max_drawdown(pnl))

    # ── Streaks ───────────────────────────────────────────────────────────────
    flags = [v > 0 for v in pnl]
    max_win_streak, max_loss_streak = _streaks(flags)

    return {
        "num_trades":     n,
        "win_rate":       _safe(win_rate),
        "avg_win":        _safe(avg_win),
        "avg_loss":       _safe(avg_loss),
        "rr_ratio":       _safe(rr_ratio),
        "expectancy":     _safe(expectancy),
        "profit_factor":  _safe(profit_factor),
        "std_dev":        _safe(std_dev),
        "variance":       _safe(variance),
        "skew":           _safe(skew_val),
        "kurtosis":       _safe(kurt_val),
        "max_drawdown":   max_dd,
        "max_win_streak":  _safe_int(max_win_streak) or 0,
        "max_loss_streak": _safe_int(max_loss_streak) or 0,
    }
