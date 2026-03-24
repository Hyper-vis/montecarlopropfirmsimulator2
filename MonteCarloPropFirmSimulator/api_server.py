"""
api_server.py
=============
FastAPI REST layer over engine_interface.py.

All simulation logic lives in the engine; this file only defines request
schemas, calls the interface functions, and serialises results.

Endpoints
---------
  POST /analyze/until_payout    — single strategy, run-until-payout mode
  POST /analyze/full_period     — single strategy, fixed-window (21-day) mode
  POST /analyze/batch           — multiple CSVs, both modes, ranked by EV
  POST /portfolio/correlation   — pairwise correlation + simultaneous drawdown
  POST /portfolio/multi_account — N concurrent Apex accounts on one strategy

Start the server
----------------
  python api_server.py

Interactive docs
----------------
  http://127.0.0.1:8000/docs     (Swagger UI)
  http://127.0.0.1:8000/redoc    (ReDoc)

Version
-------
  1.0  — Phase 3 initial release
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import numpy as np
log = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import strategy_db
import strategy_analyzer
from engine_interface import (
    analyze_until_payout,
    analyze_full_period,
    run_batch,
    analyze_correlation,
    run_multi_account,
    analyze_rescue,
)
from portfolio_optimizer import (
    load_strategy_pnl,
    compute_correlation_matrix,
    find_optimal_portfolios,
)
from services.monte_carlo_service import run_trade_simulation_profile
from services.monte_carlo_service import run_daily_simulation_profile
from apex_engine_v3_1 import load_daily_pnl as load_mt5_daily_pnl
from routers.upload_mt5 import router as upload_mt5_router
from routers.upload_ninjatrader import router as upload_ninjatrader_router

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Monte Carlo Prop Firm Simulator API",
    description=(
        "REST interface for the Apex 50k Monte Carlo engine. "
        "All endpoints accept JSON bodies and return JSON-serialisable dicts."
    ),
    version="1.0.0",
)

# Configure CORS origins via env for production deployment.
# Example:
#   ALLOWED_ORIGINS=https://creditus.ca,https://www.creditus.ca,https://sim.creditus.ca
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(upload_mt5_router)
app.include_router(upload_ninjatrader_router)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy storage — SQLite-backed; CSV files live in strategies/
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES_DIR = Path("strategies")
STRATEGIES_DIR.mkdir(exist_ok=True)

# Initialise DB and run one-shot JSON → SQLite migration on startup
strategy_db.initialize_db()
strategy_db.migrate_from_json(STRATEGIES_DIR / "registry.json")


def _resolve_csv(csv_path: Optional[str], strategy_id: Optional[str]) -> str:
    """Return a file path from either a raw csv_path or a registry strategy_id."""
    if strategy_id:
        entry = strategy_db.get_strategy(strategy_id)
        if entry is None:
            raise ValueError(f"strategy_id not found: {strategy_id!r}")
        return entry["path"]
    if csv_path:
        return csv_path
    raise ValueError("Provide either csv_path or strategy_id.")


def _resolve_entry(strategy_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not strategy_id:
        return None
    return strategy_db.get_strategy(strategy_id)


def _is_mt5_strategy(strategy_id: Optional[str]) -> bool:
    entry = _resolve_entry(strategy_id)
    return bool(entry and entry.get("source") == "mt5")


def _strategy_source(strategy_id: str) -> str:
    entry = strategy_db.get_strategy(strategy_id)
    if entry is None:
        raise ValueError(f"strategy_id not found: {strategy_id!r}")
    return str(entry.get("source") or "tradingview")


def _validate_homogeneous_sources(strategy_ids: List[str]) -> None:
    if not strategy_ids:
        return
    sources = {_strategy_source(sid) for sid in strategy_ids}
    if len(sources) > 1:
        ordered = ", ".join(sorted(sources))
        raise ValueError(
            f"Mixed strategy sources are not allowed in one run. Found: {ordered}."
        )


def _sources_for_strategy_ids(strategy_ids: List[str]) -> set[str]:
    if not strategy_ids:
        return set()
    return {_strategy_source(sid) for sid in strategy_ids}


def _load_trade_results_from_strategy_csv(csv_path: str) -> List[float]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "Net P&L USD" not in df.columns:
        raise ValueError("Stored strategy CSV is missing 'Net P&L USD' column.")
    return [float(v) for v in df["Net P&L USD"].tolist() if pd.notna(v)]


def _calendar_to_trading_days(days: int, weekends_tradable: bool) -> int:
    if weekends_tradable:
        return max(1, int(days))
    return max(1, int(math.ceil(float(days) * 5.0 / 7.0)))


def _mt5_overrides_from_request(req: Any) -> dict[str, Any]:
    start = float(req.starting_balance)
    target = float(req.target_balance)
    overall = float(req.overall_max_loss)
    daily = float(req.daily_max_loss)
    days = int(req.time_limit_days)
    weekends = bool(req.weekends_tradable)

    if start <= 0:
        raise ValueError("Starting balance must be greater than 0.")
    if target <= start:
        raise ValueError("Target balance must be greater than starting balance.")
    if overall <= 0:
        raise ValueError("Overall max loss must be greater than 0.")
    if daily <= 0:
        raise ValueError("Daily max loss must be greater than 0.")
    if days < 1:
        raise ValueError("Time limit must be at least 1 calendar day.")

    risk = float(getattr(req, "risk_multiplier", 1.0))
    if risk <= 0:
        raise ValueError("Risk multiplier must be greater than 0.")

    return {
        "account_size": start,
        "payout_threshold": target,
        "overall_max_loss": overall,
        "daily_loss_limit": daily,
        "max_days": _calendar_to_trading_days(days, weekends),
        "risk_multiplier": risk,
    }


def _run_mt5_personal_simulation(
    csv_path: str,
    req: Any,
    *,
    stop_at_payout: bool,
    progress_cb: Any = None,
) -> tuple[dict, dict[str, Any]]:
    overrides = _mt5_overrides_from_request(req)
    daily_pnl = load_mt5_daily_pnl(csv_path)
    personal = run_daily_simulation_profile(
        daily_pnl.tolist() if hasattr(daily_pnl, "tolist") else list(daily_pnl),
        n_sims=req.n_sims,
        stop_at_payout=stop_at_payout,
        profile="personal",
        config_overrides=overrides,
        progress_cb=progress_cb,
    )
    return personal, overrides


def _map_personal_until_payout(mc: dict, *, csv_path: str, n_sims: int, risk_multiplier: float, max_days: int) -> dict:
    payouts = [float(v) for v in mc.get("payouts", [])]
    days = [int(v) for v in mc.get("days", [])]
    target_days = [int(v) for v in mc.get("target_achieved_days", [])]
    target_prob = float(mc.get("target_achieved_probability", mc.get("pass_probability", 0.0)))
    mean_target_gain = float(mc.get("expected_payout", 0.0))

    return {
        "metadata": {
            "csv_path": csv_path,
            "n_sims": n_sims,
            "risk_multiplier": risk_multiplier,
            "max_days": max_days,
            "sampling_mode": "uniform",
            "weight_strength": 0.0,
            "recent_window": 0,
            "profile": "mt5_personal",
            "primary_success_label": "target_achieved",
        },
        "probabilities": {
            "payout_prob": target_prob,
            "target_achieved_prob": target_prob,
            "blow_prob": mc.get("blow_probability", 0.0),
            "timeout_prob": mc.get("timeout_probability", 0.0),
        },
        "metrics": {
            "mean_days_all": float(np.mean(days)) if days else 0.0,
            "mean_days_to_payout": float(np.mean(target_days)) if target_days else 0.0,
            "median_days_to_payout": int(np.median(target_days)) if target_days else 0,
            "mean_payout": mean_target_gain,
            "tier_label": "PERSONAL",
            "tier_description": "Personal account mode with user-defined balance, loss, target and time rules.",
        },
        "distributions": {
            "equity_paths": mc.get("paths", []),
        },
        "recency": {
            "overall_pass_probability": target_prob,
            "recent_pass_probability": None,
            "probability_delta": None,
            "recency_status": "n/a",
            "recency_comment": "Recency analysis is not applied in MT5 personal mode.",
        },
    }


def _map_personal_full_period(
    mc: dict,
    *,
    csv_path: str,
    n_sims: int,
    risk_multiplier: float,
    max_days: int,
    reset_cost: float,
) -> dict:
    outcomes = list(mc.get("outcomes", []))
    payouts = [float(v) for v in mc.get("payouts", [])]
    balances = [float(v) for v in mc.get("balances", [])]
    target_prob = float(mc.get("target_achieved_probability", mc.get("pass_probability", 0.0)))
    target_days = [int(v) for v in mc.get("target_achieved_days", [])]
    n = max(1, len(outcomes))

    payout_then_blew = sum(1 for o, p in zip(outcomes, payouts) if o == "blow" and p > 0)
    payout_survived = sum(1 for o, p in zip(outcomes, payouts) if o != "blow" and p > 0)
    blow_no_payout = outcomes.count("blow") - payout_then_blew
    timeout_no_pay = max(0.0, 1.0 - target_prob - float(mc.get("blow_probability", 0.0)))
    winning_payouts = [p for p in payouts if p > 0]
    pass_rate = target_prob
    fail_rate = float(mc.get("blow_probability", 0.0))
    avg_blow_loss = 0.0
    account_size = float(mc.get("config", {}).get("account_size", 0.0))
    blow_balances = [b for o, b in zip(outcomes, balances) if o == "blow"]
    if blow_balances:
        avg_blow_loss = float(np.mean([account_size - b for b in blow_balances]))
    mean_payout = float(np.mean(winning_payouts)) if winning_payouts else 0.0
    median_payout = float(np.median(winning_payouts)) if winning_payouts else 0.0
    e_monthly = pass_rate * mean_payout - fail_rate * avg_blow_loss - reset_cost

    return {
        "metadata": {
            "csv_path": csv_path,
            "n_sims": n_sims,
            "risk_multiplier": risk_multiplier,
            "max_days": max_days,
            "reset_cost": reset_cost,
            "sampling_mode": "uniform",
            "weight_strength": 0.0,
            "recent_window": 0,
            "profile": "mt5_personal",
            "primary_success_label": "target_achieved",
        },
        "probabilities": {
            "payout_prob": pass_rate,
            "target_achieved_prob": pass_rate,
            "payout_survived_prob": payout_survived / n,
            "payout_then_blew_prob": payout_then_blew / n,
            "blow_no_payout_prob": blow_no_payout / n,
            "timeout_no_pay_prob": timeout_no_pay,
        },
        "metrics": {
            "mean_ending_balance": float(np.mean(balances)) if balances else 0.0,
            "mean_blow_loss": avg_blow_loss,
            "mean_payout": mean_payout,
            "median_payout": median_payout,
            "mean_days_to_payout": float(np.mean(target_days)) if target_days else 0.0,
            "e_monthly": e_monthly,
            "tier_label": "PERSONAL",
            "tier_description": "Personal account mode with user-defined balance, loss, target and time rules.",
        },
        "distributions": {
            "equity_paths": mc.get("paths", []),
        },
        "recency": {
            "overall_pass_probability": pass_rate,
            "recent_pass_probability": None,
            "probability_delta": None,
            "recency_status": "n/a",
            "recency_comment": "Recency analysis is not applied in MT5 personal mode.",
        },
    }


def _run_personal_batch(
    strategy_ids: List[str],
    req: Any,
    progress_cb: Any = None,
) -> List[Dict[str, Any]]:
    overrides = _mt5_overrides_from_request(req)
    total = len(strategy_ids)
    rows: List[Dict[str, Any]] = []

    for i, sid in enumerate(strategy_ids, start=1):
        entry = strategy_db.get_strategy(sid)
        if entry is None:
            rows.append({"csv": sid, "error": f"strategy_id not found: {sid!r}"})
            if progress_cb:
                progress_cb(i, total, sid)
            continue

        csv_path = entry["path"]
        csv_name = Path(csv_path).name

        try:
            trades = _load_trade_results_from_strategy_csv(csv_path)
            utp = run_trade_simulation_profile(
                trades,
                n_sims=req.n_sims,
                stop_at_payout=True,
                profile="personal",
                config_overrides=overrides,
            )
            fp = run_trade_simulation_profile(
                trades,
                n_sims=req.n_sims,
                stop_at_payout=False,
                profile="personal",
                config_overrides=overrides,
            )

            utp_days = [
                int(d)
                for o, d in zip(utp.get("outcomes", []), utp.get("days", []))
                if o == "payout"
            ]
            utp_mean_days = float(np.mean(utp_days)) if utp_days else 0.0
            utp_ev_monthly = (
                float(utp.get("pass_probability", 0.0))
                * float(utp.get("expected_payout", 0.0))
                * (21.0 / utp_mean_days)
                if utp_mean_days > 0
                else 0.0
            )

            fp_ev_net = (
                float(fp.get("pass_probability", 0.0))
                * float(fp.get("expected_payout", 0.0))
                - float(req.reset_cost)
            )

            rows.append(
                {
                    "csv": csv_name,
                    "utp_payout_p": float(utp.get("pass_probability", 0.0)),
                    "utp_blow_p": float(utp.get("blow_probability", 0.0)),
                    "utp_mean_days": utp_mean_days,
                    "utp_ev_monthly": utp_ev_monthly,
                    "utp_rating": "PERSONAL",
                    "fp_payout_p": float(fp.get("pass_probability", 0.0)),
                    "fp_blow_no_pay_p": float(fp.get("blow_probability", 0.0)),
                    "fp_ev_net": fp_ev_net,
                    "recent_pass_probability": None,
                    "probability_delta": None,
                    "recency_status": None,
                }
            )
        except Exception as exc:
            rows.append({"csv": csv_name, "error": str(exc)})

        if progress_cb:
            progress_cb(i, total, csv_path)

    ok_rows = [r for r in rows if isinstance(r, dict) and "error" not in r]
    err_rows = [r for r in rows if isinstance(r, dict) and "error" in r]
    ok_rows.sort(key=lambda r: float(r.get("fp_ev_net") or 0.0), reverse=True)
    return ok_rows + err_rows


def _run_personal_multi_account(
    *,
    trades: List[float],
    n_accounts: int,
    n_sims: int,
    stop_at_payout: bool,
    n_path_sims: int,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    if n_accounts < 1:
        raise ValueError("n_accounts must be >= 1")

    account_runs: List[dict] = []
    for _ in range(n_accounts):
        account_runs.append(
            run_trade_simulation_profile(
                trades,
                n_sims=n_sims,
                stop_at_payout=stop_at_payout,
                profile="personal",
                config_overrides=overrides,
            )
        )

    outcomes_by_account = [r.get("outcomes", []) for r in account_runs]
    payouts_by_account = [r.get("payouts", []) for r in account_runs]
    balances_by_account = [r.get("balances", []) for r in account_runs]
    days_by_account = [r.get("days", []) for r in account_runs]
    paths_by_account = [r.get("paths", []) for r in account_runs]

    sims = min(len(v) for v in outcomes_by_account) if outcomes_by_account else 0
    if sims <= 0:
        raise ValueError("Simulation returned no outcomes for multi-account mode.")

    portfolio_payouts: List[float] = []
    blown_counts: List[int] = []
    end_balance_portfolio: List[float] = []
    any_payout_flags: List[bool] = []
    days_to_any_payout: List[int] = []

    for i in range(sims):
        payouts_i = [float(payouts_by_account[a][i]) for a in range(n_accounts)]
        outcomes_i = [str(outcomes_by_account[a][i]) for a in range(n_accounts)]
        balances_i = [float(balances_by_account[a][i]) for a in range(n_accounts)]
        days_i = [int(days_by_account[a][i]) for a in range(n_accounts)]

        total_payout = float(sum(payouts_i))
        portfolio_payouts.append(total_payout)
        blown_counts.append(sum(1 for o in outcomes_i if o == "blow"))
        end_balance_portfolio.append(float(sum(balances_i)))

        has_payout = any(p > 0 for p in payouts_i)
        any_payout_flags.append(has_payout)
        if has_payout:
            payout_days_i = [d for d, p in zip(days_i, payouts_i) if p > 0]
            days_to_any_payout.append(min(payout_days_i))

    pass_prob = float(np.mean(any_payout_flags)) if any_payout_flags else 0.0
    mean_total_payout = float(np.mean(portfolio_payouts)) if portfolio_payouts else 0.0
    mean_days_to_payout = float(np.mean(days_to_any_payout)) if days_to_any_payout else 0.0
    est_cycles_per_month = (21.0 / mean_days_to_payout) if mean_days_to_payout > 0 else 0.0
    est_monthly_gross = est_cycles_per_month * mean_total_payout

    mean_end_balance_per_acct = [
        float(np.mean([float(v) for v in balances])) if balances else 0.0
        for balances in balances_by_account
    ]

    chart_account_paths: List[List[Dict[str, Any]]] = []
    max_chart_sims = max(0, int(n_path_sims))
    for a in range(n_accounts):
        acct = []
        max_k = min(max_chart_sims, len(paths_by_account[a]), len(outcomes_by_account[a]))
        for k in range(max_k):
            acct.append(
                {
                    "path": [float(x) for x in paths_by_account[a][k]],
                    "outcome": str(outcomes_by_account[a][k]),
                }
            )
        chart_account_paths.append(acct)

    account_size = float(overrides.get("account_size", 50_000.0))
    payout_threshold = float(overrides.get("payout_threshold", 52_600.0))
    blow_floor = float(account_size - float(overrides.get("overall_max_loss", 2_500.0)))

    return {
        "metadata": {
            "n_accounts": int(n_accounts),
            "n_sims": int(sims),
            "max_days": int(overrides.get("max_days", 30)),
            "stop_at_payout": bool(stop_at_payout),
            "profile": "mt5_personal",
        },
        "portfolio": {
            "mean_total_payout": mean_total_payout,
            "median_total_payout": float(np.median(portfolio_payouts)) if portfolio_payouts else 0.0,
            "p5_payout": float(np.percentile(portfolio_payouts, 5)) if portfolio_payouts else 0.0,
            "p95_payout": float(np.percentile(portfolio_payouts, 95)) if portfolio_payouts else 0.0,
            "prob_any_payout": pass_prob,
            "mean_blown_accounts": float(np.mean(blown_counts)) if blown_counts else 0.0,
            "mean_end_balance_single": float(np.mean(mean_end_balance_per_acct)) if mean_end_balance_per_acct else 0.0,
            "mean_end_balance_portfolio": float(np.mean(end_balance_portfolio)) if end_balance_portfolio else 0.0,
            "mean_end_balance_per_acct": mean_end_balance_per_acct,
        },
        "cycle_efficiency": {
            "payout_prob_per_cycle": pass_prob,
            "mean_days_to_payout": mean_days_to_payout,
            "median_days_to_payout": float(np.median(days_to_any_payout)) if days_to_any_payout else 0.0,
            "p5_days": float(np.percentile(days_to_any_payout, 5)) if days_to_any_payout else 0.0,
            "p95_days": float(np.percentile(days_to_any_payout, 95)) if days_to_any_payout else 0.0,
            "pct_within_gate": float(np.mean([d <= 21 for d in days_to_any_payout])) if days_to_any_payout else 0.0,
            "pct_within_10": float(np.mean([d <= 10 for d in days_to_any_payout])) if days_to_any_payout else 0.0,
            "pct_within_15": float(np.mean([d <= 15 for d in days_to_any_payout])) if days_to_any_payout else 0.0,
            "est_cycles_per_month": est_cycles_per_month,
            "est_monthly_gross": est_monthly_gross,
            "est_monthly_net": est_monthly_gross,
        },
        "chart_data": {
            "portfolio_payouts": portfolio_payouts,
            "account_paths": chart_account_paths,
            "thresholds": {
                "start": account_size,
                "payout": payout_threshold,
                "blow_floor": blow_floor,
            },
        },
    }


def _resolve_csv_list(
    csv_paths: Optional[List[str]],
    strategy_ids: Optional[List[str]],
) -> List[str]:
    """Resolve a list of file paths from csv_paths or strategy_ids."""
    if strategy_ids:
        _validate_homogeneous_sources(strategy_ids)
        out: List[str] = []
        for sid in strategy_ids:
            entry = strategy_db.get_strategy(sid)
            if entry is None:
                raise ValueError(f"strategy_id not found: {sid!r}")
            out.append(entry["path"])
        return out
    if csv_paths:
        return csv_paths
    raise ValueError("Provide either csv_paths or strategy_ids.")


def _resolve_named_csvs(
    strategies: Optional[Dict[str, str]],
    strategy_ids: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Resolve {name: path} from strategies dict or strategy_ids dict."""
    if strategy_ids:
        _validate_homogeneous_sources(list(strategy_ids.values()))
        out: Dict[str, str] = {}
        for name, sid in strategy_ids.items():
            entry = strategy_db.get_strategy(sid)
            if entry is None:
                raise ValueError(f"strategy_id not found: {sid!r}")
            out[name] = entry["path"]
        return out
    if strategies:
        return strategies
    raise ValueError("Provide either strategies or strategy_ids.")


def _safe(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely drill into nested dicts without raising KeyError/TypeError."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _store_until_payout(strategy_id: str, result: Any, n_sims: int) -> None:
    """Persist an until_payout simulation result to the leaderboard table."""
    try:
        payout_prob  = _safe(result, "probabilities", "payout_prob") or 0.0
        mean_payout  = _safe(result, "metrics", "mean_payout") or 0.0
        mean_days    = _safe(result, "metrics", "mean_days_to_payout")
        # Approximate monthly payout: prob × avg_payout × (21 trading days / avg_days_per_payout)
        e_monthly = payout_prob * mean_payout * (21.0 / mean_days) if mean_days else None
        strategy_db.insert_simulation_result(
            strategy_id             = strategy_id,
            simulation_type         = "until_payout",
            pass_probability        = _safe(result, "probabilities", "payout_prob"),
            fail_probability        = _safe(result, "probabilities", "blow_prob"),
            expected_monthly_payout = e_monthly,
            num_simulations         = n_sims,
        )
    except Exception as exc:
        log.warning("leaderboard: could not store until_payout result: %s", exc)


def _store_full_period(strategy_id: str, result: Any, n_sims: int) -> None:
    """Persist a full_period simulation result to the leaderboard table."""
    try:
        strategy_db.insert_simulation_result(
            strategy_id             = strategy_id,
            simulation_type         = "full_period",
            pass_probability        = _safe(result, "probabilities", "payout_prob"),
            fail_probability        = _safe(result, "probabilities", "blow_no_payout_prob"),
            expected_monthly_payout = _safe(result, "metrics", "e_monthly"),
            num_simulations         = n_sims,
        )
    except Exception as exc:
        log.warning("leaderboard: could not store full_period result: %s", exc)


def _store_multi_account(strategy_id: str, result: Any, n_sims: int) -> None:
    """Persist a multi_account simulation result to the leaderboard table."""
    try:
        strategy_db.insert_simulation_result(
            strategy_id             = strategy_id,
            simulation_type         = "multi_account",
            pass_probability        = _safe(result, "portfolio", "prob_any_payout"),
            expected_monthly_payout = _safe(result, "cycle_efficiency", "est_monthly_net"),
            num_simulations         = n_sims,
        )
    except Exception as exc:
        log.warning("leaderboard: could not store multi_account result: %s", exc)


def _build_sid_map(strategy_ids: List[str]) -> dict[str, str]:
    """Return {stored_uuid_filename: strategy_id} for a list of strategy IDs.

    Results from run_batch carry ``row["csv"] == os.path.basename(stored_path)``
    (the UUID-named file).  This map lets us match each sorted result back to
    the correct strategy_id regardless of the sort order applied to results.
    """
    m: dict[str, str] = {}
    for sid in strategy_ids:
        try:
            entry = strategy_db.get_strategy(sid)
            if entry:
                m[Path(entry["path"]).name] = sid
        except Exception:
            pass
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────────────────────────────────────

class UntilPayoutRequest(BaseModel):
    csv_path: Optional[str]  = Field(None, description="Path to the daily PnL CSV file (mutually exclusive with strategy_id).")
    strategy_id: Optional[str] = Field(None, description="Registered strategy ID from POST /strategies/upload.")
    n_sims: int = Field(10_000, ge=1, description="Number of Monte Carlo simulations.")
    risk_multiplier: float = Field(1.0, gt=0, description="Scale factor applied to each day's PnL.")
    max_days: int = Field(90, ge=1, description="Hard cap on trading days per simulation path.")
    n_paths: int = Field(50, ge=0, description="Number of equity paths to capture (for charts).")
    sampling_mode: str = Field("uniform", description='"uniform" | "recency_weighted" | "recent_only"')
    weight_strength: float = Field(3.0, description="Exponential steepness (recency_weighted mode only).")
    recent_window: int = Field(50, ge=1, description="Look-back in trading days (recent_only mode only).")
    seed: Optional[int] = Field(None, description="RNG seed for reproducible results.")
    starting_balance: float = Field(50_000.0, gt=0, description="MT5 personal mode: starting balance.")
    overall_max_loss: float = Field(2_500.0, gt=0, description="MT5 personal mode: max absolute loss from start.")
    daily_max_loss: float = Field(700.0, gt=0, description="MT5 personal mode: max daily loss.")
    target_balance: float = Field(52_600.0, gt=0, description="MT5 personal mode: target account balance.")
    time_limit_days: int = Field(90, ge=1, description="MT5 personal mode: calendar-day time constraint.")
    weekends_tradable: bool = Field(False, description="MT5 personal mode: include weekends as tradable days.")


class FullPeriodRequest(BaseModel):
    csv_path: Optional[str]  = Field(None, description="Path to the daily PnL CSV file (mutually exclusive with strategy_id).")
    strategy_id: Optional[str] = Field(None, description="Registered strategy ID from POST /strategies/upload.")
    n_sims: int = Field(10_000, ge=1, description="Number of Monte Carlo simulations.")
    risk_multiplier: float = Field(1.0, gt=0, description="Scale factor applied to each day's PnL.")
    max_days: int = Field(21, ge=1, description="Fixed window length in trading days (default ≈ 1 month).")
    n_paths: int = Field(50, ge=0, description="Number of equity paths to capture (for charts).")
    reset_cost: float = Field(137.0, ge=0, description="Account reset fee ($) used in EV calculation.")
    sampling_mode: str = Field("uniform", description='"uniform" | "recency_weighted" | "recent_only"')
    weight_strength: float = Field(3.0, description="Exponential steepness (recency_weighted mode only).")
    recent_window: int = Field(50, ge=1, description="Look-back in trading days (recent_only mode only).")
    seed: Optional[int] = Field(None, description="RNG seed for reproducible results.")
    starting_balance: float = Field(50_000.0, gt=0, description="MT5 personal mode: starting balance.")
    overall_max_loss: float = Field(2_500.0, gt=0, description="MT5 personal mode: max absolute loss from start.")
    daily_max_loss: float = Field(700.0, gt=0, description="MT5 personal mode: max daily loss.")
    target_balance: float = Field(52_600.0, gt=0, description="MT5 personal mode: target account balance.")
    time_limit_days: int = Field(21, ge=1, description="MT5 personal mode: calendar-day time constraint.")
    weekends_tradable: bool = Field(False, description="MT5 personal mode: include weekends as tradable days.")


class BatchRequest(BaseModel):
    csv_paths: Optional[List[str]]  = Field(None, description="List of CSV file paths (mutually exclusive with strategy_ids).")
    strategy_ids: Optional[List[str]] = Field(None, description="List of registered strategy IDs (mutually exclusive with csv_paths).")
    n_sims: int = Field(10_000, ge=1, description="Simulations per CSV per mode.")
    sampling_mode: str = Field("uniform", description='"uniform" | "recency_weighted" | "recent_only"')
    weight_strength: float = Field(3.0, description="Exponential steepness (recency_weighted mode only).")
    recent_window: int = Field(50, ge=1, description="Look-back in trading days (recent_only mode only).")
    risk_multiplier: float = Field(1.0, gt=0, description="Scale factor for MT5 personal mode.")
    reset_cost: float = Field(137.0, ge=0, description="Account reset fee ($) used in EV calculation.")
    seed: Optional[int] = Field(None, description="RNG seed for reproducible results.")
    starting_balance: float = Field(50_000.0, gt=0, description="MT5 personal mode: starting balance.")
    overall_max_loss: float = Field(2_500.0, gt=0, description="MT5 personal mode: max absolute loss from start.")
    daily_max_loss: float = Field(700.0, gt=0, description="MT5 personal mode: max daily loss.")
    target_balance: float = Field(52_600.0, gt=0, description="MT5 personal mode: target account balance.")
    time_limit_days: int = Field(90, ge=1, description="MT5 personal mode: calendar-day time constraint.")
    weekends_tradable: bool = Field(False, description="MT5 personal mode: include weekends as tradable days.")


class CorrelationRequest(BaseModel):
    strategies: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of {strategy_name: csv_path}. Mutually exclusive with strategy_ids.",
        examples=[{"ES": "MultiBot_ES.csv", "YM": "MultiBot_YM.csv"}],
    )
    strategy_ids: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of {strategy_name: strategy_id}. Mutually exclusive with strategies.",
    )
    rolling_window: int = Field(30, ge=2, description="Rolling correlation window in trading days.")
    dd_threshold: float = Field(-500.0, description="Drawdown level ($) considered a danger event.")
    mc_n_sims: int = Field(5_000, ge=1, description="Monte Carlo sims for the blow-up correlation matrix.")
    mc_max_days: int = Field(30, ge=1, description="Days per Monte Carlo path for blow-up correlation.")
    sampling_mode: str = Field("uniform", description='"uniform" | "recency_weighted" | "recent_only"')
    weight_strength: float = Field(3.0, description="Exponential steepness (recency_weighted mode only).")
    recent_window: int = Field(50, ge=1, description="Look-back in trading days (recent_only mode only).")
    seed: Optional[int] = Field(None, description="RNG seed for reproducible results.")


class MultiAccountRequest(BaseModel):
    csv_path: Optional[str]  = Field(None, description="Path to the daily PnL CSV file (mutually exclusive with strategy_id).")
    strategy_id: Optional[str] = Field(None, description="Registered strategy ID from POST /strategies/upload.")
    n_accounts: int = Field(..., ge=1, description="Number of parallel Apex accounts in the portfolio.")
    n_sims: int = Field(10_000, ge=1, description="Number of Monte Carlo simulations.")
    max_days: int = Field(30, ge=1, description="Maximum trading days per account cycle.")
    stop_at_payout: bool = Field(True, description="Stop each account at its first qualifying payout.")
    risk_multiplier: float = Field(1.0, gt=0, description="Scale factor applied to each day's PnL.")
    sampling_mode: str = Field("uniform", description='"uniform" | "recency_weighted" | "recent_only"')
    weight_strength: float = Field(3.0, description="Exponential steepness (recency_weighted mode only).")
    recent_window: int = Field(50, ge=1, description="Look-back in trading days (recent_only mode only).")
    n_path_sims: int = Field(150, ge=0, description="Equity path samples to capture (for chart generation).")
    seed: Optional[int] = Field(None, description="RNG seed for reproducible results.")
    starting_balance: float = Field(50_000.0, gt=0, description="MT5 personal mode: starting balance.")
    overall_max_loss: float = Field(2_500.0, gt=0, description="MT5 personal mode: max absolute loss from start.")
    daily_max_loss: float = Field(700.0, gt=0, description="MT5 personal mode: max daily loss.")
    target_balance: float = Field(52_600.0, gt=0, description="MT5 personal mode: target account balance.")
    time_limit_days: int = Field(30, ge=1, description="MT5 personal mode: calendar-day time constraint.")
    weekends_tradable: bool = Field(False, description="MT5 personal mode: include weekends as tradable days.")


class OptimizeRequest(BaseModel):
    strategy_ids: List[str] = Field(
        ...,
        description="List of registered strategy IDs to include in the tournament search.",
    )
    n_accounts: int = Field(10, ge=1, description="Total Apex accounts to allocate across strategies.")
    max_strategies: int = Field(3, ge=1, description="Max strategies allowed in a single portfolio composition.")
    max_days: int = Field(21, ge=1, description="Trading-day cycle length per account (default 21 ≈ 1 month).")
    top_n: int = Field(5, ge=1, description="Number of top-ranked portfolios to return.")
    seed: Optional[int] = Field(42, description="RNG seed for the tournament search.")
    k_thresholds: Optional[List[int]] = Field(
        None, description="P(≥k payouts) thresholds. Defaults to [1, 2, 3, 5, 10]."
    )


class RescueRequest(BaseModel):
    strategy_ids: List[str] = Field(
        ...,
        description="List of registered strategy IDs to score from the current mid-account state.",
    )
    current_balance: float = Field(
        ..., description="Current account equity (e.g. 49800)."
    )
    trailing_stop_floor: float = Field(
        ..., description="Current trailing-stop floor (e.g. 47800)."
    )
    n_sims: int = Field(10_000, ge=100, description="Monte-Carlo simulations per strategy.")
    max_days: int = Field(90, ge=1, description="Day budget per simulation.")
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility.")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _error(message: str, status: int = 400) -> JSONResponse:
    """Return a JSON error response."""
    return JSONResponse(status_code=status, content={"error": message})


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["UI"])
def root():
    """Serve the test UI."""
    return FileResponse("ui_test.html", media_type="text/html")


@app.get("/healthz", tags=["Health"], summary="Liveness check")
def healthz() -> Any:
    """Lightweight endpoint used by the frontend to warm sleeping deployments."""
    return {"status": "ok", "service": "monte-carlo-prop-firm-simulator"}


@app.post("/strategies/upload", tags=["Strategies"], summary="Upload a strategy CSV")
async def upload_strategy(file: UploadFile = File(...)) -> Any:
    """
    Upload a CSV strategy file.  Returns a ``strategy_id`` that can be used
    in place of ``csv_path`` in all simulation endpoints.

    Only ``.csv`` files are accepted.  The file is saved to
    ``strategies/<uuid>.csv`` and recorded in the SQLite registry.

    If the identical file has been uploaded before (detected via SHA-256
    hash), the existing strategy is returned with ``"duplicate": true``
    and no extra file is stored.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        return _error("Only CSV files supported", 400)

    # Read file content into memory so we can hash it and then save it
    content = await file.read()
    await file.close()

    file_hash = hashlib.sha256(content).hexdigest()

    # ── Deduplication check ──────────────────────────────────────────────────
    existing = strategy_db.find_by_hash(file_hash)
    if existing is not None:
        return {
            "strategy_id": existing["strategy_id"],
            "filename":    existing["filename"],
            "path":        existing["path"],
            "source":      existing.get("source", "tradingview"),
            "duplicate":   True,
        }

    # ── New upload ───────────────────────────────────────────────────────────
    strategy_id = uuid4().hex
    dest = STRATEGIES_DIR / f"{strategy_id}.csv"
    dest.write_bytes(content)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    strategy_db.insert_strategy(
        strategy_id = strategy_id,
        filename    = file.filename,
        path        = str(dest),
        uploaded_at = uploaded_at,
        file_hash   = file_hash,
    )

    # ── Feature extraction (non-blocking; errors are logged, not raised) ─────
    try:
        features = strategy_analyzer.analyze_strategy(str(dest))
        strategy_db.insert_strategy_features(strategy_id, features)
    except Exception as exc:  # pragma: no cover
        log.warning("upload_strategy: feature extraction failed for %s — %s", strategy_id, exc)
        features = {}

    return {
        "strategy_id": strategy_id,
        "filename":    file.filename,
        "path":        str(dest),
        "source":      "tradingview",
        "duplicate":   False,
        "features":    features,
    }


@app.get(
    "/strategies/features",
    tags=["Strategies"],
    summary="List all strategy features",
)
def list_strategy_features_endpoint() -> Any:
    """
    Return the extracted statistical features for every uploaded strategy,
    sorted by most recently analysed first.
    """
    return strategy_db.list_all_strategy_features()


@app.get(
    "/strategies/{strategy_id}/features",
    tags=["Strategies"],
    summary="Get features for a single strategy",
)
def get_strategy_features_endpoint(strategy_id: str) -> Any:
    """
    Return the extracted statistical features for *strategy_id*.

    Features are computed automatically when the strategy CSV is uploaded.
    Returns a 404 error if no features have been stored for this strategy.
    """
    row = strategy_db.get_strategy_features(strategy_id)
    if row is None:
        return _error(f"No features found for strategy_id: {strategy_id!r}", 404)
    return {"strategy_id": strategy_id, "features": row}


@app.get("/strategies", tags=["Strategies"], summary="List registered strategies")
def list_strategies_endpoint(source: Optional[str] = Query(None)) -> Any:
    """
    Return all previously uploaded strategies sorted by newest upload first.
    Each entry includes strategy_id, filename, path, uploaded_at, and file_hash.
    """
    return strategy_db.list_strategies(source=source)


@app.delete(
    "/strategies/{strategy_id}",
    tags=["Strategies"],
    summary="Delete a registered strategy",
)
def delete_strategy_endpoint(strategy_id: str) -> Any:
    """
    Delete a strategy from the registry and remove its CSV file from disk.
    Returns ``{"deleted": true}`` on success or a 404 error if not found.
    """
    entry = strategy_db.get_strategy(strategy_id)
    if entry is None:
        return _error(f"strategy_id not found: {strategy_id!r}", 404)

    # Remove CSV file from disk (best-effort)
    csv_file = Path(entry["path"])
    if csv_file.exists():
        try:
            csv_file.unlink()
        except OSError as exc:
            return _error(f"Could not delete file: {exc}", 500)

    strategy_db.delete_strategy(strategy_id)
    strategy_db.delete_simulation_results(strategy_id)
    return {"deleted": True, "strategy_id": strategy_id}


@app.post(
    "/analyze/until_payout",
    tags=["Single Strategy"],
    summary="Run-until-payout Monte Carlo analysis",
    response_description=(
        "Nested dict with metadata, probabilities, metrics, and equity path distributions."
    ),
)
def endpoint_until_payout(req: UntilPayoutRequest) -> Any:
    """
    Simulate an Apex 50k account running until it either pays out, blows, or
    times out.  No fixed time window — the pure edge ceiling.

    Returns a dict with:
    - **metadata** – run parameters snapshot
    - **probabilities** – payout_prob, blow_prob, timeout_prob
    - **metrics** – mean/median days, mean_payout, tier rating
    - **distributions** – equity_paths (list of simulated equity curves)
    """
    try:
        csv_path = _resolve_csv(req.csv_path, req.strategy_id)
        if _is_mt5_strategy(req.strategy_id):
            personal, overrides = _run_mt5_personal_simulation(
                csv_path,
                req,
                stop_at_payout=True,
            )
            result = _map_personal_until_payout(
                personal,
                csv_path=csv_path,
                n_sims=req.n_sims,
                risk_multiplier=req.risk_multiplier,
                max_days=overrides["max_days"],
            )
            if req.strategy_id:
                _store_until_payout(req.strategy_id, result, req.n_sims)
            return result
        result = analyze_until_payout(
            csv_path        = csv_path,
            n_sims          = req.n_sims,
            risk_multiplier = req.risk_multiplier,
            max_days        = req.max_days,
            n_paths         = req.n_paths,
            sampling_mode   = req.sampling_mode,
            weight_strength = req.weight_strength,
            recent_window   = req.recent_window,
            seed            = req.seed,
        )
        if req.strategy_id:
            _store_until_payout(req.strategy_id, result, req.n_sims)
        return result
    except FileNotFoundError as exc:
        return _error(f"CSV not found: {exc}", 400)
    except Exception as exc:
        return _error(str(exc), 400)


async def _stream_single_strategy(req: Any, *, mode: str):
    """Stream single-strategy simulation progress and final result via SSE."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    total = max(1, int(getattr(req, "n_sims", 1)))

    def _progress_cb(completed: int, total_sims: int) -> None:
        loop.call_soon_threadsafe(
            q.put_nowait,
            {"type": "progress", "completed": int(completed), "total": int(total_sims)},
        )

    def _execute_sync() -> dict:
        csv_path = _resolve_csv(req.csv_path, req.strategy_id)

        if mode == "until_payout":
            if _is_mt5_strategy(req.strategy_id):
                personal, overrides = _run_mt5_personal_simulation(
                    csv_path,
                    req,
                    stop_at_payout=True,
                    progress_cb=_progress_cb,
                )
                result = _map_personal_until_payout(
                    personal,
                    csv_path=csv_path,
                    n_sims=req.n_sims,
                    risk_multiplier=req.risk_multiplier,
                    max_days=overrides["max_days"],
                )
                if req.strategy_id:
                    _store_until_payout(req.strategy_id, result, req.n_sims)
                return result

            result = analyze_until_payout(
                csv_path=csv_path,
                n_sims=req.n_sims,
                risk_multiplier=req.risk_multiplier,
                max_days=req.max_days,
                n_paths=req.n_paths,
                sampling_mode=req.sampling_mode,
                weight_strength=req.weight_strength,
                recent_window=req.recent_window,
                seed=req.seed,
            )
            if req.strategy_id:
                _store_until_payout(req.strategy_id, result, req.n_sims)
            return result

        if mode == "full_period":
            if _is_mt5_strategy(req.strategy_id):
                personal, overrides = _run_mt5_personal_simulation(
                    csv_path,
                    req,
                    stop_at_payout=False,
                    progress_cb=_progress_cb,
                )
                result = _map_personal_full_period(
                    personal,
                    csv_path=csv_path,
                    n_sims=req.n_sims,
                    risk_multiplier=req.risk_multiplier,
                    max_days=overrides["max_days"],
                    reset_cost=req.reset_cost,
                )
                if req.strategy_id:
                    _store_full_period(req.strategy_id, result, req.n_sims)
                return result

            result = analyze_full_period(
                csv_path=csv_path,
                n_sims=req.n_sims,
                risk_multiplier=req.risk_multiplier,
                max_days=req.max_days,
                n_paths=req.n_paths,
                reset_cost=req.reset_cost,
                sampling_mode=req.sampling_mode,
                weight_strength=req.weight_strength,
                recent_window=req.recent_window,
                seed=req.seed,
            )
            if req.strategy_id:
                _store_full_period(req.strategy_id, result, req.n_sims)
            return result

        raise ValueError(f"Unknown stream mode: {mode!r}")

    async def _run() -> None:
        try:
            await q.put({"type": "progress", "completed": 0, "total": total})
            result = await loop.run_in_executor(None, _execute_sync)
            await q.put({"type": "progress", "completed": total, "total": total})
            await q.put({"type": "done", "data": result})
        except FileNotFoundError as exc:
            await q.put({"type": "error", "message": f"CSV not found: {exc}"})
        except Exception as exc:
            await q.put({"type": "error", "message": str(exc)})

    async def _generate():
        task = asyncio.create_task(_run())
        while True:
            msg = await q.get()
            yield f"data: {_json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break
        await task

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post(
    "/analyze/until_payout/stream",
    tags=["Single Strategy"],
    summary="Run-until-payout with live simulation progress",
)
async def endpoint_until_payout_stream(req: UntilPayoutRequest):
    return await _stream_single_strategy(req, mode="until_payout")


@app.post(
    "/analyze/full_period",
    tags=["Single Strategy"],
    summary="Fixed-window (monthly) Monte Carlo analysis",
    response_description=(
        "Nested dict with metadata, five-way probability split, metrics, and equity paths."
    ),
)
def endpoint_full_period(req: FullPeriodRequest) -> Any:
    """
    Simulate an Apex 50k account over a fixed trading-day window
    (default 21 days ≈ one calendar month).

    Returns a dict with:
    - **metadata** – run parameters snapshot
    - **probabilities** – five-way split: payout_prob, payout_survived_prob,
      payout_then_blew_prob, blow_no_payout_prob, timeout_no_pay_prob
    - **metrics** – mean_ending_balance, mean/median payout, e_monthly, tier rating
    - **distributions** – equity_paths
    """
    try:
        csv_path = _resolve_csv(req.csv_path, req.strategy_id)
        if _is_mt5_strategy(req.strategy_id):
            personal, overrides = _run_mt5_personal_simulation(
                csv_path,
                req,
                stop_at_payout=False,
            )
            result = _map_personal_full_period(
                personal,
                csv_path=csv_path,
                n_sims=req.n_sims,
                risk_multiplier=req.risk_multiplier,
                max_days=overrides["max_days"],
                reset_cost=req.reset_cost,
            )
            if req.strategy_id:
                _store_full_period(req.strategy_id, result, req.n_sims)
            return result
        result = analyze_full_period(
            csv_path        = csv_path,
            n_sims          = req.n_sims,
            risk_multiplier = req.risk_multiplier,
            max_days        = req.max_days,
            n_paths         = req.n_paths,
            reset_cost      = req.reset_cost,
            sampling_mode   = req.sampling_mode,
            weight_strength = req.weight_strength,
            recent_window   = req.recent_window,
            seed            = req.seed,
        )
        if req.strategy_id:
            _store_full_period(req.strategy_id, result, req.n_sims)
        return result
    except FileNotFoundError as exc:
        return _error(f"CSV not found: {exc}", 400)
    except Exception as exc:
        return _error(str(exc), 400)


@app.post(
    "/analyze/full_period/stream",
    tags=["Single Strategy"],
    summary="Run full-period with live simulation progress",
)
async def endpoint_full_period_stream(req: FullPeriodRequest):
    return await _stream_single_strategy(req, mode="full_period")


@app.post(
    "/analyze/batch",
    tags=["Single Strategy"],
    summary="Batch simulation across multiple CSVs",
    response_description=(
        "List of result dicts sorted by full-period net EV descending. "
        "Failed CSVs appear as {csv, error} entries."
    ),
)
def endpoint_batch(req: BatchRequest) -> Any:
    """
    Run both simulation modes (until-payout + full-period) on every CSV in
    the list and return a ranked table sorted by full-period net EV.

    Individual CSV failures are captured as `{"csv": "...", "error": "..."`}
    entries rather than aborting the whole batch.
    """
    try:
        if req.strategy_ids and _sources_for_strategy_ids(req.strategy_ids) == {"mt5"}:
            results = _run_personal_batch(req.strategy_ids, req)
        else:
            csv_paths = _resolve_csv_list(req.csv_paths, req.strategy_ids)
            results = run_batch(
                csv_paths       = csv_paths,
                n_sims          = req.n_sims,
                sampling_mode   = req.sampling_mode,
                weight_strength = req.weight_strength,
                recent_window   = req.recent_window,
                reset_cost      = req.reset_cost,
                seed            = req.seed,
            )
        # Persist each batch row — match by csv filename, not by position,
        # because results are EV-sorted and no longer align with strategy_ids.
        if req.strategy_ids:
            sid_map = _build_sid_map(req.strategy_ids)
            for row in results:
                if not isinstance(row, dict) or "error" in row:
                    continue
                sid = sid_map.get(row.get("csv", ""))
                if not sid:
                    continue
                try:
                    strategy_db.insert_simulation_result(
                        strategy_id             = sid,
                        simulation_type         = "batch",
                        pass_probability        = row.get("utp_payout_p") or row.get("fp_payout_p"),
                        expected_monthly_payout = row.get("fp_ev_net"),
                        num_simulations         = req.n_sims,
                        recent_pass_probability = row.get("recent_pass_probability"),
                        probability_delta       = row.get("probability_delta"),
                        recency_status          = row.get("recency_status"),
                    )
                except Exception as exc:
                    log.warning("leaderboard: could not store batch result for %s: %s", sid, exc)
        return results
    except Exception as exc:
        return _error(str(exc), 400)


@app.post(
    "/analyze/batch/stream",
    tags=["Single Strategy"],
    summary="Batch simulation with live SSE progress",
)
async def endpoint_batch_stream(req: BatchRequest):
    """
    Identical to /analyze/batch but streams Server-Sent Events so the UI
    can show a live per-file progress bar.  Each event is a JSON object:

    ``{"type":"progress","completed":n,"total":N,"file":"name.csv"}``
    ``{"type":"done","data":[...]}``
    ``{"type":"error","message":"..."}``
    """
    mt5_personal_batch = False
    csv_paths: List[str] = []
    try:
        if req.strategy_ids and _sources_for_strategy_ids(req.strategy_ids) == {"mt5"}:
            mt5_personal_batch = True
        else:
            csv_paths = _resolve_csv_list(req.csv_paths, req.strategy_ids)
    except Exception as exc:
        async def _err_gen():
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"
        return StreamingResponse(_err_gen(), media_type="text/event-stream")

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Build stored-filename → friendly display name so the progress bar
    # shows the original uploaded name rather than the UUID-based filename.
    _name_map: dict[str, str] = {}
    if req.strategy_ids:
        for sid in req.strategy_ids:
            try:
                entry = strategy_db.get_strategy(sid)
                if entry:
                    _name_map[Path(entry["path"]).name] = entry["filename"]
            except Exception:
                pass

    def _progress_cb(completed: int, total: int, csv_path: str) -> None:
        raw = Path(csv_path).name
        loop.call_soon_threadsafe(q.put_nowait, {
            "type": "progress",
            "completed": completed,
            "total": total,
            "file": _name_map.get(raw, raw),
        })

    async def _run() -> None:
        try:
            if mt5_personal_batch and req.strategy_ids:
                results = await loop.run_in_executor(
                    None,
                    lambda: _run_personal_batch(req.strategy_ids or [], req, progress_cb=_progress_cb),
                )
            else:
                results = await loop.run_in_executor(
                    None,
                    lambda: run_batch(
                        csv_paths       = csv_paths,
                        n_sims          = req.n_sims,
                        sampling_mode   = req.sampling_mode,
                        weight_strength = req.weight_strength,
                        recent_window   = req.recent_window,
                        reset_cost      = req.reset_cost,
                        seed            = req.seed,
                        progress_cb     = _progress_cb,
                    ),
                )
            if req.strategy_ids:
                sid_map = _build_sid_map(req.strategy_ids)
                for row in results:
                    if not isinstance(row, dict) or "error" in row:
                        continue
                    sid = sid_map.get(row.get("csv", ""))
                    if not sid:
                        continue
                    try:
                        strategy_db.insert_simulation_result(
                            strategy_id             = sid,
                            simulation_type         = "batch",
                            pass_probability        = row.get("utp_payout_p") or row.get("fp_payout_p"),
                            expected_monthly_payout = row.get("fp_ev_net"),
                            num_simulations         = req.n_sims,
                            recent_pass_probability = row.get("recent_pass_probability"),
                            probability_delta       = row.get("probability_delta"),
                            recency_status          = row.get("recency_status"),
                        )
                    except Exception as exc2:
                        log.warning("leaderboard: batch stream store failed for %s: %s", sid, exc2)
            await q.put({"type": "done", "data": results})
        except Exception as exc:
            await q.put({"type": "error", "message": str(exc)})

    async def _generate():
        task = asyncio.create_task(_run())
        while True:
            msg = await q.get()
            yield f"data: {_json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break
        await task

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post(
    "/portfolio/correlation",
    tags=["Portfolio"],
    summary="Pairwise strategy correlation and simultaneous drawdown analysis",
    response_description=(
        "Dict with return/blow correlation matrices, rolling correlations, "
        "simultaneous drawdown events, worst overlap periods, and equity curves."
    ),
)
def endpoint_correlation(req: CorrelationRequest) -> Any:
    """
    Compute pairwise return correlations, Monte Carlo blow-up correlations,
    rolling correlations, and simultaneous drawdown risk across two or more
    strategies.

    `strategies` must map strategy name → CSV path, e.g.:
    ```json
    { "ES": "MultiBot_ES.csv", "YM": "MultiBot_YM.csv" }
    ```

    Returns:
    - **return_correlation** – {labels, matrix} Pearson return correlation
    - **blow_correlation** – {labels, matrix} MC blow-up correlation
    - **rolling_pairs** – per-pair rolling correlation time series
    - **simultaneous_dd** – daily table of how many strategies are in drawdown
    - **worst_overlap** – top worst simultaneous drawdown periods
    - **equity_curves** / **drawdown_curves** – full time series per strategy
    """
    try:
        named_csvs = _resolve_named_csvs(req.strategies, req.strategy_ids)
        result = analyze_correlation(
            named_csvs      = named_csvs,
            rolling_window  = req.rolling_window,
            dd_threshold    = req.dd_threshold,
            mc_n_sims       = req.mc_n_sims,
            mc_max_days     = req.mc_max_days,
            sampling_mode   = req.sampling_mode,
            weight_strength = req.weight_strength,
            recent_window   = req.recent_window,
            seed            = req.seed,
        )
        return result
    except ValueError as exc:
        # e.g. fewer than 2 strategies provided
        return _error(str(exc), 400)
    except FileNotFoundError as exc:
        return _error(f"CSV not found: {exc}", 400)
    except Exception as exc:
        return _error(str(exc), 400)


@app.post(
    "/portfolio/multi_account",
    tags=["Portfolio"],
    summary="N concurrent Apex accounts — portfolio Monte Carlo simulation",
    response_description=(
        "Dict with metadata, portfolio aggregate statistics, and cycle efficiency metrics."
    ),
)
def endpoint_multi_account(req: MultiAccountRequest) -> Any:
    """
    Simulate a portfolio of N identical Apex 50k accounts running
    concurrently on the same strategy, and report aggregate payout
    distribution, blowout frequency, and monthly cycle efficiency.

    Returns:
    - **metadata** – run parameters snapshot
    - **portfolio** – mean/median/P5/P95 total payout, blow rates, end balances
    - **cycle_efficiency** – estimated cycles per month, monthly gross/net EV
    """
    try:
        csv_path = _resolve_csv(req.csv_path, req.strategy_id)
        if _is_mt5_strategy(req.strategy_id):
            overrides = _mt5_overrides_from_request(req)
            trades = _load_trade_results_from_strategy_csv(csv_path)
            result = _run_personal_multi_account(
                trades=trades,
                n_accounts=req.n_accounts,
                n_sims=req.n_sims,
                stop_at_payout=req.stop_at_payout,
                n_path_sims=req.n_path_sims,
                overrides=overrides,
            )
        else:
            result = run_multi_account(
                csv_path        = csv_path,
                n_accounts      = req.n_accounts,
                n_sims          = req.n_sims,
                max_days        = req.max_days,
                stop_at_payout  = req.stop_at_payout,
                risk_multiplier = req.risk_multiplier,
                sampling_mode   = req.sampling_mode,
                weight_strength = req.weight_strength,
                recent_window   = req.recent_window,
                n_path_sims     = req.n_path_sims,
                seed            = req.seed,
            )
        if req.strategy_id:
            _store_multi_account(req.strategy_id, result, req.n_sims)
        return result
    except FileNotFoundError as exc:
        return _error(f"CSV not found: {exc}", 400)
    except Exception as exc:
        return _error(str(exc), 400)


@app.get(
    "/leaderboard",
    tags=["Leaderboard"],
    summary="Ranked strategy leaderboard",
    response_description="List of strategies ranked by the chosen metric.",
)
def endpoint_leaderboard(
    simulation_type: str = "until_payout",
    metric: str = "pass_probability",
    limit: int = 20,
) -> Any:
    """
    Return strategies ranked by *metric* for the given *simulation_type*.

    **simulation_type** options: `until_payout`, `full_period`, `batch`, `multi_account`

    **metric** options:
    - `pass_probability` — highest payout probability first
    - `expected_monthly_payout` — highest expected $ per month first
    - `sharpe` — best risk-adjusted return first
    - `profit_factor` — best profit factor first
    - `max_drawdown` — lowest drawdown first (safest)

    Each strategy appears only once (most recent run for that sim type).
    """
    rows = strategy_db.get_leaderboard(
        simulation_type = simulation_type,
        metric          = metric,
        limit           = max(1, min(limit, 100)),
    )
    return rows


@app.get(
    "/strategy/{strategy_id}/performance",
    tags=["Leaderboard"],
    summary="All simulation records for a strategy",
)
def endpoint_strategy_performance(strategy_id: str) -> Any:
    """
    Return every simulation result stored for *strategy_id*, newest first.
    """
    entry = strategy_db.get_strategy(strategy_id)
    if entry is None:
        return _error(f"strategy_id not found: {strategy_id!r}", 404)
    records = strategy_db.get_strategy_performance(strategy_id)
    return {
        "strategy_id": strategy_id,
        "filename":    entry["filename"],
        "simulations": records,
    }


@app.post(
    "/portfolio/optimize",
    tags=["Portfolio"],
    summary="Multi-stage tournament search for optimal account allocation",
    response_description=(
        "Dict with 'portfolios' list (ranked by expected total payout) and 'n_strategies' count."
    ),
)
def endpoint_optimize(req: OptimizeRequest) -> Any:
    """
    Run the full tournament search from portfolio_optimizer.py via the API.

    Enumerates all strategy-subset × account-allocation candidates, then
    eliminates them through a multi-stage tournament (fast pre-filter →
    deep MC → stress test) and returns the top portfolios ranked by
    expected total payout per cycle.

    **Note:** this can take 30 s – several minutes depending on the number
    of strategies and accounts.  Progress is printed to the server console.

    Returns:
    - **portfolios** – ranked list:
      strategies, allocation, expected_total_payout, expected_payouts,
      avg_correlation, score, payout_count_dist
    - **n_strategies** – number of strategies included in the search
    """
    try:
        if _sources_for_strategy_ids(req.strategy_ids) == {"mt5"}:
            return _error(
            "MT5 personal mode is not supported in optimizer yet. Use until_payout, full_period, batch, or multi_account for MT5.",
            400,
            )
        # ── Resolve strategy_ids → PnL arrays ────────────────────────────────
        strategy_pnl: Dict[str, Any] = {}
        for sid in req.strategy_ids:
            entry = strategy_db.get_strategy(sid)
            if entry is None:
                return _error(f"strategy_id not found: {sid!r}", 400)
            # Use filename stem as the strategy name
            name = Path(entry["filename"]).stem
            base, counter = name, 2
            while name in strategy_pnl:
                name = f"{base}_{counter}"
                counter += 1
            strategy_pnl[name] = load_strategy_pnl(entry["path"])

        if not strategy_pnl:
            return _error("At least 1 strategy required.", 400)

        # ── Correlation matrix ────────────────────────────────────────────────
        corr_df = compute_correlation_matrix(strategy_pnl)

        # ── Tournament search ─────────────────────────────────────────────────
        ranked = find_optimal_portfolios(
            strategy_pnl   = strategy_pnl,
            corr_matrix    = corr_df,
            n_accounts     = req.n_accounts,
            max_strategies = req.max_strategies,
            max_days       = req.max_days,
            top_n          = req.top_n,
            seed           = req.seed or 42,
            k_thresholds   = req.k_thresholds,
        )

        # ── Serialize (convert numpy scalars, tuples → lists, int keys) ───────
        portfolios = []
        for p in ranked:
            portfolios.append({
                "strategies":            p["strategies"],
                "allocation":            list(p["allocation"]),
                "avg_correlation":       float(p["avg_correlation"]),
                "score":                 float(p["score"]),
                "n_accounts":            int(p["n_accounts"]),
                "expected_payouts":      float(p["expected_payouts"]),
                "expected_total_payout": float(p["expected_total_payout"]),
                "payout_count_dist": {
                    int(k): float(v)
                    for k, v in p["payout_count_dist"].items()
                },
            })

        return {"portfolios": portfolios, "n_strategies": len(strategy_pnl)}

    except FileNotFoundError as exc:
        return _error(f"CSV not found: {exc}", 400)
    except Exception as exc:
        return _error(str(exc), 400)

@app.post(
    "/portfolio/optimize/stream",
    tags=["Portfolio"],
    summary="Portfolio optimizer with live SSE stage/progress updates",
)
async def endpoint_optimize_stream(req: OptimizeRequest):
    """
    Identical to /portfolio/optimize but streams Server-Sent Events so the UI
    can show a live progress bar during the (potentially multi-minute) search.

    Event types:
    ``{"type":"stage","stage":1,"n_stages":4,"candidates":240,"sims":100}``
    ``{"type":"progress","stage":1,"n_stages":4,"completed":50,"total":240}``
    ``{"type":"done","data":{...}}``
    ``{"type":"error","message":"..."}``
    """
    if _sources_for_strategy_ids(req.strategy_ids) == {"mt5"}:
        async def _err_gen():
            yield f"data: {_json.dumps({'type':'error','message':'MT5 personal mode is not supported in optimizer yet. Use until_payout, full_period, batch, or multi_account for MT5.'})}\n\n"
        return StreamingResponse(_err_gen(), media_type="text/event-stream")

    # Resolve strategies up front (fast) so we can report errors before streaming
    try:
        strategy_pnl: Dict[str, Any] = {}
        for sid in req.strategy_ids:
            entry = strategy_db.get_strategy(sid)
            if entry is None:
                raise ValueError(f"strategy_id not found: {sid!r}")
            name = Path(entry["filename"]).stem
            base, counter = name, 2
            while name in strategy_pnl:
                name = f"{base}_{counter}"
                counter += 1
            strategy_pnl[name] = load_strategy_pnl(entry["path"])
        if not strategy_pnl:
            raise ValueError("At least 1 strategy required.")
        corr_df = compute_correlation_matrix(strategy_pnl)
    except Exception as exc:
        async def _err_gen():
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"
        return StreamingResponse(_err_gen(), media_type="text/event-stream")

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _progress_cb(msg: dict) -> None:
        loop.call_soon_threadsafe(q.put_nowait, msg)

    def _run_optimizer():
        ranked = find_optimal_portfolios(
            strategy_pnl   = strategy_pnl,
            corr_matrix    = corr_df,
            n_accounts     = req.n_accounts,
            max_strategies = req.max_strategies,
            max_days       = req.max_days,
            top_n          = req.top_n,
            seed           = req.seed or 42,
            k_thresholds   = req.k_thresholds,
            progress_cb    = _progress_cb,
        )
        portfolios = []
        for p in ranked:
            portfolios.append({
                "strategies":            p["strategies"],
                "allocation":            list(p["allocation"]),
                "avg_correlation":       float(p["avg_correlation"]),
                "score":                 float(p["score"]),
                "n_accounts":            int(p["n_accounts"]),
                "expected_payouts":      float(p["expected_payouts"]),
                "expected_total_payout": float(p["expected_total_payout"]),
                "payout_count_dist": {
                    int(k): float(v)
                    for k, v in p["payout_count_dist"].items()
                },
            })
        return {"portfolios": portfolios, "n_strategies": len(strategy_pnl)}

    async def _run() -> None:
        try:
            result = await loop.run_in_executor(None, _run_optimizer)
            await q.put({"type": "done", "data": result})
        except Exception as exc:
            await q.put({"type": "error", "message": str(exc)})

    async def _generate():
        task = asyncio.create_task(_run())
        while True:
            msg = await q.get()
            yield f"data: {_json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break
        await task

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

@app.post(
    "/analyze/rescue",
    tags=["Analysis"],
    summary="What-If / Rescue Analysis — best strategy from a compromised account state",
)
async def endpoint_rescue(req: RescueRequest):
    """
    Given the current mid-account state (``current_balance`` + ``trailing_stop_floor``),
    runs Monte-Carlo for every requested strategy and returns them ranked by
    ``pass_probability`` descending.  Useful for deciding which strategy gives
    the best chance of recovery when the account is under pressure.
    """
    try:
        if _sources_for_strategy_ids(req.strategy_ids) == {"mt5"}:
            return _error(
                "MT5 personal mode is not supported in rescue yet. Use until_payout, full_period, batch, or multi_account for MT5.",
                400,
            )
        csv_paths: List[str] = []
        path_to_filename: Dict[str, str] = {}
        for sid in req.strategy_ids:
            entry = strategy_db.get_strategy(sid)
            if entry is None:
                return _error(f"strategy_id not found: {sid!r}", 400)
            csv_paths.append(entry["path"])
            path_to_filename[entry["path"]] = entry["filename"]

        if not csv_paths:
            return _error("At least one strategy is required.", 400)

        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: analyze_rescue(
                csv_paths            = csv_paths,
                current_balance      = req.current_balance,
                trailing_stop_floor  = req.trailing_stop_floor,
                n_sims               = req.n_sims,
                max_days             = req.max_days,
                seed                 = req.seed,
            ),
        )
        # Annotate each result with the human-readable filename
        for r in results:
            r["filename"] = path_to_filename.get(r.get("csv", ""), "")
        return {"results": results, "n_strategies": len(csv_paths)}

    except Exception as exc:
        return _error(str(exc), 400)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
