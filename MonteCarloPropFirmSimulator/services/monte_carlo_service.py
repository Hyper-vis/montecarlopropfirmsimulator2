"""Service wrapper to run Monte Carlo from trade-level PnL input."""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_engine_v2_0 import run_monte_carlo


class MonteCarloServiceError(ValueError):
    """Raised when simulation cannot be run or is not reliable."""


DEFAULT_CONFIG: dict[str, Any] = {
    "account_size": 50_000.0,
    "trailing_dd": 2_500.0,
    "trail_stop_level": 50_100.0,
    "payout_threshold": 52_600.0,
    "safety_net_level": 52_100.0,
    "max_payout_limit": 2_000.0,
    "max_days": 90,
    "max_trades_per_day": 10,
    "max_losing_per_day": 3,
    "max_winning_per_day": 10,
    "daily_loss_limit": 700.0,
    "daily_profit_cap": 1_050.0,
    "min_days": 8,
    "min_green_days": 5,
    "green_day_min": 50.0,
    "risk_multiplier": 1.0,
    "n_sims": 10_000,
}


def _build_config(
    *,
    n_sims: int,
    profile: str,
    config_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["n_sims"] = int(max(100, n_sims))
    if config_overrides:
        for k, v in config_overrides.items():
            if v is not None:
                config[k] = v

    if profile == "personal":
        # Personal-account mode: static floor + no prop payout eligibility gates.
        account_size = float(config["account_size"])
        overall_max_loss = float(config.get("overall_max_loss", config["trailing_dd"]))
        overall_max_loss = max(1.0, overall_max_loss)

        config["trailing_dd"] = overall_max_loss
        config["trail_stop_level"] = account_size - overall_max_loss
        config["safety_net_level"] = account_size
        config["max_payout_limit"] = 1_000_000_000.0
        config["min_days"] = 1
        config["min_green_days"] = 0
        config["green_day_min"] = 0.0
        config["max_losing_per_day"] = 10_000
        config["max_winning_per_day"] = 10_000

        daily_loss = float(config.get("daily_loss_limit", 700.0))
        config["daily_loss_limit"] = abs(daily_loss)

    return config


def run_trade_simulation_profile(
    trade_results: list[float],
    *,
    n_sims: int = 10_000,
    stop_at_payout: bool = True,
    profile: str = "prop",
    config_overrides: dict[str, Any] | None = None,
) -> dict:
    """Run a Monte Carlo simulation from per-trade PnL values with profile overrides."""
    clean = [float(v) for v in trade_results if v is not None]
    if not clean:
        raise MonteCarloServiceError("This MT5 report contains no trades.")

    if len(clean) < 10:
        raise MonteCarloServiceError(
            "This MT5 report contains too few trades for reliable simulation."
        )

    config = _build_config(
        n_sims=n_sims,
        profile=profile,
        config_overrides=config_overrides,
    )

    try:
        mc = run_monte_carlo(np.asarray(clean, dtype=float), config, stop_at_payout=stop_at_payout)
    except Exception as exc:
        raise MonteCarloServiceError("Simulation failed for this trade set.") from exc

    outcomes = list(mc.get("outcomes", []))
    payouts = [float(v) for v in mc.get("payouts", [])]
    n = len(outcomes) if outcomes else int(config["n_sims"])
    if n <= 0:
        raise MonteCarloServiceError("Simulation returned no outcomes.")

    payout_count = outcomes.count("payout")
    blow_count = outcomes.count("blow")
    payout_values = [p for p in payouts if p > 0]

    pass_probability = payout_count / n
    blow_probability = blow_count / n
    expected_payout = float(np.mean(payout_values)) if payout_values else 0.0

    return {
        "profile": profile,
        "config": config,
        "num_trades": len(clean),
        "pass_probability": round(pass_probability, 4),
        "blow_probability": round(blow_probability, 4),
        "timeout_probability": round(max(0.0, 1.0 - pass_probability - blow_probability), 4),
        "expected_payout": round(expected_payout, 2),
        "outcomes": outcomes,
        "payouts": payouts,
        "balances": [float(v) for v in mc.get("balances", [])],
        "days": [int(v) for v in mc.get("days", [])],
        "paths": [list(map(float, p)) for p in mc.get("paths", [])],
        "trade_sample": [round(v, 2) for v in clean[:3]],
    }


def run_trade_simulation(trade_results: list[float], n_sims: int = 10_000) -> dict:
    """Backward-compatible prop profile wrapper used by existing upload handlers."""
    return run_trade_simulation_profile(
        trade_results,
        n_sims=n_sims,
        stop_at_payout=True,
        profile="prop",
    )
