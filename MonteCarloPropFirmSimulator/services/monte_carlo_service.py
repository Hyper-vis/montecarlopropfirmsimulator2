"""Service wrapper to run Monte Carlo from trade-level PnL input."""

from __future__ import annotations

from typing import Any, Callable

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
        config["daily_profit_cap"] = 1_000_000_000.0
        config["min_days"] = 1
        config["min_green_days"] = 0
        config["green_day_min"] = 0.0
        config["max_losing_per_day"] = 10_000
        config["max_winning_per_day"] = 10_000
        config["require_consistency_rule"] = False

        daily_loss = float(config.get("daily_loss_limit", 700.0))
        config["daily_loss_limit"] = abs(daily_loss)

    return config


def _build_daily_config(
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
        account_size = float(config["account_size"])
        overall_max_loss = float(config.get("overall_max_loss", config["trailing_dd"]))
        overall_max_loss = max(1.0, overall_max_loss)

        config["trailing_dd"] = overall_max_loss
        config["trail_stop_level"] = account_size - overall_max_loss
        config["safety_net_level"] = account_size
        config["max_payout_limit"] = 1_000_000_000.0
        config["daily_profit_cap"] = 1_000_000_000.0
        config["min_days"] = 1
        config["min_green_days"] = 0
        config["green_day_min"] = 0.0
        config["max_losing_per_day"] = 10_000
        config["max_winning_per_day"] = 10_000
        config["require_consistency_rule"] = False

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
    progress_cb: Callable[[int, int], None] | None = None,
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
        mc = run_monte_carlo(
            np.asarray(clean, dtype=float),
            config,
            stop_at_payout=stop_at_payout,
            progress_cb=progress_cb,
        )
    except Exception as exc:
        raise MonteCarloServiceError("Simulation failed for this trade set.") from exc

    outcomes = list(mc.get("outcomes", []))
    payouts = [float(v) for v in mc.get("payouts", [])]
    balances = [float(v) for v in mc.get("balances", [])]
    paths = [list(map(float, p)) for p in mc.get("paths", [])]
    days = [int(v) for v in mc.get("days", [])]
    n = len(outcomes) if outcomes else int(config["n_sims"])
    if n <= 0:
        raise MonteCarloServiceError("Simulation returned no outcomes.")

    payout_count = outcomes.count("payout")
    blow_count = outcomes.count("blow")
    pass_probability = payout_count / n
    blow_probability = blow_count / n
    timeout_probability = max(0.0, 1.0 - pass_probability - blow_probability)
    expected_payout = float(np.mean([p for p in payouts if p > 0])) if payouts else 0.0

    target_achieved_probability = pass_probability
    target_achieved_days: list[int] = [d for o, d in zip(outcomes, days) if o == "payout"]

    if profile == "personal":
        target = float(config["payout_threshold"])
        account_size = float(config["account_size"])

        target_hit_flags: list[bool] = []
        target_hit_gains: list[float] = []
        target_achieved_days = []
        for idx, path in enumerate(paths):
            hit = any(v >= target for v in path)
            target_hit_flags.append(hit)
            if hit:
                target_achieved_days.append(days[idx] if idx < len(days) else int(config["max_days"]))
                first_hit_balance = next(v for v in path if v >= target)
                target_hit_gains.append(float(first_hit_balance - account_size))

        target_achieved_probability = float(np.mean(target_hit_flags)) if target_hit_flags else 0.0
        pass_probability = target_achieved_probability
        timeout_probability = max(0.0, 1.0 - pass_probability - blow_probability)
        expected_payout = float(np.mean(target_hit_gains)) if target_hit_gains else 0.0

    return {
        "profile": profile,
        "config": config,
        "num_trades": len(clean),
        "pass_probability": round(pass_probability, 4),
        "blow_probability": round(blow_probability, 4),
        "timeout_probability": round(timeout_probability, 4),
        "target_achieved_probability": round(target_achieved_probability, 4),
        "target_achieved_days": target_achieved_days,
        "expected_payout": round(expected_payout, 2),
        "outcomes": outcomes,
        "payouts": payouts,
        "balances": balances,
        "days": days,
        "paths": paths,
        "trade_sample": [round(v, 2) for v in clean[:3]],
    }


def run_daily_simulation_profile(
    daily_results: list[float],
    *,
    n_sims: int = 10_000,
    stop_at_payout: bool = True,
    profile: str = "personal",
    config_overrides: dict[str, Any] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Run Monte Carlo from daily PnL values.

    This is the preferred path for MT5 high-frequency uploads because the
    uploaded workbook already contains close timestamps for every trade, so
    the canonical CSV can be aggregated by calendar day before sampling.
    """
    clean = [float(v) for v in daily_results if v is not None]
    if not clean:
        raise MonteCarloServiceError("This MT5 report contains no daily PnL data.")

    if len(clean) < 2:
        raise MonteCarloServiceError(
            "This MT5 report contains too few trading days for reliable simulation."
        )

    config = _build_daily_config(
        n_sims=n_sims,
        profile=profile,
        config_overrides=config_overrides,
    )

    outcomes: list[str] = []
    balances: list[float] = []
    payouts: list[float] = []
    days_list: list[int] = []
    paths: list[list[float]] = []
    target_hit_flags: list[bool] = []
    target_hit_days: list[int] = []

    progress_every = max(1, config["n_sims"] // 100)

    for i in range(config["n_sims"]):
        balance = float(config["account_size"])
        peak = balance
        trading_days = 0
        payout_amount = 0.0
        path_target_hit = False
        path_target_hit_day = 0
        equity_path = [balance]
        outcome = "timeout"

        for _day in range(int(config["max_days"])):
            raw_pnl = float(np.random.choice(clean)) * float(config["risk_multiplier"])
            pnl = raw_pnl

            balance += pnl
            trading_days += 1
            equity_path.append(balance)

            if balance > peak:
                peak = balance

            trailing_floor = peak - float(config["trailing_dd"])
            if trailing_floor < float(config["trail_stop_level"]):
                trailing_floor = float(config["trail_stop_level"])

            if balance <= trailing_floor:
                outcome = "blow"
                break

            if balance >= float(config["payout_threshold"]):
                if not path_target_hit:
                    path_target_hit = True
                    path_target_hit_day = trading_days

                total_profit = balance - float(config["account_size"])
                eligible = total_profit > 0

                if eligible:
                    withdrawable = balance - float(config["safety_net_level"])
                    if withdrawable >= 500:
                        payout_amount = min(withdrawable, float(config["max_payout_limit"]))
                        outcome = "payout"
                        if stop_at_payout:
                            break

        outcomes.append(outcome)
        balances.append(balance)
        payouts.append(payout_amount)
        days_list.append(trading_days)
        target_hit_flags.append(path_target_hit)
        if path_target_hit:
            target_hit_days.append(path_target_hit_day)
        if i < 50:
            paths.append(equity_path)

        if progress_cb is not None and ((i + 1) % progress_every == 0 or i + 1 == config["n_sims"]):
            progress_cb(i + 1, config["n_sims"])

    target_achieved_probability = sum(1 for hit in target_hit_flags if hit) / config["n_sims"]
    pass_probability = target_achieved_probability
    blow_probability = outcomes.count("blow") / config["n_sims"]
    timeout_probability = max(0.0, 1.0 - pass_probability - blow_probability)
    winning_payouts = [p for p in payouts if p > 0]
    expected_payout = float(np.mean(winning_payouts)) if winning_payouts else 0.0

    return {
        "profile": profile,
        "config": config,
        "num_trades": len(clean),
        "pass_probability": round(pass_probability, 4),
        "blow_probability": round(blow_probability, 4),
        "timeout_probability": round(timeout_probability, 4),
        "target_achieved_probability": round(target_achieved_probability, 4),
        "target_achieved_days": target_hit_days,
        "expected_payout": round(expected_payout, 2),
        "outcomes": outcomes,
        "payouts": payouts,
        "balances": balances,
        "days": days_list,
        "paths": paths,
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
