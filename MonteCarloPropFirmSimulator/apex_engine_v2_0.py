import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# DATA LOADER — trade-level USD PnL distribution
# ─────────────────────────────────────────────────────────────────

def load_trade_data(csv_path: str) -> np.ndarray:
    """
    Load a TradingView CSV and return an array of per-trade USD PnL values.
    Only rows where Type starts with 'Exit' are used.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[df["Type"].str.startswith("Exit")].copy()

    if df.empty:
        raise ValueError(
            f"No Exit rows found in '{csv_path}'. "
            "Check that the Type column contains 'Exit long' or 'Exit short'."
        )

    trade_pnls = df["Net P&L USD"].values.astype(float)
    print(f"\nLoaded {len(trade_pnls)} trades from '{csv_path}'")
    print(f"  Avg trade PnL : ${np.mean(trade_pnls):,.2f}")
    print(f"  Std dev       : ${np.std(trade_pnls):,.2f}")
    print(f"  Win rate      : {(trade_pnls > 0).mean():.1%}")
    print(f"  Best trade    : ${np.max(trade_pnls):,.2f}")
    print(f"  Worst trade   : ${np.min(trade_pnls):,.2f}")
    return trade_pnls


# ─────────────────────────────────────────────────────────────────
# USER CONFIG PROMPT
# ─────────────────────────────────────────────────────────────────

def _ask(prompt: str, default, cast):
    raw = input(f"  {prompt} [{default}]: ").strip()
    return cast(raw) if raw else cast(default)

def prompt_config() -> dict:
    """
    Interactively prompt for all simulation parameters.
    Press Enter to accept the default shown in brackets.
    """
    print("\n" + "═" * 55)
    print("  SIMULATION CONFIGURATION")
    print("═" * 55)

    print("\n── Account & Prop Firm Rules ──")
    account_size        = _ask("Account size ($)",                  50000, float)
    trailing_dd         = _ask("Trailing drawdown ($)",              2500, float)
    trail_stop_level    = _ask("Trailing stop floor ($)",           50100, float)
    payout_threshold    = _ask("Payout threshold ($)",              52600, float)
    safety_net_level    = _ask("Safety net level after payout ($)", 52100, float)
    max_payout_limit    = _ask("Max single payout ($)",              2000, float)

    print("\n── Trade / Day Constraints ──")
    max_days            = _ask("Max trading days",                     90, int)
    max_trades_per_day  = _ask("Max trades per day",                   10, int)
    max_losing_per_day  = _ask("Max losing trades per day",             3, int)
    max_winning_per_day = _ask("Max winning trades per day",           10, int)
    daily_loss_limit    = _ask("Daily loss limit ($, positive)",      700, float)
    daily_profit_cap    = _ask("Daily profit cap ($)",               1050, float)

    print("\n── Payout Eligibility Rules ──")
    min_days            = _ask("Minimum trading days for payout",       8, int)
    min_green_days      = _ask("Minimum green days for payout",         5, int)
    green_day_min       = _ask("Green day minimum PnL ($)",            50, float)

    print("\n── Simulation Settings ──")
    risk_multiplier     = _ask("Risk multiplier (1.0 = same scale)",  1.0, float)
    n_sims              = _ask("Number of simulations",             20000, int)

    print("═" * 55 + "\n")

    return {
        "account_size":        account_size,
        "trailing_dd":         trailing_dd,
        "trail_stop_level":    trail_stop_level,
        "payout_threshold":    payout_threshold,
        "safety_net_level":    safety_net_level,
        "max_payout_limit":    max_payout_limit,
        "max_days":            max_days,
        "max_trades_per_day":  max_trades_per_day,
        "max_losing_per_day":  max_losing_per_day,
        "max_winning_per_day": max_winning_per_day,
        "daily_loss_limit":    daily_loss_limit,
        "daily_profit_cap":    daily_profit_cap,
        "min_days":            min_days,
        "min_green_days":      min_green_days,
        "green_day_min":       green_day_min,
        "risk_multiplier":     risk_multiplier,
        "n_sims":              n_sims,
    }


# ─────────────────────────────────────────────────────────────────
# SIMULATE PATH — trade-level engine
# ─────────────────────────────────────────────────────────────────

def simulate_path(trade_pnls: np.ndarray, config: dict, stop_at_payout: bool = True) -> dict:
    """
    Simulate a single equity path at TRADE level.

    Each day, individual trades are sampled from the historical trade PnL
    distribution until a day-level constraint is hit or max_trades_per_day
    is reached.

    Returns a dict with keys:
        outcome       : "blow" | "payout" | "timeout"
        balance       : final account balance
        days          : trading days completed
        payout        : payout amount (0 if no payout)
        equity_path   : list of balance snapshots (one per trade)
    """
    balance       = config["account_size"]
    peak          = config["account_size"]
    trading_days  = 0
    green_days    = 0
    max_day_pnl   = 0.0
    payout_amount = 0.0

    equity_path = [balance]

    for _day in range(config["max_days"]):
        daily_pnl     = 0.0
        losing_today  = 0
        winning_today = 0
        trades_today  = 0

        # ── Intra-day trade loop ──────────────────────────────────
        while trades_today < config["max_trades_per_day"]:
            trade_pnl   = float(np.random.choice(trade_pnls)) * config["risk_multiplier"]

            balance    += trade_pnl
            daily_pnl  += trade_pnl
            trades_today += 1
            equity_path.append(balance)

            if trade_pnl < 0:
                losing_today += 1
            elif trade_pnl > 0:
                winning_today += 1

            # Intra-day stop conditions
            if losing_today  >= config["max_losing_per_day"]:
                break
            if winning_today >= config["max_winning_per_day"]:
                break
            if daily_pnl <= -config["daily_loss_limit"]:
                break
            if daily_pnl >= config["daily_profit_cap"]:
                break

        # ── End-of-day accounting ─────────────────────────────────
        trading_days += 1

        if balance > peak:
            peak = balance

        # Apex trailing floor: rises with peak but capped at trail_stop_level
        if peak - config["trailing_dd"] < config["trail_stop_level"]:
            trailing_floor = peak - config["trailing_dd"]
        else:
            trailing_floor = config["trail_stop_level"]

        if daily_pnl >= config["green_day_min"]:
            green_days += 1

        if daily_pnl > max_day_pnl:
            max_day_pnl = daily_pnl

        # Blow check
        if balance <= trailing_floor:
            return {
                "outcome":     "blow",
                "balance":     balance,
                "days":        trading_days,
                "payout":      0.0,
                "equity_path": equity_path,
            }

        # Payout eligibility check
        if balance >= config["payout_threshold"]:
            total_profit = balance - config["account_size"]

            if (trading_days >= config["min_days"] and
                green_days   >= config["min_green_days"] and
                total_profit >  0 and
                max_day_pnl  <= 0.30 * total_profit):

                withdrawable = balance - config["safety_net_level"]
                if withdrawable >= 500:
                    payout_amount = min(withdrawable, config["max_payout_limit"])

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


# ─────────────────────────────────────────────────────────────────
# RUN MONTE CARLO — aggregate n_sims paths
# ─────────────────────────────────────────────────────────────────

def run_monte_carlo(trade_pnls: np.ndarray, config: dict, stop_at_payout: bool = True) -> dict:
    """
    Run config["n_sims"] simulation paths and collect results.
    Returns a dict of per-path outcome lists plus 50 sample equity paths.
    """
    n_sims    = config["n_sims"]
    outcomes  = []
    balances  = []
    payouts   = []
    days_list = []
    paths     = []

    for i in range(n_sims):
        result = simulate_path(trade_pnls, config, stop_at_payout)
        outcomes.append(result["outcome"])
        balances.append(result["balance"])
        payouts.append(result["payout"])
        days_list.append(result["days"])
        if i < 50:
            paths.append(result["equity_path"])

        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1:,} / {n_sims:,} simulations complete")

    return {
        "outcomes": outcomes,
        "balances": balances,
        "payouts":  payouts,
        "days":     days_list,
        "paths":    paths,
    }


# ─────────────────────────────────────────────────────────────────
# PRINT SUMMARY — shared output block
# ─────────────────────────────────────────────────────────────────

def print_summary(results: dict, config: dict):
    outcomes  = results["outcomes"]
    payouts   = results["payouts"]
    balances  = results["balances"]
    days_list = results["days"]
    n_sims    = config["n_sims"]

    payout_count    = sum(1 for p in payouts if p > 0)
    blow_count      = outcomes.count("blow")
    timeout_count   = outcomes.count("timeout")
    winning_payouts = [p for p in payouts if p > 0]
    payout_days     = [d for o, d in zip(outcomes, days_list) if o == "payout"]

    payout_prob  = payout_count  / n_sims
    blow_prob    = blow_count    / n_sims
    timeout_prob = timeout_count / n_sims

    mean_balance     = np.mean(balances)
    mean_payout_cond = np.mean(winning_payouts) if winning_payouts else 0.0
    median_days      = int(np.median(payout_days)) if payout_days else None

    print("\n" + "═" * 55)
    print("  MONTE CARLO RESULTS")
    print("═" * 55)
    print(f"  Simulations        : {n_sims:,}")
    print(f"  Payout probability : {payout_prob:.2%}  ({payout_count:,} / {n_sims:,})")
    print(f"  Blow probability   : {blow_prob:.2%}  ({blow_count:,} / {n_sims:,})")
    print(f"  Timeout probability: {timeout_prob:.2%}  ({timeout_count:,} / {n_sims:,})")
    print(f"  Mean end balance   : ${mean_balance:,.2f}")
    print(f"  Mean payout (cond) : ${mean_payout_cond:,.2f}")
    if median_days is not None:
        print(f"  Median days→payout : {median_days} days")
    else:
        print(f"  Median days→payout : N/A (no payouts recorded)")
    print("═" * 55 + "\n")