import pandas as pd
import numpy as np

ACCOUNT_SIZE = 50000
TRAILING_DD = 2500
TRAIL_STOP_LEVEL = 50100
PAYOUT_THRESHOLD = 52600

DAILY_LOSS_LIMIT = -700
DAILY_PROFIT_CAP = 1050

MIN_DAYS = 8
MIN_GREEN_DAYS = 5
GREEN_DAY_MIN = 50

# -----------------------------
# Load and prepare daily returns
# -----------------------------
def load_daily_returns(csv_path):
    df = pd.read_csv(csv_path)

    # Clean whitespace just in case
    df.columns = df.columns.str.strip()

    # Keep only exit trades ("Exit long" / "Exit short")
    df = df[df["Type"].str.startswith("Exit")]

    # Convert date column
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    df["Date"] = df["Date and time"].dt.date

    # Use correct PnL column
    daily_pnl = df.groupby("Date")["Net P&L USD"].sum().values

    # Convert to return %
    daily_returns = daily_pnl / ACCOUNT_SIZE

    return daily_returns


# -----------------------------
# Single simulation path
# -----------------------------
def simulate_path(
    daily_returns,
    risk_multiplier=1.0,
    max_days=90,
    stop_at_payout=True,
    # Optional overrides — None means use module-level Apex defaults
    account_size=None,
    trailing_dd=None,
    trail_stop_level=None,
    daily_loss_limit=None,
    daily_profit_cap=None,
    payout_threshold=None,
    max_payout=None,
):
    # Resolve effective settings (falls back to Apex constants when not overridden)
    _account_size     = account_size     if account_size     is not None else ACCOUNT_SIZE
    _trailing_dd      = trailing_dd      if trailing_dd      is not None else TRAILING_DD
    _trail_stop_level = trail_stop_level if trail_stop_level is not None else TRAIL_STOP_LEVEL
    _daily_loss_limit = daily_loss_limit if daily_loss_limit is not None else DAILY_LOSS_LIMIT
    _daily_profit_cap = daily_profit_cap if daily_profit_cap is not None else DAILY_PROFIT_CAP
    _payout_threshold = payout_threshold if payout_threshold is not None else PAYOUT_THRESHOLD
    _max_payout       = max_payout       if max_payout       is not None else 2000

    balance = _account_size
    peak = _account_size

    trading_days = 0
    green_days = 0
    max_day_profit = 0

    equity_path = [balance]
    payout_amount = 0
    payout_day = None

    for day in range(max_days):

        r = np.random.choice(daily_returns)
        r *= risk_multiplier

        pnl = balance * r

        pnl = min(pnl, _daily_profit_cap)
        pnl = max(pnl, _daily_loss_limit)

        balance += pnl
        equity_path.append(balance)

        trading_days += 1

        if balance > peak:
            peak = balance

        # Trailing drawdown logic — floor locks once peak reaches trail_stop_level
        if peak - _trailing_dd < _trail_stop_level:
            trailing_floor = peak - _trailing_dd
        else:
            trailing_floor = _trail_stop_level

        if pnl >= GREEN_DAY_MIN:
            green_days += 1

        if pnl > max_day_profit:
            max_day_profit = pnl

        if balance <= trailing_floor:
            return {
                "outcome": "blow",
                "balance": balance,
                "days": trading_days,
                "payout": payout_amount,   # preserve any payout already earned
                "equity_path": equity_path
            }

        # Payout eligibility
        if balance >= _payout_threshold:
            total_profit = balance - _account_size

            if trading_days >= MIN_DAYS and green_days >= MIN_GREEN_DAYS:

                if max_day_profit <= 0.3 * total_profit:

                    # Withdrawable = balance above the minimum-payout floor
                    payout_floor = _payout_threshold - 500
                    withdrawable = balance - payout_floor

                    if withdrawable >= 500:
                        payout_amount = min(withdrawable, _max_payout)
                        payout_day = trading_days

                        if stop_at_payout:
                            return {
                                "outcome": "payout",
                                "balance": balance,
                                "days": payout_day,
                                "payout": payout_amount,
                                "equity_path": equity_path
                            }

    return {
        "outcome": "timeout",
        "balance": balance,
        "days": trading_days,
        "payout": payout_amount,
        "equity_path": equity_path
    }
