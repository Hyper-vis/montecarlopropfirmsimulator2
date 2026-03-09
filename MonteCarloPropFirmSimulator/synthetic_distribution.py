"""
synthetic_distribution.py
=========================
Generate a realistic synthetic daily PnL distribution for Monte Carlo
simulation when only summary statistics are available instead of raw
historical trade data.

Use this module as a drop-in data source for apex_engine_v3_1 when you
have a strategy's win rate and average win/loss figures but no CSV export.
The output array has the same shape and semantics as the array returned by
apex_engine_v3_1.load_daily_pnl() — a 1-D NumPy array of daily aggregate
PnL values in USD — and can be passed directly to simulate_path() or
run_simulations().

Public API
----------
  estimate_stats_from_rr(win_rate, reward_risk, risk_per_trade)
      → (avg_win, avg_loss)

  generate_synthetic_daily_pnl(win_rate, avg_win, avg_loss,
                                trades_per_day, n_days, ...)
      → np.ndarray  shape (n_days,)

Distributions used
------------------
  Wins  — lognormal, parameterised so E[X] ≈ avg_win.
           Lognormal naturally produces right-skewed payoffs consistent with
           real trading: most wins cluster near the average but an occasional
           large winner is possible.

  Losses — normal, centred at -avg_loss, clipped at -3 × avg_loss.
            Losses in practice are roughly symmetric around the stop level
            with occasional over-fills but rarely more than 3× the intended
            risk per trade.

Notes
-----
  The lognormal parameters mu and sigma are derived from the desired mean
  (avg_win) and the user-specified spread factor (win_sigma_factor):

      win_sigma = avg_win * win_sigma_factor
      sigma     = sqrt(log(1 + (win_sigma / avg_win)^2))
      mu        = log(avg_win) - sigma^2 / 2

  This ensures E[lognormal(mu, sigma)] = avg_win exactly by construction.
"""

import math
import numpy as np
import numpy.random as npr


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def estimate_stats_from_rr(
    win_rate: float,
    reward_risk: float,
    risk_per_trade: float,
) -> tuple[float, float]:
    """
    Compute average win and average loss from a reward-to-risk ratio and a
    fixed dollar risk per trade.

    Parameters
    ----------
    win_rate       : float  — fraction of trades that are winners  (e.g. 0.45)
    reward_risk    : float  — ratio of average win to average loss  (e.g. 2.0)
    risk_per_trade : float  — dollar risk per losing trade           (e.g. 100)

    Returns
    -------
    (avg_win, avg_loss) : (float, float)
        avg_win  = reward_risk × risk_per_trade
        avg_loss = risk_per_trade

    Example
    -------
    >>> avg_win, avg_loss = estimate_stats_from_rr(0.45, 2.0, 100)
    >>> avg_win
    200.0
    >>> avg_loss
    100.0
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError(f"win_rate must be in (0, 1); got {win_rate}")
    if reward_risk <= 0:
        raise ValueError(f"reward_risk must be > 0; got {reward_risk}")
    if risk_per_trade <= 0:
        raise ValueError(f"risk_per_trade must be > 0; got {risk_per_trade}")

    avg_win  = reward_risk * risk_per_trade
    avg_loss = risk_per_trade
    return avg_win, avg_loss


def _lognormal_params(mean: float, sigma_factor: float) -> tuple[float, float]:
    """
    Derive lognormal (mu, sigma) such that E[X] = mean and the coefficient
    of variation equals sigma_factor.

    Parameters
    ----------
    mean         : float — desired expected value of the lognormal
    sigma_factor : float — std dev as a fraction of the mean (e.g. 0.6 → 60%)

    Returns
    -------
    (mu, sigma) : parameters for numpy.random.lognormal
    """
    cv     = sigma_factor                     # coefficient of variation
    sigma  = math.sqrt(math.log(1.0 + cv ** 2))
    mu     = math.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_daily_pnl(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    trades_per_day: int,
    n_days: int = 10_000,
    win_sigma_factor: float = 0.6,
    loss_sigma_factor: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic daily PnL array suitable for use as a drop-in
    replacement for apex_engine_v3_1.load_daily_pnl() output.

    Parameters
    ----------
    win_rate         : float — probability that a single trade is a winner
    avg_win          : float — expected value of a winning trade in USD
    avg_loss         : float — expected absolute value of a losing trade in USD
    trades_per_day   : int   — number of trades executed each simulated day
    n_days           : int   — number of synthetic trading days to generate
                               (default 10,000)
    win_sigma_factor : float — spread of winning trades as a fraction of
                               avg_win; higher = fatter right tail
                               (default 0.6)
    loss_sigma_factor: float — spread of losing trades as a fraction of
                               avg_loss; higher = more variable losses
                               (default 0.3)
    seed             : int | None — optional RNG seed for reproducibility

    Returns
    -------
    np.ndarray  shape (n_days,)
        Daily aggregate PnL in USD. Each element is the sum of
        trades_per_day simulated trades on that day.

    Raises
    ------
    ValueError  — if win_rate, avg_win, avg_loss, or trades_per_day are
                  outside valid ranges.

    Notes
    -----
    The returned array can be passed directly to:
        apex_engine_v3_1.simulate_path(daily_pnl, ...)
        apex_engine_v3_1.run_simulations(n_sims, daily_pnl, ...)
        portfolio_optimizer.simulate_portfolio_mc(pools, ...)  # as a pool entry
    """
    # ── Validation ──────────────────────────────────────────────────────────
    if not (0.0 < win_rate < 1.0):
        raise ValueError(f"win_rate must be in (0, 1); got {win_rate}")
    if avg_win <= 0:
        raise ValueError(f"avg_win must be > 0; got {avg_win}")
    if avg_loss <= 0:
        raise ValueError(f"avg_loss must be > 0; got {avg_loss}")
    if trades_per_day < 1:
        raise ValueError(f"trades_per_day must be >= 1; got {trades_per_day}")
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1; got {n_days}")
    if not (0.0 < win_sigma_factor):
        raise ValueError(f"win_sigma_factor must be > 0; got {win_sigma_factor}")
    if not (0.0 < loss_sigma_factor):
        raise ValueError(f"loss_sigma_factor must be > 0; got {loss_sigma_factor}")

    rng = npr.default_rng(seed)

    # ── Pre-compute distribution parameters ──────────────────────────────────
    win_mu, win_sigma   = _lognormal_params(avg_win, win_sigma_factor)
    loss_mean           = -avg_loss
    loss_sigma          = avg_loss * loss_sigma_factor
    loss_clip_floor     = -3.0 * avg_loss          # hard floor on loss size

    total_trades = n_days * trades_per_day

    # ── Simulate all trades in one vectorised batch ──────────────────────────
    # Determine which trades are winners
    is_win = rng.random(total_trades) < win_rate   # bool array, shape (total_trades,)

    # Sample the full win and loss pools up front (vectorised, fast)
    raw_wins   = rng.lognormal(mean=win_mu, sigma=win_sigma, size=total_trades)
    raw_losses = rng.normal(loc=loss_mean, scale=loss_sigma, size=total_trades)
    raw_losses = np.clip(raw_losses, loss_clip_floor, 0.0)   # losses ≤ 0

    # Combine: winner gets win sample, loser gets loss sample
    trade_pnl = np.where(is_win, raw_wins, raw_losses)       # shape (total_trades,)

    # ── Aggregate to daily totals ─────────────────────────────────────────────
    daily_pnl = trade_pnl.reshape(n_days, trades_per_day).sum(axis=1)

    return daily_pnl


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS HELPER  (mirrors apex_engine_v3_1 print style)
# ─────────────────────────────────────────────────────────────────────────────

def print_synthetic_diagnostics(daily_pnl: np.ndarray, label: str = "Synthetic") -> None:
    """
    Print a concise diagnostic summary for a synthetic (or real) daily PnL
    array.  Output format matches apex_engine_v3_1.load_daily_pnl() so the
    two sources look identical in terminal output.

    Parameters
    ----------
    daily_pnl : np.ndarray — 1-D array of daily PnL values in USD
    label     : str        — display name shown in the header line
    """
    n          = len(daily_pnl)
    mean_pnl   = np.mean(daily_pnl)
    std_pnl    = np.std(daily_pnl)
    best_day   = np.max(daily_pnl)
    worst_day  = np.min(daily_pnl)
    win_day_rt = (daily_pnl > 0).mean()

    print(f"\n[Data] Source        : {label}")
    print(f"       Trading days   : {n:,}")
    print(f"       Mean day PnL   : ${mean_pnl:>10,.2f}")
    print(f"       Std dev        : ${std_pnl:>10,.2f}")
    print(f"       Best day       : ${best_day:>10,.2f}")
    print(f"       Worst day      : ${worst_day:>10,.2f}")
    print(f"       Win-day rate   : {win_day_rt:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── Parameters ───────────────────────────────────────────────────────────
    WIN_RATE        = 0.45
    REWARD_RISK     = 2.0
    RISK_PER_TRADE  = 100.0
    TRADES_PER_DAY  = 3
    N_DAYS          = 10_000

    avg_win, avg_loss = estimate_stats_from_rr(WIN_RATE, REWARD_RISK, RISK_PER_TRADE)

    print("=" * 55)
    print("  SYNTHETIC DISTRIBUTION — SELF-TEST")
    print("=" * 55)
    print(f"  Win rate       : {WIN_RATE:.0%}")
    print(f"  Reward / risk  : {REWARD_RISK:.1f}×")
    print(f"  Risk per trade : ${RISK_PER_TRADE:,.0f}")
    print(f"  Avg win        : ${avg_win:,.2f}")
    print(f"  Avg loss       : ${avg_loss:,.2f}")
    print(f"  Trades per day : {TRADES_PER_DAY}")
    print(f"  Days generated : {N_DAYS:,}")

    # Theoretical per-day EV
    ev_per_trade = WIN_RATE * avg_win - (1 - WIN_RATE) * avg_loss
    ev_per_day   = ev_per_trade * TRADES_PER_DAY
    print(f"\n  Theoretical EV / trade : ${ev_per_trade:,.2f}")
    print(f"  Theoretical EV / day   : ${ev_per_day:,.2f}")

    # ── Generate ─────────────────────────────────────────────────────────────
    daily_pnl = generate_synthetic_daily_pnl(
        win_rate        = WIN_RATE,
        avg_win         = avg_win,
        avg_loss        = avg_loss,
        trades_per_day  = TRADES_PER_DAY,
        n_days          = N_DAYS,
        seed            = 42,
    )

    print_synthetic_diagnostics(daily_pnl, label="Synthetic (45% wr, 2R, $100 risk, 3 trades/day)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(daily_pnl, bins=120, color="#4C72B0", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(daily_pnl.mean(),   color="#DD4949", linewidth=1.8, linestyle="--",
               label=f"Mean  ${daily_pnl.mean():,.0f}")
    ax.axvline(np.median(daily_pnl), color="#F5A623", linewidth=1.5, linestyle=":",
               label=f"Median ${np.median(daily_pnl):,.0f}")
    ax.axvline(0, color="white", linewidth=1.0, linestyle="-", alpha=0.5)

    ax.set_title(
        f"Synthetic Daily PnL Distribution\n"
        f"Win rate {WIN_RATE:.0%}  |  {REWARD_RISK:.1f}R  |  "
        f"${RISK_PER_TRADE:.0f} risk  |  {TRADES_PER_DAY} trades/day  |  "
        f"{N_DAYS:,} days",
        fontsize=11,
    )
    ax.set_xlabel("Daily PnL (USD)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor("#1C1C1E")
    fig.patch.set_facecolor("#1C1C1E")
    ax.tick_params(colors="lightgrey")
    ax.xaxis.label.set_color("lightgrey")
    ax.yaxis.label.set_color("lightgrey")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    plt.savefig("synthetic_distribution_test.png", dpi=150)
    print("\n  Histogram saved → synthetic_distribution_test.png")
    plt.show()
