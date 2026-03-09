"""
strategy_correlation_analyzer.py
---------------------------------
Loads multiple strategy CSVs and analyses:
  1. Return correlation matrix
  2. Rolling 30-day correlation between every pair
  3. Drawdown curves and overlap periods
  4. Simultaneous drawdown risk (how often >= N strategies are down together)
  5. Monte Carlo blow correlation (how often accounts blow on the same sim run)

v3.1: relies on apex_engine_v3_1 — Monte Carlo samples daily PnL in dollars.
Supports uniform, recency_weighted, and recent_only sampling modes.
Default mode is "uniform" (unchanged from v3). Override MC_BLOW_MODE below.
Saves all charts to PNG — no GUI window opened.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from apex_engine_v3_1 import ACCOUNT_SIZE, simulate_path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_pnl_series(csv_path: str, strategy_name: str) -> pd.Series:
    """
    Return a date-indexed daily net P&L Series (USD dollars) from a CSV.

    Filter priority (v3 style):
      1. Type == "Trade"   — NinjaTrader / standard export
      2. Type starts with "Exit" — TradingView export fallback
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    mask = df["Type"].str.strip() == "Trade"
    if mask.sum() == 0:
        mask = df["Type"].str.strip().str.startswith("Exit")

    df = df[mask].copy()
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    df["Date"] = df["Date and time"].dt.date
    series = df.groupby("Date")["Net P&L USD"].sum()
    series.index = pd.to_datetime(series.index)
    series.name = strategy_name
    return series


def load_strategies() -> dict[str, pd.Series]:
    """Interactively collect strategy CSVs from the user."""
    strategies = {}
    print("\nEnter strategy CSVs one at a time. Press Enter with no name when done.")
    while True:
        name = input("  Strategy name (or Enter to finish): ").strip()
        if not name:
            if len(strategies) < 2:
                print("  Please enter at least 2 strategies.")
                continue
            break
        csv_path = input(f"  CSV filename for '{name}': ").strip()
        if not csv_path.endswith(".csv"):
            csv_path += ".csv"
        try:
            series = load_daily_pnl_series(csv_path, name)
            strategies[name] = series
            print(f"  Loaded {len(series)} trading days for '{name}'.\n")
        except Exception as e:
            print(f"  Error loading '{csv_path}': {e}\n")
    return strategies


# ---------------------------------------------------------------------------
# Alignment + derived series
# ---------------------------------------------------------------------------

def build_aligned_frame(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    """Align all strategies on their common trading dates."""
    df = pd.DataFrame(strategies)
    df = df.dropna(how="all")
    # Forward-fill at most 1 day for minor date gaps; remaining NaN → 0
    df = df.fillna(0)
    return df.sort_index()


def equity_curves(pnl_df: pd.DataFrame, account_size: float = ACCOUNT_SIZE) -> pd.DataFrame:
    return account_size + pnl_df.cumsum()


def drawdown_curves(equity_df: pd.DataFrame) -> pd.DataFrame:
    rolling_max = equity_df.cummax()
    return equity_df - rolling_max          # always <= 0


def drawdown_pct(equity_df: pd.DataFrame) -> pd.DataFrame:
    rolling_max = equity_df.cummax()
    return (equity_df - rolling_max) / rolling_max * 100


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def return_correlation(pnl_df: pd.DataFrame) -> pd.DataFrame:
    returns = pnl_df / ACCOUNT_SIZE
    return returns.corr()


def rolling_correlation_pairs(pnl_df: pd.DataFrame, window: int = 30) -> dict[str, pd.Series]:
    returns = pnl_df / ACCOUNT_SIZE
    names = list(pnl_df.columns)
    result = {}
    for a, b in itertools.combinations(names, 2):
        key = f"{a}  ↔  {b}"
        result[key] = returns[a].rolling(window).corr(returns[b])
    return result


# ---------------------------------------------------------------------------
# Drawdown overlap
# ---------------------------------------------------------------------------

def simultaneous_drawdown_table(dd_df: pd.DataFrame, dd_threshold: float = -500) -> pd.DataFrame:
    """
    For each day count how many strategies are in drawdown deeper than dd_threshold ($).
    Returns a DataFrame with columns = strategy names + 'n_in_dd'.
    """
    in_dd = (dd_df < dd_threshold).astype(int)
    in_dd["n_in_dd"] = in_dd.sum(axis=1)
    return in_dd


def worst_overlap_periods(dd_df: pd.DataFrame, dd_threshold: float = -500, top_n: int = 10) -> pd.DataFrame:
    """Return top_n dates sorted by number of strategies simultaneously in drawdown."""
    sim = simultaneous_drawdown_table(dd_df, dd_threshold)
    worst = sim.sort_values("n_in_dd", ascending=False).head(top_n)
    return worst


# ---------------------------------------------------------------------------
# Monte Carlo blow correlation
# ---------------------------------------------------------------------------

def mc_blow_correlation(
    strategies: dict[str, pd.Series],
    n_sims: int = 5000,
    max_days: int = 30,
    mode: str = "uniform",
    weight_strength: float = 3.0,
    recent_window: int = 50,
) -> pd.DataFrame:
    """
    Run n_sims Monte Carlo simulations for each strategy using the v3.1 engine.

    Each strategy's daily PnL Series (USD dollars) is passed directly to
    simulate_path — no ratio conversion, matching how apex_engine_v3_1 works.
    The sampling mode (uniform / recency_weighted / recent_only) is forwarded.

    Returns a DataFrame of blow indicators (1 = blow, 0 = no blow) per sim per strategy.
    """
    names = list(strategies.keys())
    # v3.1: pass dollar daily PnL directly — do NOT divide by ACCOUNT_SIZE
    daily_pnl_map = {name: series.values.astype(float) for name, series in strategies.items()}

    blow_matrix = np.zeros((n_sims, len(names)), dtype=int)

    for sim_idx in range(n_sims):
        for col_idx, name in enumerate(names):
            result = simulate_path(
                daily_pnl_map[name],
                max_days        = max_days,
                stop_at_payout  = True,
                mode            = mode,
                weight_strength = weight_strength,
                recent_window   = recent_window,
            )
            if result["outcome"] == "blow":
                blow_matrix[sim_idx, col_idx] = 1

    return pd.DataFrame(blow_matrix, columns=names)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_correlation_matrix(corr: pd.DataFrame):
    print_section("Return Correlation Matrix")
    names = list(corr.columns)
    col_w = max(len(n) for n in names) + 2
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for row_name in names:
        row = f"{row_name:<{col_w}}" + "".join(f"{corr.loc[row_name, col]:>{col_w}.3f}" for col in names)
        print(row)

    print("\n  Interpretation guide:")
    print("  > +0.70  → Strongly correlated   (poor diversification)")
    print("  0 to 0.3 → Low / no correlation   (good)")
    print("  < 0      → Negative correlation   (ideal hedge)")


def print_drawdown_overlap(dd_df: pd.DataFrame, n_strats: int, dd_threshold: float = -500):
    print_section(f"Simultaneous Drawdown Risk  (threshold: ${dd_threshold:,.0f})")
    sim = simultaneous_drawdown_table(dd_df, dd_threshold)
    total_days = len(sim)
    print(f"\n  {'# Strats in DD':<22}  {'Days':>6}  {'% of history':>14}")
    print("  " + "-" * 46)
    for k in range(n_strats + 1):
        count = (sim["n_in_dd"] == k).sum()
        pct = count / total_days * 100
        bar = "█" * int(pct / 2)
        print(f"  {k:<22}  {count:>6}  {pct:>13.1f}%  {bar}")


def print_blow_correlation(blow_df: pd.DataFrame):
    print_section("Monte Carlo Blow Correlation")
    n_sims = len(blow_df)
    names = list(blow_df.columns)
    print(f"\n  Simulations: {n_sims:,}")
    print(f"\n  Individual blow rates:")
    for name in names:
        rate = blow_df[name].mean()
        print(f"    {name:<30} {rate:.2%}")

    print(f"\n  Simultaneous blow probability (both blow same sim):")
    for a, b in itertools.combinations(names, 2):
        both = (blow_df[a] & blow_df[b]).mean()
        if_a  = blow_df.loc[blow_df[a] == 1, b].mean() if blow_df[a].sum() > 0 else 0
        print(f"    {a}  ↔  {b}")
        print(f"      P(both blow):          {both:.2%}")
        print(f"      P(B blows | A blows):  {if_a:.2%}")

    blow_df["n_blows"] = blow_df.sum(axis=1)
    print(f"\n  Portfolio blow distribution (per sim):")
    for k in range(len(names) + 1):
        pct = (blow_df["n_blows"] == k).mean()
        print(f"    {k} simultaneous blows: {pct:.2%}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6",
          "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60"]


def plot_equity_curves(equity_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(equity_df.columns):
        ax.plot(equity_df.index, equity_df[col], label=col,
                color=COLORS[i % len(COLORS)], linewidth=1.2)
    ax.axhline(ACCOUNT_SIZE, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Start")
    ax.set_title("Equity Curves by Strategy", fontsize=13)
    ax.set_ylabel("Account Balance ($)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("corr_equity_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: corr_equity_curves.png")


def plot_drawdown_curves(dd_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(dd_df.columns):
        ax.fill_between(dd_df.index, dd_df[col], 0,
                        color=COLORS[i % len(COLORS)], alpha=0.25, label=col)
        ax.plot(dd_df.index, dd_df[col],
                color=COLORS[i % len(COLORS)], linewidth=0.9)
    ax.set_title("Drawdown Curves by Strategy", fontsize=13)
    ax.set_ylabel("Drawdown ($)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("corr_drawdown_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: corr_drawdown_curves.png")


def plot_correlation_heatmap(corr: pd.DataFrame):
    names = list(corr.columns)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), max(4, n * 1.2)))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if abs(val) < 0.7 else "white")
    ax.set_title("Return Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig("corr_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved: corr_heatmap.png")


def plot_rolling_correlations(rolling_pairs: dict[str, pd.Series], window: int = 30):
    n_pairs = len(rolling_pairs)
    if n_pairs == 0:
        return
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.5 * n_pairs), squeeze=False)
    for idx, (pair_name, series) in enumerate(rolling_pairs.items()):
        ax = axes[idx][0]
        ax.plot(series.index, series.values, color=COLORS[idx % len(COLORS)], linewidth=1.1)
        ax.axhline(0,    color="grey",  linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.7,  color="red",   linestyle=":",  linewidth=0.8, alpha=0.7, label="+0.7 threshold")
        ax.axhline(-0.7, color="green", linestyle=":",  linewidth=0.8, alpha=0.7, label="-0.7 threshold")
        ax.fill_between(series.index, series.values, 0,
                        where=series.values > 0, alpha=0.15, color="red")
        ax.fill_between(series.index, series.values, 0,
                        where=series.values <= 0, alpha=0.15, color="green")
        ax.set_title(f"Rolling {window}-Day Correlation:  {pair_name}", fontsize=11)
        ax.set_ylabel("Pearson r", fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("corr_rolling.png", dpi=150)
    plt.close(fig)
    print("  Saved: corr_rolling.png")


def plot_simultaneous_drawdown(dd_df: pd.DataFrame, dd_threshold: float = -500):
    sim = simultaneous_drawdown_table(dd_df, dd_threshold)
    n_strats = len(dd_df.columns)
    cmap = plt.get_cmap("Reds", n_strats + 1)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(sim.index, sim["n_in_dd"].values, width=1.5,
           color=[cmap(v) for v in sim["n_in_dd"].values], linewidth=0)
    ax.set_title(f"Number of Strategies Simultaneously in Drawdown > ${abs(dd_threshold):,.0f}", fontsize=11)
    ax.set_ylabel("# Strategies", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_strats))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, label="# Strategies in DD")
    plt.tight_layout()
    plt.savefig("corr_simultaneous_dd.png", dpi=150)
    plt.close(fig)
    print("  Saved: corr_simultaneous_dd.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Strategy Correlation Analyzer  (v3 engine)")
    print("  Reference: Apex 50k settings (apex_engine_v3.py)")
    print("=" * 60)

    strategies = load_strategies()

    roll_window = 30
    dd_threshold = -500
    n_mc_sims       = 5000
    # v3.1 sampling mode for Monte Carlo blow correlation
    # Set to "recency_weighted" or "recent_only" to weight toward recent history
    MC_BLOW_MODE    = "uniform"   # "uniform" | "recency_weighted" | "recent_only"
    MC_BLOW_WSTR    = 3.0         # weight_strength (recency_weighted only)
    MC_BLOW_WINDOW  = 50          # recent_window in days (recent_only only)

    print(f"\nUsing rolling window: {roll_window} days")
    print(f"Drawdown threshold:   ${dd_threshold:,.0f}")
    print(f"MC blow sims:         {n_mc_sims:,}  [sampling mode: {MC_BLOW_MODE}]")

    # --- Build aligned data ---
    pnl_df    = build_aligned_frame(strategies)
    eq_df     = equity_curves(pnl_df)
    dd_df     = drawdown_curves(pnl_df)
    corr      = return_correlation(pnl_df)
    roll_corr = rolling_correlation_pairs(pnl_df, window=roll_window)

    # --- Print results ---
    print_correlation_matrix(corr)
    print_drawdown_overlap(dd_df, n_strats=len(strategies), dd_threshold=dd_threshold)

    print(f"\nRunning Monte Carlo blow correlation ({n_mc_sims:,} sims)...")
    blow_df = mc_blow_correlation(
        strategies, n_sims=n_mc_sims,
        mode=MC_BLOW_MODE, weight_strength=MC_BLOW_WSTR, recent_window=MC_BLOW_WINDOW,
    )
    print_blow_correlation(blow_df)

    # --- Save charts ---
    print_section("Saving Charts")
    plot_equity_curves(eq_df)
    plot_drawdown_curves(dd_df)
    plot_correlation_heatmap(corr)
    plot_rolling_correlations(roll_corr, window=roll_window)
    plot_simultaneous_drawdown(dd_df, dd_threshold=dd_threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
