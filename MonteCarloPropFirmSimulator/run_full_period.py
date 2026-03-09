import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file instead of opening a window
import matplotlib.pyplot as plt
from apex_engine import load_daily_returns, simulate_path
from strategy_score import print_score

csv_path = input("Enter CSV filename (must be in this folder): ").strip()
if not csv_path.endswith(".csv"):
    csv_path += ".csv"
daily_returns = load_daily_returns(csv_path)

def run_simulations(n_sims=10000, risk_multiplier=1.0):

    outcomes = []
    ending_balances = []
    payouts = []
    paths = []
    payout_then_blew = 0

    for i in range(n_sims):
        result = simulate_path(
            daily_returns,
            risk_multiplier=risk_multiplier,
            max_days=21,
            stop_at_payout=False
        )

        outcomes.append(result["outcome"])
        ending_balances.append(result["balance"])
        payouts.append(result["payout"])

        if result["outcome"] == "blow" and result["payout"] > 0:
            payout_then_blew += 1

        if i < 50:
            paths.append(result["equity_path"])

    n_payout       = sum(1 for o, p in zip(outcomes, payouts) if p > 0)   # earned payout (with or without later blow)
    n_blow_only    = sum(1 for o, p in zip(outcomes, payouts) if o == "blow" and p == 0)
    n_blow_total   = outcomes.count("blow")
    n_timeout      = outcomes.count("timeout")
    n_timeout_paid = sum(1 for o, p in zip(outcomes, payouts) if o == "timeout" and p > 0)

    payout_prob    = n_payout / n_sims
    blow_prob      = n_blow_total / n_sims
    timeout_prob   = n_timeout / n_sims
    mean_end_balance = np.mean(ending_balances)
    winning_payouts = [p for p in payouts if p > 0]
    mean_payout = np.mean(winning_payouts) if winning_payouts else 0.0

    print()
    print(f"{'='*54}")
    print(f"  Outcome Breakdown ({n_sims:,} simulations)")
    print(f"{'='*54}")
    print(f"  Payout (any point):       {payout_prob:>6.2%}  ({n_payout:>6,} sims)")
    print(f"    of which payout+survived:{n_timeout_paid / n_sims:>6.2%}  ({n_timeout_paid:>6,} sims)")
    print(f"    of which payout+blew:   {payout_then_blew / n_sims:>6.2%}  ({payout_then_blew:>6,} sims)")
    print(f"  Blow only (no payout):    {n_blow_only / n_sims:>6.2%}  ({n_blow_only:>6,} sims)")
    print(f"  Timeout (no payout):      {(n_timeout - n_timeout_paid) / n_sims:>6.2%}  ({n_timeout - n_timeout_paid:>6,} sims)")
    print(f"{'─'*54}")
    print(f"  Total:                    100.00%  ({n_sims:>6,} sims)")
    print(f"{'='*54}")
    print(f"  Mean ending balance:      ${mean_end_balance:>10,.2f}")
    print(f"  Mean payout (if earned):  ${mean_payout:>10,.2f}")
    print()

    print_score(payout_prob, mode="full_period")

    # Plot equity paths
    plt.figure(figsize=(10,6))
    for path in paths:
        plt.plot(path, alpha=0.3)

    plt.axhline(52600, linestyle='--')
    plt.axhline(50100, linestyle='--')
    plt.title(f"Monte Carlo Paths - Full {21} Days")
    plt.savefig("monte_carlo_full_period.png", dpi=150)
    print("Chart saved to: monte_carlo_full_period.png")

run_simulations(n_sims=20000, risk_multiplier=1.0)
