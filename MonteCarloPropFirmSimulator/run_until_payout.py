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
    payouts = []
    paths = []

    for i in range(n_sims):
        result = simulate_path(
            daily_returns,
            risk_multiplier=risk_multiplier,
            max_days=90,
            stop_at_payout=True
        )

        outcomes.append(result["outcome"])
        payouts.append(result["payout"])

        if i < 50:
            paths.append(result["equity_path"])

    payout_count = sum(1 for p in payouts if p > 0)
    payout_prob = payout_count / n_sims
    blow_prob = outcomes.count("blow") / n_sims
    winning_payouts = [p for p in payouts if p > 0]
    mean_payout = np.mean(winning_payouts) if winning_payouts else 0.0

    print(f"Payout probability:        {payout_prob:.2%}  ({payout_count} / {n_sims})")
    print(f"Blow probability:          {blow_prob:.2%}")
    print(f"Mean payout (conditional): ${mean_payout:,.2f}")

    print_score(payout_prob, mode="until_payout")

    # Plot equity paths
    plt.figure(figsize=(10,6))
    for path in paths:
        plt.plot(path, alpha=0.3)

    plt.axhline(52600, linestyle='--')
    plt.axhline(50100, linestyle='--')
    plt.title("Monte Carlo Paths - Stop at Payout")
    plt.savefig("monte_carlo_until_payout.png", dpi=150)
    print("Chart saved to: monte_carlo_until_payout.png")

run_simulations(n_sims=20000, risk_multiplier=1.0)
