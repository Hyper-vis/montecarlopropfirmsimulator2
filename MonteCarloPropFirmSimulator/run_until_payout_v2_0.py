import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from apex_engine import load_trade_data, prompt_config, run_monte_carlo, print_summary

# ── CSV input ────────────────────────────────────────────────────
csv_path = input("Enter CSV filename (must be in this folder): ").strip()
if not csv_path.endswith(".csv"):
    csv_path += ".csv"
trade_pnls = load_trade_data(csv_path)

# ── Simulation config ─────────────────────────────────────────────
config = prompt_config()

# ── Run (stop as soon as payout is reached) ───────────────────────
print(f"Running {config['n_sims']:,} simulations (stop_at_payout=True) ...")
results = run_monte_carlo(trade_pnls, config, stop_at_payout=True)

# ── Print summary ────────────────────────────────────────────────
print_summary(results, config)

# ── Plot equity paths ────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for path in results["paths"]:
    plt.plot(path, alpha=0.3, linewidth=0.7)

plt.axhline(config["payout_threshold"],  color="green", linestyle="--", label=f'Payout ${config["payout_threshold"]:,.0f}')
plt.axhline(config["trail_stop_level"], color="red",   linestyle="--", label=f'Floor ${config["trail_stop_level"]:,.0f}')
plt.title(f"Monte Carlo – Stop at Payout ({config['max_days']} day max, {config['n_sims']:,} sims)")
plt.xlabel("Trades")
plt.ylabel("Balance ($)")
plt.legend()
plt.tight_layout()
plt.savefig("monte_carlo_until_payout.png", dpi=150)
print("Chart saved to: monte_carlo_until_payout.png")

# ── Histogram of ending balances ─────────────────────────────────
plt.figure(figsize=(10, 5))
plt.hist(results["balances"], bins=80, color="steelblue", edgecolor="none", alpha=0.8)
plt.axvline(config["account_size"],     color="orange", linestyle="--", label="Start")
plt.axvline(config["payout_threshold"], color="green",  linestyle="--", label="Payout threshold")
plt.axvline(config["trail_stop_level"], color="red",    linestyle="--", label="Trailing floor")
plt.title("Distribution of Ending Balances – Until Payout")
plt.xlabel("Ending Balance ($)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("monte_carlo_until_payout_hist.png", dpi=150)
print("Histogram saved to: monte_carlo_until_payout_hist.png")