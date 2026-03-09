import numpy as np
from apex_engine import load_daily_returns, simulate_path

daily_returns = load_daily_returns("MultiBot_-_2.3_CBOT_MINI_YM1!_2026-02-24.csv")
n_sims = 20000
outcomes, payouts, balances = [], [], []

for i in range(n_sims):
    r = simulate_path(daily_returns, risk_multiplier=1.0, max_days=90, stop_at_payout=False)
    outcomes.append(r["outcome"])
    payouts.append(r["payout"])
    balances.append(r["balance"])

payout_count = sum(1 for p in payouts if p > 0)
winning = [p for p in payouts if p > 0]

print(f"Payout probability:        {payout_count/n_sims:.2%}  ({payout_count} / {n_sims})")
print(f"Blow probability:          {outcomes.count('blow')/n_sims:.2%}")
print(f"Mean ending balance:       ${np.mean(balances):,.2f}")
print(f"Mean payout (conditional): ${np.mean(winning) if winning else 0:,.2f}")
