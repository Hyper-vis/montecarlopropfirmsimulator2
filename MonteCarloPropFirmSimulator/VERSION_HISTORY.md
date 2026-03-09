# Monte Carlo Prop Firm Simulator — Version History

Full chronological record of every file created and every change made, from initial build to current state.

---

## What We Have Built — System Overview

### The Problem This Solves

Prop firm trading — particularly on evaluation accounts like Apex's 50k — is not simply a question of whether a strategy is profitable in the long run. It is a question of *how* a strategy's daily profit and loss distribution interacts with a specific, rigid set of rules: a trailing drawdown that rises with your peak balance, a daily loss limit that cuts your day short, a minimum number of green days before you can even request a payout, and a hard cap on how much you can withdraw in a single cycle. A strategy with a 70% win rate can still blow up regularly if its losing days are clustered. A strategy with a modest win rate can perform brilliantly if its wins are consistent and its losses are shallow. No amount of backtesting a strategy's raw returns tells you what you need to know — you need to simulate *the account* as a system, not just the strategy in isolation.

That is exactly what this simulator does. It takes real historical trade data exported from a charting platform, converts it into an empirical daily PnL distribution, and then runs tens of thousands of Monte Carlo paths through a complete, rules-accurate Apex account model — trailing drawdown mechanics, daily caps and floors, green-day eligibility gates, payout thresholds, and maximum withdrawal limits, all encoded precisely. Every path shows you one possible future for your account. Across 20,000 futures, you get a probability distribution: how often does this strategy pay out? How often does it blow? How often does it time out inside the monthly window? What is the expected dollar return per cycle, net of reset fees? These are the numbers that actually matter for managing a portfolio of prop firm accounts as a business.

Over time the system grew from a single-account edge tester into a full multi-strategy portfolio optimizer — a tool capable of telling you not just whether a strategy is worth running, but exactly how many accounts to assign to each strategy out of a fixed capital pool, which combination of strategies produces the highest expected payout per month, and whether diversifying across multiple strategies adds value or just introduces uncorrelated noise. The entire codebase is built on a single shared simulation engine so every tool — the single-account runners, the batch processor, the correlation analyzer, and the portfolio optimizer — is working from the same rulebook.

---

### Component Map and Purposes

**`apex_engine_v3_1.py` — The Simulation Core**

The heart of everything. Encodes the complete Apex 50k evaluation account ruleset as module-level constants and exposes a clean library API. `simulate_path()` runs a single Monte Carlo path: each simulated trading day draws a random value from the empirical daily PnL pool, clamps it to the daily profit cap and loss limit, updates the running balance and trailing drawdown floor, and checks blow and payout eligibility conditions end-of-day. `run_simulations()` runs thousands of these paths and aggregates the results into payout probability, blow probability, expected payouts, and equity path samples. Version 3.1 adds configurable sampling modes — `"uniform"` (equal weight on all historical days), `"recency_weighted"` (exponential bias toward more recent data), and `"recent_only"` (sample exclusively from the last N days) — so you can tune the simulation to reflect how much you trust older data relative to the current regime.

**`run_until_payout_v3_1.py` — Pure Edge Test**

Answers the question: given no time pressure, what fraction of accounts eventually reach payout? This is the ceiling — the best-case measure of your strategy's raw edge against the Apex ruleset. It runs with a generous 90-day cap and stops each simulated account the moment it qualifies for a payout. Use this to screen strategies before deploying real capital. Anything below 55% payout probability in this mode is not worth running.

**`run_full_period_v3_1.py` — Monthly Window Test**

Answers the question: what actually happens in one calendar month? Runs each account for 21 trading days regardless of outcome and produces a three-way breakdown: payout, blow-only, and timeout (account still alive but unresolved at month end). Timeout is the hidden killer — capital tied up in a live account with no return for the whole month. This mode gives you the true monthly EV picture and is the one to use for planning how many accounts to run and what gross monthly cash flow to expect.

**`strategy_score.py` — Automated Tier Rating**

Reads a payout probability and mode, compares it against calibrated tier thresholds, and prints a colour-coded rating: ELITE, STRONG, ACCEPTABLE, MARGINAL, or REJECT — with separate thresholds for the unconstrained edge test and the 21-day monthly window, because the same raw number means different things in each context. Integrated into all single-account runners so every simulation run ends with an instant verdict.

**`batch_runner.py` — Multi-Strategy Screener**

Auto-discovers every CSV in the folder, runs both simulation modes on each, computes net EV per cycle (payout probability × mean payout minus blow probability × reset fee), and prints a ranked comparison table with colour-coded tier labels and saves a machine-readable CSV. This is the tool you run when you have several strategy variants or position sizing configurations and want to know which ones are worth scaling before committing accounts to them.

**`strategy_correlation_analyzer.py` — Diversification Analyzer**

Computes the pairwise daily PnL correlation between every pair of loaded strategies. High correlation means two strategies are effectively running the same bet — assigning accounts to both provides no diversification benefit. Low or negative correlation means their drawdown cycles tend to be uncorrelated, which smooths aggregate payout variance across a portfolio. Used as an input to the portfolio optimizer's scoring function to penalize highly correlated combinations.

**`multi_account_simulator.py` — Fixed-Configuration Portfolio Simulator**

Runs a specified number of simultaneous independent accounts on a single strategy through the full Monte Carlo, aggregating payout totals, blowout counts, equity paths, and cycle efficiency metrics. Produces histogram and equity-path grid charts. Includes a cycle efficiency analysis block: mean and median days to payout, estimated cycles per month, estimated monthly gross, and estimated monthly net after reset fees. Use this when you already know which strategy you are running and want to model the aggregate cash flow from, say, 10 simultaneous accounts.

**`portfolio_optimizer.py` — Optimal Allocation Finder**

The most sophisticated tool in the suite. Given a fixed number of Apex accounts and a set of strategies to choose from, it finds the optimal allocation: which strategies to run and how many accounts to assign to each. It enumerates all strategy subsets (1 strategy up to the configured maximum) and all possible account splits across each subset (exact enumeration up to 1,000 combinations, random sampling above that threshold), then runs a multi-stage tournament search to rank them. The tournament funnel — four stages with progressively higher simulation counts and progressively smaller survivor pools — reduces total path-simulation count by roughly 9× compared to evaluating every candidate at full depth. The final ranking is by expected total payout per 21-day cycle, with a composite score (payout rate, dollar efficiency, and inter-strategy correlation penalty) used as a tiebreaker. The output shows the top 5 portfolios with full payout-count probability distributions and bar charts.

**`regime_validator.py` — Regime Detection Utility**

Detects whether the strategy is currently operating in a regime consistent with its historical distribution or whether recent performance is diverging in a way that should prompt a pause. Line-buffered stdout so it works correctly when piped through PowerShell.

---

### The Workflow

1. **Export** your trade history from TradingView or NinjaTrader as a CSV.
2. **Screen** each strategy variant with `batch_runner.py` — one command, all CSVs ranked by monthly net EV in under a minute.
3. **Deep-dive** any ELITE or STRONG strategy with `run_until_payout_v3_1.py` and `run_full_period_v3_1.py` — understand the pure edge ceiling and the realistic monthly window outcome in detail.
4. **Check correlation** between the strategies you shortlisted using `strategy_correlation_analyzer.py` — identify which pairs are genuinely uncorrelated.
5. **Optimize the portfolio** with `portfolio_optimizer.py` — tell it how many accounts you have, which CSVs to consider, and how many strategies to combine at most. Let the tournament search find the allocation that maximizes expected monthly payout.
6. **Model the cash flow** of the winning allocation with `multi_account_simulator.py` — get the full distribution of monthly payout outcomes, cycle efficiency metrics, and equity path charts.
7. **Monitor ongoing performance** with `regime_validator.py` — flag when live results are diverging from the historical simulation distribution.

---

### Why This Is Uniquely Valuable for Prop Firm Traders

Most traders evaluate strategies in isolation using generic backtesting tools that have no concept of prop firm rules. They see a profitable equity curve and assume the strategy will work on the evaluation account — then are surprised when they blow repeatedly despite being "profitable." The Apex ruleset is not a minor constraint; the trailing drawdown, daily loss floor, and green-day gate together constitute a fundamentally different risk environment from a normal brokerage account. A strategy that works brilliantly in simulation on a regular account may have a 40% blow rate on Apex because its losing days cluster in ways that collapse the trailing floor before the edge can compound.

This system closes that gap entirely. Every simulation is a complete, rules-accurate recreation of how an Apex account actually behaves. Every metric — payout probability, blow rate, timeout rate, expected payout, net EV — is measured against the actual rules you are subject to, not a generic profit/loss curve. The multi-account and portfolio optimizer layers go further: they treat prop firm trading as what it actually is at scale — a *portfolio management* problem, where the question is not just "is this strategy good?" but "how do I allocate a fixed pool of capital across strategies to maximize risk-adjusted monthly cash flow?" The tournament optimizer answers that question rigorously and efficiently, reducing what would otherwise be tens of millions of path simulations to under two million without sacrificing ranking accuracy. The result is a complete decision-support system for operating a prop firm account portfolio as a serious, data-driven enterprise.

---

## v1.0.0 — Initial Engine + Basic Simulation Scripts

### `apex_engine.py` — Created

The core simulation engine. All other scripts import from this file. Encodes the complete Apex 50k evaluation account ruleset as module-level constants and exposes two public functions.

**Apex 50k Rules encoded:**

| Constant | Value | Meaning |
|---|---|---|
| `ACCOUNT_SIZE` | $50,000 | Starting balance |
| `TRAILING_DD` | $2,500 | Max trailing drawdown from peak |
| `TRAIL_STOP_LEVEL` | $50,100 | Balance at which trailing floor permanently locks |
| `PAYOUT_THRESHOLD` | $52,600 | Minimum balance to qualify for withdrawal |
| `DAILY_LOSS_LIMIT` | -$700 | Hard floor on single-day loss |
| `DAILY_PROFIT_CAP` | $1,050 | Hard ceiling on single-day profit |
| `MIN_DAYS` | 8 | Minimum trading days before payout eligible |
| `MIN_GREEN_DAYS` | 5 | Minimum green days (days with profit ≥ $50) required |
| `GREEN_DAY_MIN` | $50 | Minimum PnL for a day to count as "green" |

**`load_daily_returns(csv_path)`**
- Reads a NinjaTrader-format CSV export
- Strips whitespace from column headers
- Filters to rows where `Type` starts with `"Exit"` (exit trades only, ignoring entries)
- Parses `"Date and time"` column, extracts date, groups trades by date to compute daily net PnL
- Converts daily PnL to a **percentage return** by dividing by `ACCOUNT_SIZE`
- Returns a NumPy array of daily return fractions (e.g. 0.021 = 2.1% day)

**`simulate_path(daily_returns, risk_multiplier, max_days, stop_at_payout)`**
- Runs a single Monte Carlo path by randomly sampling (with replacement) from the historical daily returns array
- Applies `risk_multiplier` scaling to each sampled return
- Enforces `DAILY_PROFIT_CAP` and `DAILY_LOSS_LIMIT` on each day's PnL
- Tracks `peak` balance to compute the trailing drawdown floor:
  - If `peak - TRAILING_DD < TRAIL_STOP_LEVEL`: floor = `peak - TRAILING_DD` (floor rises with peak)
  - Otherwise: floor permanently locks at `TRAIL_STOP_LEVEL` ($50,100)
- Counts `green_days` (days where PnL ≥ `GREEN_DAY_MIN`) and `max_day_profit`
- **Blow condition:** `balance <= trailing_floor` → returns `outcome: "blow"`
- **Payout eligibility check** (all conditions must pass):
  1. `balance >= PAYOUT_THRESHOLD` ($52,600)
  2. `trading_days >= MIN_DAYS` (8 days)
  3. `green_days >= MIN_GREEN_DAYS` (5 green days)
  4. `max_day_profit <= 0.3 * total_profit` (no single day dominates profit — consistency rule)
  5. `withdrawable >= 500` where `withdrawable = balance - (PAYOUT_THRESHOLD - 500)`
- Payout amount = `min(withdrawable, $2,000)`
- If `stop_at_payout=True`: returns immediately on first eligible payout
- If `stop_at_payout=False`: records payout but continues simulating
- Returns dict: `{outcome, balance, days, payout, equity_path}`

---

### `run_until_payout.py` — Created

Simulates a **single account running until it either pays out or blows up**, with no fixed time pressure beyond the 90-day cap.

**Purpose:** Measure the pure, unconstrained edge of a strategy. Answers: "Given infinite time (up to 90 days), what fraction of accounts eventually succeed?"

**Parameters:**
- `n_sims = 20,000` — number of Monte Carlo paths
- `max_days = 90` — hard cap (reflects that Apex has no real time limit; 90 days is a generous ceiling)
- `stop_at_payout = True` — account stops the moment payout is triggered

**Output:**
- Payout probability with raw count (e.g. `73.58%  (14,715 / 20,000)`)
- Blow probability
- Mean payout (conditional on having won)
- Chart: 50 sample equity paths saved to `monte_carlo_until_payout.png`

**Reference lines on chart:**
- `$52,600` — payout threshold
- `$50,100` — trail stop lock level

---

### `run_full_period.py` — Created

Simulates a **single account running for a fixed number of trading days**, regardless of whether payout or blow occurs first.

**Purpose:** Model real operating conditions — what actually happens in one calendar month of running an account.

**Parameters:**
- `n_sims = 20,000`
- `max_days = 30` *(later changed — see v1.3)*
- `stop_at_payout = False` — account continues even after qualifying for payout

**Initial output:**
- Payout probability
- Blow probability
- Mean ending balance
- Mean payout (conditional)
- Chart saved to `monte_carlo_full_period.png`

---

## v1.1.0 — Parameter Override System

### `apex_engine.py` — Modified

**Problem:** `multi_account_simulator.py` needed to support custom account rules (non-Apex firms or different account sizes) without duplicating the simulation logic.

**Change:** `simulate_path()` signature expanded with 7 optional keyword arguments, all defaulting to `None`:

```python
def simulate_path(
    daily_returns,
    risk_multiplier=1.0,
    max_days=90,
    stop_at_payout=True,
    account_size=None,
    trailing_dd=None,
    trail_stop_level=None,
    daily_loss_limit=None,
    daily_profit_cap=None,
    payout_threshold=None,
    max_payout=None,
):
```

Each parameter resolves at runtime: if `None`, falls back to the module-level Apex constant. This is **fully backward-compatible** — all existing callers with no keyword arguments continue to work identically.

**Also fixed:** A hardcoded `$52,100` payout floor was replaced with the correct formula `_payout_threshold - 500`, so custom payout thresholds compute the withdrawal floor correctly.

---

## v1.2.0 — Multi-Account Portfolio Simulator

### `multi_account_simulator.py` — Created

Runs a **portfolio of N simultaneous Apex accounts** through Monte Carlo simulation and reports aggregate statistics and visualisations.

**Interactive prompts:**
1. CSV filename
2. Number of accounts
3. Use Apex settings? (yes/no) — if no, prompts for every parameter individually

**If using Apex settings:**

| Setting | Value |
|---|---|
| Account size | $50,000 |
| Trailing DD | $2,500 |
| Trail stop level | $50,100 |
| Daily loss limit | -$700 |
| Daily profit cap | $1,050 |
| Payout threshold | $52,600 |
| Max payout | $2,000 |
| Max days | 30 |
| Stop at payout | True |

**Simulation mechanics:**
- `n_sims = 10,000` per run
- For each simulation, all N accounts are run independently using `simulate_path()` with the override parameters
- `portfolio_payouts[sim]` = sum of all account payouts in that simulation
- `portfolio_blowouts[sim]` = count of accounts that blew in that simulation
- `account_end_balances[sim, acct_idx]` = ending balance per account per sim
- `account_payout_days[sim, acct_idx]` = days to payout (NaN if no payout that run)
- `PATH_SIMS = 150` — first 150 simulations also collect full equity paths for charting

**Aggregate statistics printed:**
- Mean, median, 5th and 95th percentile of total portfolio payout
- Probability that total payout > 0
- Mean blown accounts per simulation
- Mean ending balance per account and portfolio total

**Cycle efficiency analysis block:**
- `TRADING_DAYS_PER_MONTH = 21`
- `CYCLE_GATE = 8` (Apex minimum gate before payout eligible)
- Computes from all payout-day observations across all sims and all accounts
- Prints: mean/median/P5/P95 days to payout, % hitting within 8/10/15 days
- Estimated cycles per month = `21 / mean_days_to_payout`
- Estimated monthly gross = `n_accounts × cycles × payout_prob × mean_payout`
- Estimated monthly net subtracts reset fees ($214.50) for blown cycles

**Charts saved:**

`multi_account_histogram.png`
- Histogram of total portfolio payout distribution across all simulations
- Vertical lines for mean and median

`multi_account_paths.png`
- Grid of equity path subplots, one per account (up to 9 shown)
- Each subplot shows up to 150 sample paths, colour-coded by outcome:
  - Green = payout
  - Red = blow
  - Grey = timeout
- Reference lines: payout threshold ($52,600), account size ($50,000), max draw floor ($47,500)
- Shared legend at bottom

---

## v1.3.0 — Output Accuracy Fixes

### `apex_engine.py` — Bug Fix

**Bug:** When an account hit payout eligibility and then continued running (in `stop_at_payout=False` mode), if it subsequently blew, the blow return hardcoded `"payout": 0`. This erased the fact that a payout had already been earned earlier in the path.

**Fix:** Changed the blow return to preserve `payout_amount`:
```python
# Before (wrong):
return {"outcome": "blow", "balance": balance, "days": trading_days, "payout": 0, ...}

# After (correct):
return {"outcome": "blow", "balance": balance, "days": trading_days, "payout": payout_amount, ...}
```

**Impact:** In the `run_full_period.py` 30-day results, ~3.5% of simulations were "payout then blew" paths that were being miscounted as pure blows. This was suppressing the reported payout probability and inflating the blow probability.

---

### `run_full_period.py` — Major Output Overhaul

**Problems identified:**
1. Output only showed payout % and blow % — these two numbers did not add to 100%. The missing bucket (~13%) was **timeout** paths (accounts that ran all 30 days without blowing or paying out) — these were silently swallowed.
2. Chart title still read `"Monte Carlo Paths - Full 90 Days"` despite the 30-day window.

**New output format — three-way breakdown that always totals 100%:**
```
Payout (any point):         59.37%  (11,874 sims)
  of which payout+survived: 58.33%  (11,665 sims)
  of which payout+blew:      1.04%  (   209 sims)
Blow only (no payout):      27.29%  ( 5,458 sims)
Timeout (no payout):        13.34%  ( 2,668 sims)
─────────────────────────────────────────────────
Total:                      100.00% (20,000 sims)
```

**Additional metrics added:**
- Mean ending balance
- Mean payout (conditional on having earned one)

**Chart title fixed** to reflect the actual window.

---

## v1.4.0 — max_days Calibration

### `run_full_period.py` — max_days changed 30 → 21

**Rationale:**
- 30 trading days = ~6 calendar weeks, which overstates one management cycle
- 21 trading days = 1 calendar month — matches how accounts are actually reviewed and recycled
- Cutting to 21 days increases the timeout rate (accounts that would have resolved on days 22–30 become additional timeouts), which is the **correct and more conservative** direction for monthly planning
- A shorter window gives a more honest picture of per-month capital turnover

---

## v1.5.0 — Stdout Buffering Fix

### `regime_validator.py` — Modified

**Problem:** The script produced no output when run through PowerShell pipes (`|`). Python's default block-buffered stdout holds output in memory until the buffer fills; when piped, the pipe terminates before the buffer flushes, making the script appear to hang or produce nothing.

**Fix:** Added at the top of the file:
```python
import os, sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)
```

This switches stdout to line-buffered mode, flushing after every `print()` call regardless of whether output is piped or going to a terminal.

---

## v1.6.0 — Strategy Scoring System

### `strategy_score.py` — Created

Shared scoring module imported by both `run_until_payout.py` and `run_full_period.py`. Rates a strategy into one of five tiers based on payout probability, with separate calibrated thresholds per simulation mode.

**Why separate thresholds per mode:**
`run_until_payout` measures the best-case ceiling (no time pressure) so it naturally produces higher payout rates. `run_full_period` is constrained by the monthly window so it naturally produces lower rates. The same raw number means different things in each context.

**`run_until_payout` tier table:**

| Tier | P(payout) threshold | Meaning |
|---|---|---|
| ELITE | ≥ 75% | Top-tier edge. Extremely high payout rate with no time cap. |
| STRONG | ≥ 65% | Solid strategy. Reliable edge, worth scaling. |
| ACCEPTABLE | ≥ 55% | Positive EV but verify mean payout justifies reset risk. |
| MARGINAL | ≥ 45% | Weak edge. Only viable if mean payout is unusually large. |
| REJECT | < 45% | Negative or near-zero EV after reset costs. Do not run. |

**`run_full_period` tier table (21-day window):**

| Tier | P(payout) threshold | Meaning |
|---|---|---|
| ELITE | ≥ 60% | Majority of accounts pay out within the month. Scale aggressively. |
| STRONG | ≥ 50% | More than half resolve as payouts. Good monthly capital turn. |
| ACCEPTABLE | ≥ 40% | Positive EV but meaningful timeout drag. Monitor closely. |
| MARGINAL | ≥ 30% | Too much capital sitting idle. Consider tightening the strategy. |
| REJECT | < 30% | Poor monthly resolution. Capital tied up without return. Do not run. |

**Public API:**

`score(payout_prob, mode)` → `dict` with keys `label`, `description`, `tiers`

`print_score(payout_prob, mode)` → prints the full tier table to terminal with ANSI colour coding and a `<<< YOU ARE HERE` marker on the current tier.

**ANSI colour mapping:**
- ELITE → magenta
- STRONG → green
- ACCEPTABLE → yellow
- MARGINAL → dark yellow
- REJECT → red

---

### `run_until_payout.py` — Modified

- Added `from strategy_score import print_score`
- Added `print_score(payout_prob, mode="until_payout")` call after stats output

---

### `run_full_period.py` — Modified

- Added `from strategy_score import print_score`
- Added `print_score(payout_prob, mode="full_period")` call after stats output

---

## v1.7.0 — Batch Runner

### `batch_runner.py` — Created

Runs both simulations (`run_until_payout` mode and `run_full_period` mode) on every CSV in the folder automatically, and prints a ranked comparison table. Designed for testing multiple strategies or position-sizing variants in a single command.

**Usage:**
```
python batch_runner.py                         # auto-discovers all CSVs in folder
python batch_runner.py file1.csv file2.csv     # explicit list
```

**Configuration constants (top of file):**

| Constant | Value | Description |
|---|---|---|
| `N_SIMS` | 20,000 | Simulations per CSV per mode |
| `MAX_DAYS_FP` | 21 | Full-period window (trading days) |
| `MAX_DAYS_UTP` | 90 | Until-payout cap |
| `RESET_COST` | $137 | Apex reset fee used in EV calculation |

**`simulate_csv(csv_path, n_sims)`**

Runs both modes on a single CSV and returns a result dict with:

*Until-payout metrics:*
- `utp_payout_p` — payout probability
- `utp_mean_pay` — mean payout (conditional)
- `utp_ev_net` — net EV per cycle = `P(payout) × mean_payout − P(blow) × reset_cost`
- `utp_rating` — tier label from `strategy_score`

*Full-period metrics:*
- `fp_payout_p` — payout probability within 21 days
- `fp_blow_only_p` — blow rate (no payout earned before blow)
- `fp_timeout_p` — timeout rate (no resolution within 21 days)
- `fp_mean_pay` — mean payout (conditional)
- `fp_mean_bal` — mean ending balance
- `fp_ev_net` — net EV per cycle = `P(payout) × mean_payout − P(blow_only) × reset_cost`
- `fp_rating` — tier label from `strategy_score`

**`main()`**
1. Discovers CSV files (excludes `batch_results.csv` from glob to avoid self-processing)
2. Prints live progress line per file as it completes: `UTP 78.3% ELITE | FP 62.7% ELITE | EV $1,176`
3. Sorts all results by `fp_ev_net` descending (best monthly EV at top)
4. Prints ranked summary table with ANSI colour-coded tier labels
5. Prints detailed block per strategy
6. Saves full results to `batch_results.csv` for Excel/pandas analysis

**Output files:**
- `batch_results.csv` — machine-readable full results for all CSVs processed

---

## Current File State Summary (v1.7)

| File | Role | Key Settings |
|---|---|---|
| `apex_engine.py` | Core simulation engine | Apex 50k constants, `simulate_path()` with 7 overridable params |
| `run_until_payout.py` | Single-account pure edge test | 20,000 sims, 90-day cap, stop at payout |
| `run_full_period.py` | Single-account monthly window test | 20,000 sims, **21-day** window, runs full period |
| `multi_account_simulator.py` | Portfolio simulator for N accounts | Interactive prompts, cycle efficiency analysis, 2 chart outputs |
| `strategy_score.py` | Tier rating system | 5 tiers per mode, ANSI colour output |
| `batch_runner.py` | Multi-CSV batch processor | 20,000 sims, auto-discovers CSVs, ranked EV table |

---

## EV Decision Framework (derived from session analysis)

### Per-cycle net EV formula

$$EV_{net} = P(\text{payout}) \times \bar{P} - P(\text{blow only}) \times C_{reset}$$

Where $\bar{P}$ is mean payout amount and $C_{reset}$ is the reset fee ($137 for Apex).

### Timeout drag adjustment

When accounts timeout (no resolution within the window), they consume a full cycle with no return. The adjusted monthly EV accounting for this:

$$EV_{adj} = \frac{EV_{net}}{E[\text{attempts to resolve}]} = EV_{net} \times (P(\text{payout}) + P(\text{blow}))$$

### Key benchmarks from strategy evaluations

| Strategy | UTP | FP (21d) | Net EV/cycle | Tier |
|---|---|---|---|---|
| YM 2026-02-24 | 88.6% | 75.0% | $1,410 | ELITE / ELITE |
| YM 2026-02-24 (1) | 78.3% | 62.7% | $1,176 | ELITE / ELITE |
| ES 2026-03-01 (1) — 7 MES | 77.4% | 57.1% | $1,051 | ELITE / STRONG |
| ES 2026-02-24 (1) | 59.1% | 43.3% | $763 | ACCEPTABLE / ACCEPTABLE |
| ES 2026-03-01 — 5 MES | 80.9% | 30.0% | $353 | ELITE / MARGINAL |

**Notable finding — 5 MES vs 7 MES:** The 5 MES strategy has a higher pure edge (80.9%) than 7 MES (77.4%) but has a 64.6% timeout rate in 21 days, meaning two thirds of accounts are still unresolved at month end. This locks up capital and collapses the real monthly EV from $553 (gross) to $353 (net). The 7 MES strategy resolves faster with higher payouts, making it dominant on monthly EV despite the lower raw payout probability.

**Sizing comparison (7 MES vs 5 MES):**

| Metric | 5 MES | 7 MES | Delta |
|---|---|---|---|
| P(payout) | 73.6% | 69.3% | -4.3% |
| Mean payout | $897 | $1,747 | +$850 |
| Net EV / cycle | $624 | $1,169 | +$545 |

7 MES wins by ~$544/cycle net despite a lower payout rate, because the payout amount nearly doubles. The break-even reset fee where 5 MES becomes competitive is $12,483 — far above Apex's actual $137.


---

## v2.0.0 — Engine Rewrite: Trade-Level USD PnL Sampling

### `apex_engine_v2_0.py` — Created

**Problem with v1:** The simulation sampled per-trade percentage returns and multiplied them by the running balance. This introduced artificial compounding: a large winning trade inflated the balance, which then caused the next trade to have a bigger absolute dollar impact than any historical trade ever had. This distorted both the upside (payouts arrived too easily) and the downside (blowouts were amplified).

**v2 change:** Samples draws from an empirical **trade-level USD PnL distribution**. Each simulated trade is drawn directly from the historical dollar outcomes, removing balance-multiplication compounding entirely.

**Simulation mechanics (trade-level):**
- Each day iterates through individual trades sampled with replacement from `trade_pnls`
- Intra-day stops applied per trade: `max_losing_per_day`, `max_winning_per_day`, `daily_loss_limit`, `daily_profit_cap`, `max_trades_per_day`
- Trailing drawdown and payout eligibility checked at end-of-day

**New interactive configuration system (`prompt_config()`):**

| Parameter | Default |
|---|---|
| Account size | $50,000 |
| Trailing drawdown | $2,500 |
| Trailing stop floor | $50,100 |
| Payout threshold | $52,600 |
| Safety net after payout | $52,100 |
| Max single payout | $2,000 |
| Max trading days | 90 |
| Max trades per day | 10 |
| Max losing trades/day | 3 |
| Max winning trades/day | 10 |
| Daily loss limit | $700 |
| Daily profit cap | $1,050 |
| Min days for payout | 8 |
| Min green days | 5 |
| Green day minimum PnL | $50 |
| Risk multiplier | 1.0 |
| Number of simulations | 20,000 |

`run_monte_carlo(trade_pnls, config, stop_at_payout)` — runs all sims, returns raw outcome/balance/payout/days lists plus 50 sample equity paths.

`print_summary(results, config)` — prints payout/blow/timeout rates, mean balance, mean conditional payout, median days to payout.

### `run_until_payout_v2_0.py` — Created

Wraps `apex_engine_v2_0` for the pure-edge test: calls `prompt_config()`, runs `run_monte_carlo` with `stop_at_payout=True`, prints summary.

### `run_full_period_v2_0.py` — Created

Wraps `apex_engine_v2_0` for the fixed-window monthly test: calls `prompt_config()`, overrides `stop_at_payout=False`, runs `run_monte_carlo` for the configured `max_days` window, prints summary.

---

## v3.0.0 — Engine v3: Daily-Level Sampling, Library API

### `apex_engine_v3.py` — Created

**Problem with v2:** Trade-level simulation required sampling multiple trades per simulated day and was slow. The interactive prompt-based config made batch processing awkward. The engine was not cleanly importable as a library.

**v3 change:** Samples draws from an empirical **daily PnL distribution** — one sampled value per simulated trading day. This matches how Apex rules actually operate (daily profit cap, daily loss limit, and trailing drawdown are all evaluated end-of-day) while being substantially faster. All Apex rule mechanics are fully preserved.

**Key architectural decisions:**

1. **Daily resolution, not trade resolution.** The profit cap and loss limit are applied as clamps on the drawn daily value, not as per-trade accumulators.
2. **Dollar amounts, not percentage returns.** Raw USD PnL values preserve the absolute dollar character of the historical data without balance-multiplication artefacts.
3. **Constants baked in, not prompted.** All Apex rule values are module-level constants — the engine is a pure library, imported by runner scripts.

**Module-level constants:**

```python
ACCOUNT_SIZE       = 50_000.0
TRAILING_DD        = 2_500.0
TRAIL_STOP_LEVEL   = 50_100.0
PAYOUT_THRESHOLD   = 52_600.0
DAILY_LOSS_LIMIT   = -700.0
DAILY_PROFIT_CAP   =  1_050.0
MIN_DAYS           = 8
MIN_GREEN_DAYS     = 5
GREEN_DAY_MIN      = 50.0
MAX_PAYOUT         = 2_000.0
```

**Public API:**

`load_daily_pnl(csv_path)` → `np.ndarray`
- Filters CSV to `Type == "Trade"` rows; falls back to `"Exit …"` rows (TradingView export format)
- Groups by date component of `Date and time`, sums `Net P&L USD` per day
- Returns chronologically ordered array of daily PnL values

`compute_diagnostics(daily_pnl)` → `dict`
- Prints full statistical report: mean, std, win-day rate, skewness, kurtosis
- Geometric edge ratio: target distance vs drawdown distance
- Analytical payout probability estimate and suggested prop-score label

`simulate_path(daily_pnl, risk_multiplier, max_days, stop_at_payout)` → `dict`
- Per-day loop: sample daily PnL, clamp to profit cap / loss limit, apply risk multiplier
- Track peak for trailing drawdown floor; check blow and payout eligibility end-of-day
- Returns `{outcome, balance, days, payout, equity_path}`

`run_simulations(n_sims, daily_pnl, ...)` → `dict`
- Runs n_sims paths; collects outcomes, payouts, balances, days, 100 equity paths
- Returns aggregate statistics (payout_prob, blow_prob, mean/median payout, etc.)

### `run_until_payout_v3.py` — Created

Imports `apex_engine_v3`. Prompts for CSV path, simulation count, risk multiplier. Prints results and `strategy_score` tier output.

### `run_full_period_v3.py` — Created

Imports `apex_engine_v3`. Fixed-window (21 trading days) analysis with three-way outcome breakdown (payout / blow-only / timeout). Integrates `strategy_score` tier output.

### `batch_runner.py`, `multi_account_simulator.py`, `strategy_correlation_analyzer.py` — Updated to v3

All three scripts updated to import from `apex_engine_v3`. API surface unchanged; runtime improved due to daily-level sampling.

---

## v3.1.0 — Recency-Weighted Sampling

### `apex_engine_v3_1.py` — Created

**Problem:** v3 samples all historical trading days with equal probability regardless of when they occurred. If a strategy had regime shifts (e.g. a quiet period followed by a recent volatile streak), old data and new data are treated as equally likely to recur — potentially over-representing stale conditions.

**New in v3.1:** Three configurable sampling modes via the `mode` parameter:

| Mode | Description |
|---|---|
| `"uniform"` | Equal probability for all days — identical to v3. Default. |
| `"recency_weighted"` | Exponential weighting: recent days have higher probability. Formula: `weight_i = exp(weight_strength × (i / N))` where `i` is the day's chronological index (0 = oldest). |
| `"recent_only"` | Sample uniformly from only the last `recent_window` trading days. |

**`compute_sampling_weights(daily_pnl, mode, weight_strength, recent_window)`**
- Pre-computes `(sample_pool, weights_or_None)` once before the day loop per path
- Weights normalized to sum to 1 (valid `p=` argument for `np.random.choice`)
- Called once per `simulate_path()` call; constant across all days within one simulation

**`weight_strength` tuning guide:**

| `weight_strength` | Effect |
|---|---|
| 0 | Uniform (identical to `mode="uniform"`) |
| 3 | Moderate recency bias (~20× more weight on last day vs first over 100-day history) |
| 6 | Strong recency bias (~400× more weight on last vs first) |
| 10 | Extreme bias; effectively only the last few days matter |

All simulation mechanics (trailing drawdown, profit cap, loss limit, payout eligibility) unchanged from v3.

**Additions to v3 function signatures:**
- `simulate_path(..., mode="uniform", weight_strength=3.0, recent_window=60)`
- `run_simulations(..., mode="uniform", weight_strength=3.0, recent_window=60)`

### `run_until_payout_v3_1.py` — Created

Wraps `apex_engine_v3_1`. Prompts include sampling mode, weight strength, and recent window. Integrates `strategy_score` tier output.

### `run_full_period_v3_1.py` — Created

Wraps `apex_engine_v3_1`. Fixed 21-day window. Prompts for sampling mode parameters. Three-way outcome breakdown (payout / blow-only / timeout).

### `batch_runner.py`, `multi_account_simulator.py`, `strategy_correlation_analyzer.py` — Updated to v3.1

All three scripts updated to import from `apex_engine_v3_1`. Default `mode="uniform"` ensures identical output to v3.0 unless explicitly overridden.

---

## v4.0.0 — Portfolio Optimizer

### `portfolio_optimizer.py` — Created

New standalone module that finds the optimal allocation of a fixed number of Apex accounts across multiple trading strategies.

**Core model — independent accounts per strategy:**

Each strategy runs on its own dedicated Apex account. No blending of daily PnL across strategies. A "portfolio" is an ordered tuple such as `(7, 3)` meaning Strategy A gets 7 accounts and Strategy B gets 3. Each account is simulated independently using `simulate_path()` from `apex_engine_v3_1`.

**Why independent accounts, not blended:**
- Blending daily PnL into a single account averages out volatility, hiding true Apex rule interactions and misrepresenting reality
- Each Apex account is a standalone entity with its own balance, trailing drawdown floor, and payout eligibility gate — blending is not physically meaningful

**Module-level constants:**

```python
N_SIMS_INDIVIDUAL  = 5_000   # MC sims per strategy in solo analysis
N_SIMS_PORTFOLIO   = 2_000   # MC trials per portfolio candidate
MAX_DAYS           = 21      # path ceiling — one calendar month (trading days)
STOP_AT_PAYOUT     = True    # stop each account at first qualifying payout
TOP_N_PORTFOLIOS   = 5
MAX_EXACT_ALLOCS   = 1_000
MAX_RANDOM_ALLOCS  = 1_000
DEFAULT_K_THRESHOLDS = [1, 2, 3, 5, 10]
```

**Key functions:**

`load_strategy_pnl(csv_path)` — column detection: checks for `pnl` column first, falls back to `net p&l usd` (TradingView MultiBot export). Filters to `type == "trade"` rows (or `"exit"` prefix fallback). Groups by date, sums daily PnL. Returns chronological `np.ndarray`.

`compute_mc_metrics(pnl, n_sims, max_days, stop_at_payout, *, label_days)` — delegates to `run_simulations()` from `apex_engine_v3_1`. Returns `{payout_prob, risk_of_ruin, expected_payout, label_days}`.

`compute_trade_stats(pnl)` — returns `{n_trades, win_rate, avg_win, avg_loss, profit_factor, expectancy}` from the daily PnL array.

`compute_correlation_matrix(strategy_pnl)` — pairwise Pearson correlation. Aligns series to the shortest available history (most recent N days). Zero-variance series → correlation = 0.0 (safe fallback).

**Composition enumeration — allocation search space:**

`_count_compositions(n, k)` — count of ordered k-tuples of positive integers summing to n: $\binom{n-1}{k-1}$.

`_compositions(n, k)` — generates every such tuple exactly (stars-and-bars enumeration).

`_sample_compositions(n, k, count, rng)` — draws `count` random compositions via rejection sampling when the exact space exceeds `MAX_EXACT_ALLOCS`.

`get_allocations(n_accounts, k_strategies, rng)` — exact enumeration when count ≤ `MAX_EXACT_ALLOCS`, otherwise `MAX_RANDOM_ALLOCS` random samples.

`_simulate_portfolio_trial(pools, allocation, max_days, stop_at_payout)` — for each (strategy pool, account count) pair, calls `simulate_path()` once per account independently. Returns `(n_payouts, total_payout_amount)` for one trial.

`simulate_portfolio_mc(pools, allocation, n_sims, max_days, stop_at_payout, k_thresholds)` — runs `n_sims` trials, collects payout-count distribution. Returns `{n_accounts, expected_payouts, expected_total_payout, payout_count_dist}`.

**`_score_portfolio(mc, avg_corr)` — composite score:**

$$\text{score} = 0.60 \times \frac{\text{expected\_payouts}}{n\_accounts} + 0.25 \times \frac{\text{expected\_total\_payout}}{\text{MAX\_PAYOUT} \times n\_accounts} - 0.15 \times \text{avg\_corr}$$

`print_portfolio_results(portfolios, k_thresholds, cycle_days, width)` — header shows cycle length (`RANKED PORTFOLIOS — 21-day cycle (ranked by expected total payout)`). Per portfolio: allocation string, ★ expected total payout/cycle, P(≥ k payouts) bar chart for each threshold, risk score as secondary metric.

**`main()` interactive prompts:**

| Prompt | Default |
|---|---|
| CSV folder | current directory |
| Include all CSVs or pick | all |
| Number of accounts | 10 |
| Max strategies combined | 3 |
| Cycle length (trading days) | 21 |
| Top N portfolios to display | 5 |
| Simulations per strategy (individual) | 5,000 |
| P(≥ k) thresholds | 1, 2, 3, 5, 10 |

---

## v4.1.0 — Multi-Stage Tournament Search

### `portfolio_optimizer.py` — Major Search Algorithm Refactor

**Problem:** Running the full `N_SIMS_PORTFOLIO` count on every candidate portfolio was prohibitively slow. With 5 strategies and 10 accounts the candidate space contains hundreds of allocations across all strategy subsets; evaluating each with 2,000 simulations × 10 accounts = 20,000 path-sims per candidate created tens of millions of path simulations for a flat exhaustive search.

**New approach — multi-stage tournament funnel:**

Candidates are evaluated in successive stages. Weaker candidates are eliminated at each stage before spending further computation on them. Each stage uses more simulations than the previous, concentrating resources on the most promising allocations.

**`DEFAULT_STAGES` constant:**

```python
DEFAULT_STAGES = [
    (100,   0.20),   # Stage 1: quick filter  — keep top 20%
    (500,   0.25),   # Stage 2: mid filter    — keep top 25% of survivors
    (2_000, 20),     # Stage 3: deep filter   — keep top 20 absolute
    (5_000, 5),      # Stage 4: stress test   — keep final top 5
]
```

Each tuple is `(n_sims_this_stage, survivors)` where:
- `float` survivors → fraction of field to keep (e.g. `0.20` = top 20%)
- `int` survivors → absolute count to keep (e.g. `20` = best 20 regardless of field size)

**`_estimate_stage_budget(stages, n_candidates, n_accounts)`** — pre-computes and prints a table of projected field sizes, simulations run, survivors, and total path-sims per stage before the tournament starts:

```
Stage  Sims  Field  Survivors  Path-sims
─────────────────────────────────────────
  1     100   400      80          400,000
  2     500    80      20          400,000
  3   2,000    20      20          400,000
  4   5,000    20       5        1,000,000
─────────────────────────────────────────
Total path-sims:                2,200,000
```

Equivalent flat search (400 candidates × 5,000 sims × 10 accounts) = **20,000,000** path-sims — roughly a 9× reduction.

**Tournament mechanics in `find_optimal_portfolios(...)`:**
1. Build complete candidate list: all strategy subsets × all allocations
2. Print budget table via `_estimate_stage_budget`
3. For each stage: evaluate all survivors at this stage's sim count, print elapsed time and ETA, eliminate bottom performers, forward top survivors
4. Final sort: primary key `expected_total_payout` descending, secondary `score` descending
5. Return top `top_n` portfolios

**Per-stage progress output:**
```
  Stage 1/4 — 400 candidates × 100 sims ... elapsed 4s  ETA 16s
  Stage 2/4 — 80 candidates × 500 sims  ... elapsed 8s  ETA 4s
  Stage 3/4 — 20 candidates × 2000 sims ... elapsed 14s  ETA 0s
  Stage 4/4 — 20 candidates × 5000 sims ... elapsed 22s  ETA 0s
```

**`n_sims_portfolio` prompt removed from `main()`:** Previously prompted explicitly; now driven entirely by `DEFAULT_STAGES`. Stages can be tuned at the module level; no user prompt needed for ordinary runs.

**`MAX_EXACT_ALLOCS` and `MAX_RANDOM_ALLOCS` raised 200 → 1,000:**

Ensures exact enumeration covers all practically relevant subset sizes. With k=5 strategies and n=10 accounts, $\binom{9}{4} = 126$ compositions — well within the 1,000 threshold, so exact rather than random allocation sampling is always used for up to 5 strategies on 10 accounts.

---

## Current File State Summary (v4.1)

| File | Role | Key Settings |
|---|---|---|
| `apex_engine.py` | v1 engine (trade-level, % returns) | Legacy; not imported by v3+ scripts |
| `apex_engine_v2_0.py` | v2 engine (trade-level, USD PnL) | Interactive config prompts |
| `apex_engine_v3.py` | v3 engine (daily-level, USD PnL) | Baked-in Apex constants, clean library API |
| `apex_engine_v3_1.py` | v3.1 engine | Recency-weighted / recent-only sampling modes |
| `run_until_payout.py` | v1 runner | 20,000 sims, 90-day cap |
| `run_until_payout_v2_0.py` | v2 runner | Interactive config |
| `run_until_payout_v3.py` | v3 runner | Imports apex_engine_v3 |
| `run_until_payout_v3_1.py` | v3.1 runner | Sampling mode prompts |
| `run_full_period.py` | v1 fixed-window runner | 21-day window |
| `run_full_period_v2_0.py` | v2 fixed-window runner | Interactive config |
| `run_full_period_v3.py` | v3 fixed-window runner | Imports apex_engine_v3 |
| `run_full_period_v3_1.py` | v3.1 fixed-window runner | Sampling mode prompts |
| `multi_account_simulator.py` | N simultaneous accounts | Imports apex_engine_v3_1, cycle efficiency analysis |
| `strategy_score.py` | Tier rating system | 5 tiers per mode, ANSI colour output |
| `strategy_correlation_analyzer.py` | Pairwise correlation tool | Imports apex_engine_v3_1 |
| `batch_runner.py` | Multi-CSV batch processor | Imports apex_engine_v3_1, ranked EV table |
| `portfolio_optimizer.py` | **Multi-strategy portfolio optimizer** | Independent accounts, tournament search, 21-day cycle |
| `synthetic_distribution.py` | **Synthetic PnL generator** | Lognormal wins, normal losses, drop-in for load_daily_pnl() |
| `regime_validator.py` | Regime detection utility | Line-buffered stdout for piped output |
| `quick_stats.py` | Quick summary stats | Standalone utility |

---

## v4.2.0 — Synthetic PnL Distribution Generator

### `synthetic_distribution.py` — Created

New standalone module that generates a realistic synthetic daily PnL distribution when only summary statistics are available — win rate, average win, average loss, and trades per day — instead of a raw historical trade CSV. The output array is a direct drop-in for the array returned by `apex_engine_v3_1.load_daily_pnl()` and can be passed to `simulate_path()`, `run_simulations()`, or `portfolio_optimizer.simulate_portfolio_mc()` without any adaptation.

**Motivation:** Backtesting software may not export trade-level CSVs. Strategy parameters may come from a broker statement, a vendor's published statistics, or a walk-forward summary. This module lets you run the full Apex Monte Carlo pipeline from nothing more than four summary numbers.

---

**`estimate_stats_from_rr(win_rate, reward_risk, risk_per_trade)`**

Converts a reward-to-risk ratio and a fixed dollar risk per trade into the `avg_win` and `avg_loss` parameters required by the generator:

```python
avg_win  = reward_risk * risk_per_trade
avg_loss = risk_per_trade
```

Returns `(avg_win, avg_loss)` as a tuple. Validates all inputs are in legal ranges.

---

**`generate_synthetic_daily_pnl(win_rate, avg_win, avg_loss, trades_per_day, n_days, win_sigma_factor, loss_sigma_factor, seed)`**

Generates `n_days` synthetic trading days. For each day, `trades_per_day` trades are simulated. Each trade is independently drawn from one of two distributions:

**Win distribution — lognormal**

Lognormal is chosen because winning trades in discretionary and systematic trading are naturally right-skewed: most wins cluster near the average, but occasionally a trend extends and delivers a much larger payoff. The parameters are derived from `avg_win` and `win_sigma_factor` using the method-of-moments formula that guarantees the expected value equals `avg_win` exactly:

```
win_sigma = avg_win * win_sigma_factor       # desired std dev of wins
sigma  = sqrt(log(1 + (win_sigma / avg_win)^2))
mu     = log(avg_win) - sigma^2 / 2
```

This is implemented in the internal helper `_lognormal_params(mean, sigma_factor)`.

**Loss distribution — normal, clipped**

Losses are drawn from a normal distribution centred at `-avg_loss` with standard deviation `avg_loss * loss_sigma_factor`. Losses are clipped at `-3 * avg_loss` to prevent unrealistic tail fills while still allowing moderate over-fill variation:

```
loss_clip_floor = -3 * avg_loss
raw_loss = clip(normal(-avg_loss, loss_sigma), loss_clip_floor, 0)
```

**Vectorised implementation:**

All `n_days * trades_per_day` trades are drawn in a single batch using `numpy.random.default_rng` (seeded if `seed` is provided), then reshaped to `(n_days, trades_per_day)` and summed along axis 1 to produce the daily PnL array. This is O(n_days × trades_per_day) with no Python-level loops.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `win_rate` | float | — | Fraction of trades that are winners (0 < x < 1) |
| `avg_win` | float | — | Expected value of a winning trade in USD |
| `avg_loss` | float | — | Expected absolute value of a losing trade in USD |
| `trades_per_day` | int | — | Trades simulated per day |
| `n_days` | int | 10,000 | Number of synthetic days to generate |
| `win_sigma_factor` | float | 0.6 | Win spread as fraction of avg_win (higher = fatter right tail) |
| `loss_sigma_factor` | float | 0.3 | Loss spread as fraction of avg_loss (higher = more variable stops) |
| `seed` | int \| None | None | RNG seed for reproducibility |

**Returns:** `np.ndarray` of shape `(n_days,)` — daily aggregate PnL in USD.

---

**`print_synthetic_diagnostics(daily_pnl, label)`**

Prints a concise diagnostic summary matching the output format of `apex_engine_v3_1.load_daily_pnl()` — trading days, mean, std dev, best/worst day, win-day rate — so synthetic and real data sources look identical in terminal output.

---

**`__main__` self-test block:**

When run directly (`python synthetic_distribution.py`), executes a reference scenario:

| Setting | Value |
|---|---|
| Win rate | 45% |
| Reward / risk | 2.0× |
| Risk per trade | $100 |
| Trades per day | 3 |
| Days generated | 10,000 |

Prints theoretical EV per trade and per day, then actual simulated statistics. Saves a styled histogram to `synthetic_distribution_test.png` and displays it.

**Theoretical EV check** (45% wr, 2R, $100 risk, 3 trades/day):

$$EV_{trade} = 0.45 \times 200 - 0.55 \times 100 = 90 - 55 = +\$35 \text{ per trade}$$
$$EV_{day} = 35 \times 3 = +\$105 \text{ per day}$$

The simulated mean should converge to approximately $105/day with sufficient days.

---

**Integration with the existing pipeline:**

```python
from synthetic_distribution import estimate_stats_from_rr, generate_synthetic_daily_pnl
from apex_engine_v3_1 import run_simulations

avg_win, avg_loss = estimate_stats_from_rr(
    win_rate=0.45, reward_risk=2.0, risk_per_trade=100
)

daily_pnl = generate_synthetic_daily_pnl(
    win_rate=0.45,
    avg_win=avg_win,
    avg_loss=avg_loss,
    trades_per_day=3,
    n_days=10_000,
)

results = run_simulations(
    n_sims=20_000,
    daily_pnl=daily_pnl,
    max_days=21,
    stop_at_payout=True,
)
```


---

## Phase 2 — Programmatic API Layer

### Overview

Phase 2 introduced a clean, importable Python API (`engine_interface.py`) that wraps all simulation capabilities without requiring interactive input. The goal was to make the entire Monte Carlo pipeline callable programmatically — by a FastAPI server, a notebook, a test suite, or any external tool — with deterministic results, standardised return shapes, and guaranteed JSON serializability.

All underlying simulation logic, probability calculations, and Apex rule mechanics remain completely unchanged. Phase 2 only adds the interface layer on top.

---

### `run_until_payout_v3_1.py` — Refactored (pure function extracted)

A new pure function `run_until_payout_analysis(csv_path, n_sims, ...)` was extracted. All `input()` prompts remain in `main()`. The file still runs identically as a CLI script.

---

### `run_full_period_v3_1.py` — Refactored (pure function extracted)

Same pattern. `run_full_period_analysis(csv_path, n_sims, ...)` extracted as a pure function. CLI behavior unchanged.

---

### `multi_account_simulator.py` — Fully Refactored

Complete rewrite. The entire module-level script was extracted into `run_multi_account_analysis(csv_path, n_accounts, ...)`. All `input()` prompts moved into `main()`. Returns `{metadata, portfolio, cycle_efficiency, _portfolio_payouts, _account_paths}`.

---

### `strategy_correlation_analyzer.py` — Refactored (module-level I/O removed)

The module-level script section (header print, `load_strategies()` with `input()`, all analysis calls) was wrapped in `def main():` with `if __name__ == "__main__": main()` guard. All pure helper functions remain unchanged and fully importable without triggering any I/O.

---

### `engine_interface.py` — Created (v1.2)

Five public functions, all returning plain Python dicts that are fully JSON-serializable.

Internal helpers: `_json_safe(obj)` (recursive numpy/pandas type coercion) and `_corr_dict_to_matrix(corr_dict)` ({labels, matrix} format).

All five functions have `seed: Optional[int] = None`. When seed is set, `np.random.seed(seed)` is called at the start of the function for full reproducibility.

Public functions:
- `analyze_until_payout(...)` → `{metadata, probabilities, metrics, distributions}`
- `analyze_full_period(...)` → `{metadata, probabilities, metrics, distributions}`
- `run_batch(...)` → `list[dict]` sorted by `fp_ev_net` desc, error-safe
- `analyze_correlation(...)` → `{labels, return_correlation{labels,matrix}, blow_correlation{labels,matrix}, rolling_pairs, simultaneous_dd, worst_overlap, equity_curves, drawdown_curves}`
- `run_multi_account(...)` → `{metadata, portfolio, cycle_efficiency}` (private keys stripped)

---

### `engine_interface_test.py` — Created

12-test validation suite. All 12 tests pass. Confirmed: structure, JSON safety, seed determinism, error handling, and combined `json.dumps()` on all five function outputs (32.4 KB payload).

---

### Current File State Summary (Phase 2)

| File | Role | Notes |
|---|---|---|
| `engine_interface.py` | Public programmatic API | v1.2 — 5 functions, seed support, JSON-safe |
| `engine_interface_test.py` | API validation suite | 12 tests, all passing |
| `run_until_payout_v3_1.py` | CLI runner + pure function | `run_until_payout_analysis()` exported |
| `run_full_period_v3_1.py` | CLI runner + pure function | `run_full_period_analysis()` exported |
| `multi_account_simulator.py` | CLI runner + pure function | `run_multi_account_analysis()` exported |
| `strategy_correlation_analyzer.py` | CLI correlation tool | Refactored — safe to import, `main()` guard added |
| `batch_runner.py` | Multi-CSV batch CLI | Unchanged CLI behavior |
| `apex_engine_v3_1.py` | Simulation core | Unchanged |


---

## Phase 3 — FastAPI Server Layer

### Overview

Phase 3 wraps `engine_interface.py` in a FastAPI REST service. No simulation logic was modified. The API layer is a thin adapter: parse request body → call engine function → return result dict.

**Dependencies added:** `fastapi 0.135.1`, `uvicorn[standard] 0.41.0` (installed into `.venv`).

---

### `api_server.py` — Created (v1.0)

**Start the server:**
```
python api_server.py
```

**Interactive docs (auto-generated by FastAPI):**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc:       http://127.0.0.1:8000/redoc

---

#### Endpoints

| Method | Path | Engine Function | Description |
|---|---|---|---|
| GET  | `/` | — | Health check |
| POST | `/analyze/until_payout` | `analyze_until_payout` | Single strategy, run-until-payout mode |
| POST | `/analyze/full_period` | `analyze_full_period` | Single strategy, fixed 21-day window |
| POST | `/analyze/batch` | `run_batch` | Multiple CSVs, both modes, ranked by EV |
| POST | `/portfolio/correlation` | `analyze_correlation` | Pairwise correlation + simultaneous DD |
| POST | `/portfolio/multi_account` | `run_multi_account` | N concurrent Apex accounts |

---

#### Pydantic request models

| Model | Required fields | Key optional fields |
|---|---|---|
| `UntilPayoutRequest` | `csv_path` | `n_sims=10000`, `max_days=90`, `seed=null` |
| `FullPeriodRequest` | `csv_path` | `n_sims=10000`, `max_days=21`, `reset_cost=137.0`, `seed=null` |
| `BatchRequest` | `csv_paths` | `n_sims=10000`, `reset_cost=137.0`, `seed=null` |
| `CorrelationRequest` | `strategies` (dict) | `rolling_window=30`, `mc_n_sims=5000`, `seed=null` |
| `MultiAccountRequest` | `csv_path`, `n_accounts` | `n_sims=10000`, `max_days=30`, `seed=null` |

All models support the `sampling_mode`, `weight_strength`, and `recent_window` parameters for recency-weighted sampling.

---

#### Error handling

Every endpoint wraps its engine call in `try/except`. On failure it returns:
```json
{ "error": "descriptive message" }
```
with HTTP status **400**. `FileNotFoundError` for missing CSVs and `ValueError` for invalid inputs (e.g. fewer than 2 strategies in correlation) are caught separately to produce clear messages.

---

#### Example requests

**Health check:**
```bash
curl http://127.0.0.1:8000/
```

**Until-payout (minimal):**
```bash
curl -X POST http://127.0.0.1:8000/analyze/until_payout \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "MultiBot_-_2.3_CME_MINI_ES1!_2026-03-01.csv", "n_sims": 10000}'
```

**Full-period with seed:**
```bash
curl -X POST http://127.0.0.1:8000/analyze/full_period \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "MultiBot_-_2.3_CME_MINI_ES1!_2026-03-01.csv", "n_sims": 10000, "seed": 42}'
```

**Batch across two CSVs:**
```bash
curl -X POST http://127.0.0.1:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"csv_paths": ["MultiBot_-_2.3_CME_MINI_ES1!_2026-03-01.csv", "MultiBot_-_2.3_CBOT_MINI_YM1!_2026-02-24.csv"]}'
```

**Correlation (two strategies):**
```bash
curl -X POST http://127.0.0.1:8000/portfolio/correlation \
  -H "Content-Type: application/json" \
  -d '{"strategies": {"ES": "MultiBot_-_2.3_CME_MINI_ES1!_2026-03-01.csv", "YM": "MultiBot_-_2.3_CBOT_MINI_YM1!_2026-02-24.csv"}}'
```

**Multi-account (5 accounts):**
```bash
curl -X POST http://127.0.0.1:8000/portfolio/multi_account \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "MultiBot_-_2.3_CME_MINI_ES1!_2026-03-01.csv", "n_accounts": 5}'
```

---

#### Verified behaviour (smoke-tested on launch)

- `GET /` → `{"status": "ok", "service": "Monte Carlo Prop Firm Simulator API", "version": "1.0.0"}`
- `POST /analyze/until_payout` (ES CSV, seed=42, 300 sims) → `payout_prob=0.81`, `tier_label=ELITE`
- `POST /analyze/until_payout` (missing CSV) → HTTP 400 `{"error": "CSV not found: ..."}`

---

### Current File State Summary (Phase 3)

| File | Role | Notes |
|---|---|---|
| `api_server.py` | FastAPI REST server | v1.0 — 5 endpoints + health check |
| `engine_interface.py` | Public programmatic API | v1.2 — unchanged |
| `engine_interface_test.py` | API validation suite | 12 tests, all passing — unchanged |
| All CLI runners | CLI tools | Unchanged |
| `apex_engine_v3_1.py` | Simulation core | Unchanged |
