"""
engine_interface_test.py
========================
Validation tests for the public API surface of engine_interface.py.

Tests verify:
  1. Each public function returns a Python dict (or list[dict] for run_batch).
  2. Required top-level keys are present.
  3. json.dumps(result) succeeds without errors.
  4. Key numeric values are in plausible ranges.
  5. seed parameter produces reproducible (deterministic) results.

Run with:
    python engine_interface_test.py

A final summary table lists PASSED / FAILED for every test, and the
complete backend interface is printed at the end.
"""

import json
import sys
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — fast-mode settings (small n_sims for speed; real CSV files)
# ─────────────────────────────────────────────────────────────────────────────

# Two real CSV files present in this folder
CSV_ES = "MultiBot_-_2.3_CME_MINI_ES1!_2026-03-01.csv"
CSV_YM = "MultiBot_-_2.3_CBOT_MINI_YM1!_2026-02-24.csv"

N_SIMS_FAST   = 300    # quick run for most tests
N_CORR_SIMS   = 200    # MC blow-correlation sims (expensive)
SEED_A        = 42
SEED_B        = 99

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT
# ─────────────────────────────────────────────────────────────────────────────

try:
    from engine_interface import (
        analyze_until_payout,
        analyze_full_period,
        run_batch,
        analyze_correlation,
        run_multi_account,
    )
except Exception as exc:
    print(f"\n[FATAL] Could not import engine_interface: {exc}")
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (test_name, passed, message)


def _check_json_safe(obj: object, context: str = "result") -> None:
    """Assert that json.dumps(obj) succeeds; raise AssertionError otherwise."""
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"json.dumps({context}) failed: {exc}\n"
            f"  Offending type: {type(exc.__context__ or exc)}"
        )


def _check_keys(d: dict, *required: str, context: str = "result") -> None:
    """Assert that every key in `required` is present in dict `d`."""
    missing = [k for k in required if k not in d]
    if missing:
        raise AssertionError(
            f"Missing key(s) in {context}: {missing}\n"
            f"  Actual keys: {sorted(d.keys())}"
        )


def _check_prob(val: float, name: str) -> None:
    """Assert that value is between 0.0 and 1.0 (inclusive)."""
    if not (0.0 <= val <= 1.0):
        raise AssertionError(f"{name} = {val} is outside [0, 1]")


def run_test(name: str, fn) -> None:
    """Run a single test function and record the outcome."""
    print(f"\n  {'─'*60}")
    print(f"  TEST: {name}")
    print(f"  {'─'*60}")
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  ✓  PASSED")
    except AssertionError as exc:
        _results.append((name, False, str(exc)))
        print(f"  ✗  FAILED: {exc}")
    except Exception as exc:
        _results.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  ✗  ERROR")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — analyze_until_payout
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_until_payout_structure() -> None:
    result = analyze_until_payout(CSV_ES, n_sims=N_SIMS_FAST, n_paths=5, seed=SEED_A)

    assert isinstance(result, dict), "Return value must be a dict"
    _check_keys(result, "metadata", "probabilities", "metrics", "distributions")

    # metadata
    _check_keys(result["metadata"], "csv_path", "n_sims", "risk_multiplier",
                "max_days", "sampling_mode", context="metadata")
    assert result["metadata"]["n_sims"] == N_SIMS_FAST

    # probabilities
    probs = result["probabilities"]
    _check_keys(probs, "payout_prob", "blow_prob", "timeout_prob", context="probabilities")
    for key in ("payout_prob", "blow_prob", "timeout_prob"):
        _check_prob(probs[key], f"probabilities.{key}")
    prob_sum = probs["payout_prob"] + probs["blow_prob"] + probs["timeout_prob"]
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, expected ~1.0"

    # metrics
    _check_keys(result["metrics"], "mean_days_all", "mean_days_to_payout",
                "median_days_to_payout", "mean_payout", "tier_label",
                "tier_description", context="metrics")
    assert isinstance(result["metrics"]["tier_label"], str)
    assert result["metrics"]["tier_label"] in ("ELITE", "STRONG", "ACCEPTABLE",
                                               "MARGINAL", "REJECT")

    # distributions
    _check_keys(result["distributions"], "equity_paths", context="distributions")
    assert isinstance(result["distributions"]["equity_paths"], list)

    # JSON safety
    _check_json_safe(result, "analyze_until_payout result")
    print(f"    payout_prob={probs['payout_prob']:.2%}  "
          f"blow_prob={probs['blow_prob']:.2%}  "
          f"tier={result['metrics']['tier_label']}")


def test_analyze_until_payout_seed_determinism() -> None:
    r1 = analyze_until_payout(CSV_ES, n_sims=N_SIMS_FAST, n_paths=0, seed=SEED_A)
    r2 = analyze_until_payout(CSV_ES, n_sims=N_SIMS_FAST, n_paths=0, seed=SEED_A)
    assert r1["probabilities"]["payout_prob"] == r2["probabilities"]["payout_prob"], \
        "Same seed must produce identical payout_prob"
    assert r1["metrics"]["mean_days_all"] == r2["metrics"]["mean_days_all"], \
        "Same seed must produce identical mean_days_all"
    print(f"    Seed {SEED_A}: payout_prob={r1['probabilities']['payout_prob']:.4f} (both runs)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — analyze_full_period
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_full_period_structure() -> None:
    result = analyze_full_period(CSV_ES, n_sims=N_SIMS_FAST, n_paths=5,
                                 reset_cost=137.0, seed=SEED_A)

    assert isinstance(result, dict)
    _check_keys(result, "metadata", "probabilities", "metrics", "distributions")

    # metadata
    _check_keys(result["metadata"], "csv_path", "n_sims", "risk_multiplier",
                "max_days", "reset_cost", "sampling_mode", context="metadata")
    assert result["metadata"]["reset_cost"] == 137.0

    # probabilities — five distinct outcome splits must sum to ~1
    probs = result["probabilities"]
    _check_keys(probs, "payout_prob", "payout_survived_prob",
                "payout_then_blew_prob", "blow_no_payout_prob",
                "timeout_no_pay_prob", context="probabilities")
    for k, v in probs.items():
        _check_prob(v, f"probabilities.{k}")
    prob_sum = (probs["payout_survived_prob"] + probs["payout_then_blew_prob"]
                + probs["blow_no_payout_prob"] + probs["timeout_no_pay_prob"])
    assert abs(prob_sum - 1.0) < 0.01, f"Probability components sum to {prob_sum}"

    # metrics
    _check_keys(result["metrics"], "mean_ending_balance", "mean_blow_loss",
                "mean_payout", "median_payout", "e_monthly",
                "tier_label", "tier_description", context="metrics")
    assert isinstance(result["metrics"]["tier_label"], str)

    # distributions
    assert isinstance(result["distributions"]["equity_paths"], list)

    _check_json_safe(result, "analyze_full_period result")
    print(f"    payout_prob={probs['payout_prob']:.2%}  "
          f"e_monthly=${result['metrics']['e_monthly']:,.0f}  "
          f"tier={result['metrics']['tier_label']}")


def test_analyze_full_period_seed_determinism() -> None:
    r1 = analyze_full_period(CSV_ES, n_sims=N_SIMS_FAST, n_paths=0, seed=SEED_A)
    r2 = analyze_full_period(CSV_ES, n_sims=N_SIMS_FAST, n_paths=0, seed=SEED_A)
    assert r1["probabilities"]["payout_prob"] == r2["probabilities"]["payout_prob"], \
        "Same seed must produce identical payout_prob"
    print(f"    Seed {SEED_A}: payout_prob={r1['probabilities']['payout_prob']:.4f} (both runs)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — run_batch
# ─────────────────────────────────────────────────────────────────────────────

def test_run_batch_structure() -> None:
    rows = run_batch([CSV_ES, CSV_YM], n_sims=N_SIMS_FAST, reset_cost=137.0,
                     seed=SEED_A)

    assert isinstance(rows, list), "run_batch must return a list"
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    for i, row in enumerate(rows):
        assert isinstance(row, dict), f"Row {i} must be a dict"
        _check_keys(row, "csv", "utp_payout_p", "utp_mean_pay", "utp_ev_net",
                    "utp_rating", "fp_payout_p", "fp_blow_only_p",
                    "fp_timeout_p", "fp_mean_pay", "fp_mean_bal", "fp_ev_net",
                    "fp_rating", context=f"row[{i}]")
        _check_prob(row["utp_payout_p"], f"row[{i}].utp_payout_p")
        _check_prob(row["fp_payout_p"],  f"row[{i}].fp_payout_p")

    _check_json_safe(rows, "run_batch result")

    # Verify sorted by fp_ev_net descending
    ev_vals = [r["fp_ev_net"] for r in rows if "error" not in r]
    assert ev_vals == sorted(ev_vals, reverse=True), \
        "run_batch rows must be sorted by fp_ev_net descending"

    for r in rows:
        print(f"    {r['csv'][:40]}  UTP={r['utp_payout_p']:.1%}  "
              f"FP={r['fp_payout_p']:.1%}  EV=${r['fp_ev_net']:,.0f}")


def test_run_batch_error_handling() -> None:
    """run_batch must not crash on a bad CSV — it records the error in the row."""
    rows = run_batch(["__does_not_exist__.csv"], n_sims=50, seed=SEED_A)
    assert len(rows) == 1
    assert "error" in rows[0], "Failed CSV must produce an 'error' key"
    _check_json_safe(rows, "error row")
    print(f"    Error row: {rows[0]['error'][:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — analyze_correlation
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_correlation_structure() -> None:
    result = analyze_correlation(
        named_csvs={"ES": CSV_ES, "YM": CSV_YM},
        rolling_window=20,
        mc_n_sims=N_CORR_SIMS,
        mc_max_days=21,
        seed=SEED_A,
    )

    assert isinstance(result, dict)
    _check_keys(result, "labels", "return_correlation", "blow_correlation",
                "rolling_pairs", "simultaneous_dd", "worst_overlap",
                "equity_curves", "drawdown_curves")

    labels = result["labels"]
    assert set(labels) == {"ES", "YM"}, f"labels mismatch: {labels}"

    # Correlation matrices must use {labels, matrix} shape
    for key in ("return_correlation", "blow_correlation"):
        corr = result[key]
        _check_keys(corr, "labels", "matrix", context=key)
        assert corr["labels"] == labels, f"{key}.labels mismatch"
        assert len(corr["matrix"]) == len(labels), \
            f"{key}.matrix row count mismatch"
        assert len(corr["matrix"][0]) == len(labels), \
            f"{key}.matrix column count mismatch"
        # Diagonal elements must be 1.0
        for i in range(len(labels)):
            assert abs(corr["matrix"][i][i] - 1.0) < 1e-9, \
                f"{key} diagonal[{i}] != 1.0 (got {corr['matrix'][i][i]})"

    # Equity / drawdown curves must have one list per strategy
    for strat in labels:
        assert strat in result["equity_curves"],  f"equity_curves missing {strat}"
        assert strat in result["drawdown_curves"], f"drawdown_curves missing {strat}"

    _check_json_safe(result, "analyze_correlation result")

    for i, s in enumerate(labels):
        for j, t in enumerate(labels):
            print(f"    return_corr[{s},{t}] = "
                  f"{result['return_correlation']['matrix'][i][j]:.3f}")


def test_analyze_correlation_seed_determinism() -> None:
    r1 = analyze_correlation(
        named_csvs={"ES": CSV_ES, "YM": CSV_YM},
        mc_n_sims=N_CORR_SIMS, mc_max_days=21, seed=SEED_A,
    )
    r2 = analyze_correlation(
        named_csvs={"ES": CSV_ES, "YM": CSV_YM},
        mc_n_sims=N_CORR_SIMS, mc_max_days=21, seed=SEED_A,
    )
    m1 = r1["blow_correlation"]["matrix"]
    m2 = r2["blow_correlation"]["matrix"]
    assert m1 == m2, "Same seed must produce identical blow_correlation matrix"
    print(f"    Seed {SEED_A}: blow_corr[ES,YM]={m1[0][1]:.4f} (both runs)")


def test_analyze_correlation_minimum_strategies() -> None:
    """Must raise ValueError if fewer than 2 strategies provided."""
    try:
        analyze_correlation({"ES": CSV_ES})
        raise AssertionError("Expected ValueError for 1-strategy input")
    except ValueError as exc:
        print(f"    Correctly raised ValueError: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — run_multi_account
# ─────────────────────────────────────────────────────────────────────────────

def test_run_multi_account_structure() -> None:
    result = run_multi_account(
        csv_path    = CSV_ES,
        n_accounts  = 2,
        n_sims      = N_SIMS_FAST,
        max_days    = 30,
        n_path_sims = 0,
        seed        = SEED_A,
    )

    assert isinstance(result, dict)
    _check_keys(result, "metadata", "portfolio", "cycle_efficiency")

    # metadata
    meta = result["metadata"]
    _check_keys(meta, "csv_path", "n_accounts", "n_sims", "max_days",
                "stop_at_payout", "risk_multiplier", "sampling_mode",
                context="metadata")
    assert meta["n_accounts"] == 2
    assert meta["n_sims"] == N_SIMS_FAST

    # portfolio
    port = result["portfolio"]
    _check_keys(port, "mean_total_payout", "median_total_payout",
                "p5_payout", "p95_payout", "prob_any_payout",
                "mean_blown_accounts", "mean_end_balance_single",
                "mean_end_balance_portfolio", "mean_end_balance_per_acct",
                context="portfolio")
    _check_prob(port["prob_any_payout"], "portfolio.prob_any_payout")
    assert isinstance(port["mean_end_balance_per_acct"], list)
    assert len(port["mean_end_balance_per_acct"]) == 2

    # cycle_efficiency — only present when payouts occurred
    if result["cycle_efficiency"]:
        ce = result["cycle_efficiency"]
        _check_keys(ce, "payout_prob_per_cycle", "mean_days_to_payout",
                    "median_days_to_payout", "p5_days", "p95_days",
                    "pct_within_gate", "pct_within_10", "pct_within_15",
                    "est_cycles_per_month", "est_monthly_gross",
                    "est_monthly_net", context="cycle_efficiency")
        _check_prob(ce["payout_prob_per_cycle"], "cycle_efficiency.payout_prob_per_cycle")

    # No private keys (_portfolio_payouts, _account_paths) in API output
    for key in result:
        assert not key.startswith("_"), f"Private key '{key}' leaked into API result"

    _check_json_safe(result, "run_multi_account result")

    port_str = f"mean_payout=${port['mean_total_payout']:,.0f}  prob={port['prob_any_payout']:.2%}"
    ce_str   = (f"  cycles/mo={result['cycle_efficiency']['est_cycles_per_month']:.1f}"
                if result["cycle_efficiency"] else "  (no payouts)")
    print(f"    {port_str}{ce_str}")


def test_run_multi_account_seed_determinism() -> None:
    r1 = run_multi_account(CSV_ES, n_accounts=2, n_sims=N_SIMS_FAST,
                           n_path_sims=0, seed=SEED_A)
    r2 = run_multi_account(CSV_ES, n_accounts=2, n_sims=N_SIMS_FAST,
                           n_path_sims=0, seed=SEED_A)
    v1 = r1["portfolio"]["mean_total_payout"]
    v2 = r2["portfolio"]["mean_total_payout"]
    assert v1 == v2, f"Same seed: mean_total_payout differs: {v1} vs {v2}"
    print(f"    Seed {SEED_A}: mean_total_payout=${v1:,.2f} (both runs)")


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA — full JSON dump of all results together
# ─────────────────────────────────────────────────────────────────────────────

def test_combined_json_serialisation() -> None:
    """Collect results from all five functions and json.dumps the whole bundle."""
    all_results = {
        "analyze_until_payout": analyze_until_payout(CSV_ES, n_sims=N_SIMS_FAST,
                                                     n_paths=3, seed=SEED_A),
        "analyze_full_period":  analyze_full_period(CSV_ES, n_sims=N_SIMS_FAST,
                                                    n_paths=3, seed=SEED_A),
        "run_batch":            run_batch([CSV_ES, CSV_YM], n_sims=N_SIMS_FAST,
                                          seed=SEED_A),
        "analyze_correlation":  analyze_correlation(
                                    {"ES": CSV_ES, "YM": CSV_YM},
                                    mc_n_sims=N_CORR_SIMS, mc_max_days=21,
                                    seed=SEED_A),
        "run_multi_account":    run_multi_account(CSV_ES, n_accounts=2,
                                                  n_sims=N_SIMS_FAST,
                                                  n_path_sims=0, seed=SEED_A),
    }
    serialised = json.dumps(all_results)
    payload_kb = len(serialised) / 1024
    print(f"    json.dumps(all_results) succeeded — {payload_kb:,.1f} KB payload")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run all tests and print summary
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 64

    print(f"\n{sep}")
    print("  engine_interface_test.py  —  Phase 2 Verification")
    print(f"{sep}")
    print(f"  Fast-mode sims : {N_SIMS_FAST}  |  corr sims: {N_CORR_SIMS}")
    print(f"  CSV (ES)       : {CSV_ES}")
    print(f"  CSV (YM)       : {CSV_YM}")
    print(f"{sep}")

    # ── 1. analyze_until_payout ───────────────────────────────────────────────
    run_test("1a. analyze_until_payout — structure & JSON safety",
             test_analyze_until_payout_structure)
    run_test("1b. analyze_until_payout — seed determinism",
             test_analyze_until_payout_seed_determinism)

    # ── 2. analyze_full_period ────────────────────────────────────────────────
    run_test("2a. analyze_full_period — structure & JSON safety",
             test_analyze_full_period_structure)
    run_test("2b. analyze_full_period — seed determinism",
             test_analyze_full_period_seed_determinism)

    # ── 3. run_batch ──────────────────────────────────────────────────────────
    run_test("3a. run_batch — structure & JSON safety",
             test_run_batch_structure)
    run_test("3b. run_batch — error handling for bad CSV",
             test_run_batch_error_handling)

    # ── 4. analyze_correlation ────────────────────────────────────────────────
    run_test("4a. analyze_correlation — structure & JSON safety",
             test_analyze_correlation_structure)
    run_test("4b. analyze_correlation — seed determinism",
             test_analyze_correlation_seed_determinism)
    run_test("4c. analyze_correlation — raises on <2 strategies",
             test_analyze_correlation_minimum_strategies)

    # ── 5. run_multi_account ──────────────────────────────────────────────────
    run_test("5a. run_multi_account — structure & JSON safety",
             test_run_multi_account_structure)
    run_test("5b. run_multi_account — seed determinism",
             test_run_multi_account_seed_determinism)

    # ── Combined ──────────────────────────────────────────────────────────────
    run_test("6.  Combined json.dumps(all_results)",
             test_combined_json_serialisation)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed

    print(f"\n{sep}")
    print(f"  RESULTS  ({passed} passed, {failed} failed, {len(_results)} total)")
    print(sep)

    for name, ok, msg in _results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok and msg:
            for line in msg.splitlines()[:3]:
                print(f"         {line}")

    # ── Final backend interface listing ───────────────────────────────────────
    print(f"\n{sep}")
    print("  CONFIRMED BACKEND INTERFACE  (engine_interface.py v1.2)")
    print(sep)
    endpoints = [
        ("analyze_until_payout",
         "csv_path, n_sims, risk_multiplier, max_days, n_paths,\n"
         "               sampling_mode, weight_strength, recent_window, seed",
         "{metadata, probabilities, metrics, distributions}"),
        ("analyze_full_period",
         "csv_path, n_sims, risk_multiplier, max_days, n_paths, reset_cost,\n"
         "               sampling_mode, weight_strength, recent_window, seed",
         "{metadata, probabilities, metrics, distributions}"),
        ("run_batch",
         "csv_paths, n_sims, sampling_mode, weight_strength, recent_window,\n"
         "               reset_cost, progress_cb, seed",
         "list[{csv, utp_*, fp_*, utp_rating, fp_rating}]"),
        ("analyze_correlation",
         "named_csvs, rolling_window, dd_threshold, mc_n_sims, mc_max_days,\n"
         "               sampling_mode, weight_strength, recent_window, seed",
         "{labels, return_correlation{labels,matrix}, blow_correlation{labels,matrix},\n"
         "               rolling_pairs, simultaneous_dd, worst_overlap,\n"
         "               equity_curves, drawdown_curves}"),
        ("run_multi_account",
         "csv_path, n_accounts, n_sims, max_days, stop_at_payout,\n"
         "               risk_multiplier, sampling_mode, weight_strength,\n"
         "               recent_window, n_path_sims, seed",
         "{metadata, portfolio, cycle_efficiency}"),
    ]
    for fn, params, returns in endpoints:
        print(f"\n  def {fn}(")
        for i, line in enumerate(params.splitlines()):
            print(f"    {line}{',' if i == 0 else ''}")
        print(f"  ) -> {returns}")

    print(f"\n{sep}")
    print("  All functions return JSON-serialisable Python dicts.")
    print("  seed: int | None = None  is supported in every function.")
    print(f"{sep}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
