"""
strategy_score.py
-----------------
Shared scoring/rating logic used by batch_runner.py and the run_* scripts.

Compatible with apex_engine_v3 (and earlier versions) — this module has no
direct engine dependency and works with any pass-rate float input.

Tiers are calibrated differently per script because they answer different questions:
  run_until_payout  — pure edge with no time pressure (best-case ceiling)
  run_full_period   — real operating conditions within a fixed monthly window
"""

# ---------------------------------------------------------------------------
# Tier tables — (min_payout_prob, label, description)
# ---------------------------------------------------------------------------

TIERS_UNTIL_PAYOUT = [
    (0.75, "ELITE",      "Top-tier edge. Extremely high payout rate with no time cap."),
    (0.65, "STRONG",     "Solid strategy. Reliable edge, worth scaling."),
    (0.55, "ACCEPTABLE", "Positive EV but verify mean payout justifies reset risk."),
    (0.45, "MARGINAL",   "Weak edge. Only viable if mean payout is unusually large."),
    (0.00, "REJECT",     "Negative or near-zero EV after reset costs. Do not run."),
]

TIERS_FULL_PERIOD = [
    (0.60, "ELITE",      "Majority of accounts pay out within the month. Scale aggressively."),
    (0.50, "STRONG",     "More than half resolve as payouts. Good monthly capital turn."),
    (0.40, "ACCEPTABLE", "Positive EV but meaningful timeout drag. Monitor closely."),
    (0.30, "MARGINAL",   "Too much capital sitting idle. Consider tightening the strategy."),
    (0.00, "REJECT",     "Poor monthly resolution. Capital tied up without return. Do not run."),
]

# ---------------------------------------------------------------------------
# Tier colors (ANSI — works in most terminals, ignored where unsupported)
# ---------------------------------------------------------------------------

_ANSI = {
    "ELITE":      "\033[95m",   # magenta
    "STRONG":     "\033[92m",   # green
    "ACCEPTABLE": "\033[93m",   # yellow
    "MARGINAL":   "\033[33m",   # dark yellow
    "REJECT":     "\033[91m",   # red
    "RESET":      "\033[0m",
}


def score(payout_prob: float, mode: str) -> dict:
    """
    Score a strategy.

    Parameters
    ----------
    payout_prob : float   — e.g. 0.6931
    mode        : str     — "until_payout" or "full_period"

    Returns dict with keys: label, description, tier_table
    """
    tiers = TIERS_UNTIL_PAYOUT if mode == "until_payout" else TIERS_FULL_PERIOD
    for threshold, label, description in tiers:
        if payout_prob >= threshold:
            return {"label": label, "description": description, "tiers": tiers}
    return {"label": "REJECT", "description": tiers[-1][2], "tiers": tiers}


def print_score(payout_prob: float, mode: str):
    """Print the full tier table and highlight where this strategy lands."""
    result = score(payout_prob, mode)
    label  = result["label"]
    tiers  = result["tiers"]

    col   = _ANSI.get(label, "")
    reset = _ANSI["RESET"]

    mode_label = "Run-Until-Payout" if mode == "until_payout" else "Full-Period (21d)"

    print()
    print(f"  {'='*54}")
    print(f"  Strategy Rating  [{mode_label}]")
    print(f"  {'='*54}")

    # Tier table
    prev_hi = 1.01
    for threshold, tlabel, tdesc in tiers:
        hi_str  = f"{prev_hi*100:.0f}%" if prev_hi < 1.01 else " -- "
        lo_str  = f"{threshold*100:.0f}%"
        range_s = f"{lo_str}-{hi_str}" if prev_hi < 1.01 else f">={lo_str}"
        marker  = "  <<< YOU ARE HERE" if tlabel == label else ""
        tc      = _ANSI.get(tlabel, "")
        print(f"  {tc}{tlabel:<12}{reset}  P(payout) >= {lo_str:<6}  {marker}")
        prev_hi = threshold

    print(f"  {'-'*54}")
    print(f"  This strategy:  P(payout) = {payout_prob:.2%}")
    print(f"  Rating:  {col}{label}{reset}  —  {result['description']}")
    print(f"  {'='*54}")
    print()
