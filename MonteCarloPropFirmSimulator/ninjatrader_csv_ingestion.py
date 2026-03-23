"""
ninjatrader_csv_ingestion.py
----------------------------
Parses a NinjaTrader trade performance report (exported as CSV) and extracts
the realized PnL for each completed trade as a List[float].

NinjaTrader Profit column format:
  Positive value : $769.24
  Negative value : ($455.76)   ← accounting parentheses, not a minus sign

Usage
-----
    from ninjatrader_csv_ingestion import parse_ninjatrader_trade_results

    results = parse_ninjatrader_trade_results("trades.csv")
    # e.g. [769.24, -455.76, -60.76, 439.24, ...]
"""

import csv
import re
from datetime import datetime
from typing import List


# Matches an optional leading "(" for negatives, an optional "$", digits/commas/periods,
# and an optional closing ")" — e.g.  "$769.24"  or  "($455.76)"
_PROFIT_RE = re.compile(r"^\(?\$?([\d,]+\.?\d*)\)?$")


def _parse_profit(raw: str) -> float | None:
    """
    Convert a single NinjaTrader profit cell to a float.

    Handles:
      "$769.24"    →  769.24
      "($455.76)"  → -455.76
      ""           →  None  (skip)
      "$0.00"      →  0.0

    Returns None if the value cannot be parsed.
    """
    value = raw.strip()
    if not value:
        return None

    negative = value.startswith("(") and value.endswith(")")

    m = _PROFIT_RE.match(value)
    if not m:
        return None

    # Remove any thousand-separator commas before converting
    amount = float(m.group(1).replace(",", ""))
    return -amount if negative else amount


def parse_ninjatrader_trade_results(file_path: str) -> List[float]:
    """
    Load a NinjaTrader CSV trade report and return the list of per-trade
    realized PnL values ready for Monte Carlo simulation.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the NinjaTrader CSV export file.

    Returns
    -------
    List[float]
        Realized PnL per trade, in chronological order.
        Rows with missing or unparseable Profit values are silently dropped.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If no "Profit" column can be found in the CSV header.
    """
    trade_results: List[float] = []

    with open(file_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)

        # Validate that the expected column is present
        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {file_path}")

        # Normalise header names (strip whitespace) to be resilient against
        # minor formatting differences in NinjaTrader exports
        normalised_headers = {h.strip(): h for h in reader.fieldnames if h}
        if "Profit" not in normalised_headers:
            raise ValueError(
                f"No 'Profit' column found in {file_path}. "
                f"Available columns: {list(normalised_headers.keys())}"
            )

        profit_key = normalised_headers["Profit"]

        for row in reader:
            raw = row.get(profit_key, "")
            if raw is None:
                continue  # missing cell

            value = _parse_profit(raw)
            if value is None:
                # Skip empty rows, header repetitions, summary lines, etc.
                continue

            trade_results.append(value)

    return trade_results


def parse_ninjatrader_trade_rows(file_path: str) -> List[dict]:
    """Parse NinjaTrader CSV into timestamped trade rows.

    Returns rows in source order with keys:
      - exit_time: datetime | None
      - pnl: float
    """
    rows: List[dict] = []

    with open(file_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {file_path}")

        normalised_headers = {h.strip(): h for h in reader.fieldnames if h}
        if "Profit" not in normalised_headers:
            raise ValueError(
                f"No 'Profit' column found in {file_path}. "
                f"Available columns: {list(normalised_headers.keys())}"
            )

        profit_key = normalised_headers["Profit"]
        exit_key = normalised_headers.get("Exit time")

        for row in reader:
            raw = row.get(profit_key, "")
            if raw is None:
                continue
            pnl = _parse_profit(raw)
            if pnl is None:
                continue

            exit_dt = None
            if exit_key:
                exit_raw = (row.get(exit_key) or "").strip()
                if exit_raw:
                    # NinjaTrader export example: 2/13/2025 11:31:46 AM
                    try:
                        exit_dt = datetime.strptime(exit_raw, "%m/%d/%Y %I:%M:%S %p")
                    except ValueError:
                        exit_dt = None

            rows.append({"exit_time": exit_dt, "pnl": float(pnl)})

    return rows


# ---------------------------------------------------------------------------
# Example usage (run this file directly to test against a local CSV)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"NinjaTraderCsvs\32k profit, 5k drawdown, 1.5 profit factor. (1 year).csv"
    )

    try:
        results = parse_ninjatrader_trade_results(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Parsed {len(results)} trades from: {path}")
    print(f"First 10 results : {results[:10]}")
    print(f"Last  10 results : {results[-10:]}")
    print(f"Total PnL        : ${sum(results):,.2f}")
    print(f"Win rate         : {sum(1 for r in results if r > 0) / len(results):.1%}")
