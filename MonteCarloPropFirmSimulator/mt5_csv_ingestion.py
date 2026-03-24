"""
mt5_csv_ingestion.py
====================
Robust MT5 report ingestion for PassPlan Monte Carlo inputs.

Supports MT5 exports in XLSX and CSV and converts noisy multi-section reports
into a clean realized-trade PnL list:
        List[float]

Supported report families
-------------------------
    1) Strategy Tester Reports
    2) Trade History Reports

Public API
----------
    detect_mt5_report_type(source) -> str
    parse_mt5_trade_results(source) -> list[float]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterable, Literal
import io
import re

import pandas as pd


ReportType = Literal["strategy_tester", "trade_history", "unknown"]


class MT5ParseError(ValueError):
    """Raised when no valid realized trade PnL values can be extracted."""


_PROFIT_HINTS = (
    "profit",
    "net p&l",
    "net pnl",
    "p&l",
    "pnl",
)

_ENTRY_HINTS = (
    "entry",
    "direction",
    "in out",
    "in/out",
)

_HEADER_CONTEXT_HINTS = (
    "ticket",
    "deal",
    "order",
    "time",
    "date",
    "type",
    "entry",
    "symbol",
    "volume",
)

_IGNORE_TYPE_TOKENS = (
    "balance",
    "credit",
    "deposit",
    "withdraw",
    "withdrawal",
    "bonus",
    "correction",
    "charge",
    "fee",
    "interest",
    "tax",
)

_TRADE_TYPE_TOKENS = (
    "buy",
    "sell",
    "trade",
    "close",
    "closed",
    "exit",
    "out",
)

_IGNORE_SYMBOL_TOKENS = (
    "balance",
    "credit",
    "commission",
    "fee",
    "swap",
    "charge",
)

_SECTION_BREAK_WORDS = (
    "orders",
    "deals",
    "positions",
    "results",
    "summary",
    "statistics",
    "report",
    "parameters",
    "inputs",
)

_SUMMARY_ROW_WORDS = (
    "total",
    "summary",
    "result",
    "gross",
    "net",
    "drawdown",
    "balance",
)


@dataclass
class _CandidateTable:
    header_row_idx: int
    table: pd.DataFrame
    score: int


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _norm_header(value: Any) -> str:
    s = _norm_text(value).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _dedupe_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in headers:
        base = raw or "col"
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}_{n + 1}")
    return out


def _is_excel_path(path: Path) -> bool:
    return path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}


def _is_excel_bytes(payload: bytes) -> bool:
    # XLSX is a ZIP container and starts with PK.
    return payload.startswith(b"PK")


def _ensure_string_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.fillna("")
    return df.map(_norm_text)


def _read_raw_csv_table(text: str) -> pd.DataFrame:
    best_df: pd.DataFrame | None = None
    best_score = -1

    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                header=None,
                dtype=str,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            continue

        if df.empty:
            continue

        non_empty_cells = int(df.notna().sum().sum())
        width_bonus = int(df.shape[1]) * 2
        score = non_empty_cells + width_bonus
        if score > best_score:
            best_score = score
            best_df = df

    if best_df is None or best_df.empty:
        raise MT5ParseError("Could not read CSV content.")

    return _ensure_string_frame(best_df)


def _source_to_raw_frames(source: str | Path | bytes | IO[str] | IO[bytes]) -> list[pd.DataFrame]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if _is_excel_path(path):
            xls = pd.ExcelFile(path, engine="openpyxl")
            frames: list[pd.DataFrame] = []
            for name in xls.sheet_names:
                sheet_df = pd.read_excel(path, sheet_name=name, header=None, dtype=str, engine="openpyxl")
                if not sheet_df.empty:
                    frames.append(_ensure_string_frame(sheet_df))
            if not frames:
                raise MT5ParseError("Excel workbook has no readable rows.")
            return frames

        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        return [_read_raw_csv_table(text)]

    if isinstance(source, bytes):
        if _is_excel_bytes(source):
            bio = io.BytesIO(source)
            xls = pd.ExcelFile(bio, engine="openpyxl")
            frames = []
            for name in xls.sheet_names:
                sheet_df = pd.read_excel(io.BytesIO(source), sheet_name=name, header=None, dtype=str, engine="openpyxl")
                if not sheet_df.empty:
                    frames.append(_ensure_string_frame(sheet_df))
            if not frames:
                raise MT5ParseError("Excel workbook has no readable rows.")
            return frames

        try:
            text = source.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = source.decode("latin-1", errors="ignore")
        return [_read_raw_csv_table(text)]

    if hasattr(source, "read"):
        payload = source.read()
        if isinstance(payload, str):
            return [_read_raw_csv_table(payload)]
        if isinstance(payload, bytes):
            return _source_to_raw_frames(payload)
        return [_read_raw_csv_table(str(payload))]

    raise TypeError("Unsupported source type. Provide path, bytes, or file-like object.")


def _flatten_sample_tokens(raw_df: pd.DataFrame, max_rows: int = 250) -> str:
    subset = raw_df.head(max_rows)
    tokens: list[str] = []
    for row in subset.itertuples(index=False):
        for cell in row:
            t = _norm_header(cell)
            if t:
                tokens.append(t)
    return " ".join(tokens)


def _row_non_empty_count(row: Iterable[str]) -> int:
    return sum(1 for c in row if _norm_text(c))


def _row_is_likely_header(values: list[str]) -> bool:
    if _row_non_empty_count(values) < 2:
        return False

    normalized = [_norm_header(v) for v in values if _norm_header(v)]
    if not normalized:
        return False

    has_profit = any(any(h in c for h in _PROFIT_HINTS) for c in normalized)
    has_context = any(any(h in c for h in _HEADER_CONTEXT_HINTS) for c in normalized)
    return has_profit and has_context


def _is_section_break_row(values: list[str]) -> bool:
    non_empty = [_norm_header(v) for v in values if _norm_text(v)]
    if not non_empty:
        return True

    if len(non_empty) == 1 and any(word in non_empty[0] for word in _SECTION_BREAK_WORDS):
        return True

    return False


def _extract_candidate_tables(raw_df: pd.DataFrame) -> list[_CandidateTable]:
    candidates: list[_CandidateTable] = []
    row_count = len(raw_df)

    for row_idx in range(row_count):
        row_values = [str(v) for v in raw_df.iloc[row_idx].tolist()]
        if not _row_is_likely_header(row_values):
            continue

        header = _dedupe_headers([_norm_header(v) for v in row_values])
        data_rows: list[list[str]] = []

        for j in range(row_idx + 1, row_count):
            values = [str(v) for v in raw_df.iloc[j].tolist()]
            if _is_section_break_row(values):
                if data_rows:
                    break
                continue

            if _row_non_empty_count(values) < 2:
                if data_rows:
                    break
                continue

            values = values[: len(header)]
            if len(values) < len(header):
                values += [""] * (len(header) - len(values))
            data_rows.append(values)

        if not data_rows:
            continue

        table = pd.DataFrame(data_rows, columns=header)
        score = len(data_rows) + len(header)
        candidates.append(_CandidateTable(header_row_idx=row_idx, table=table, score=score))

    return candidates


def _find_col(columns: Iterable[str], hints: Iterable[str]) -> str | None:
    cols = list(columns)

    for hint in hints:
        for col in cols:
            if col == hint:
                return col

    for hint in hints:
        for col in cols:
            if hint in col:
                return col

    return None


def _parse_number(value: Any) -> float | None:
    s = _norm_text(value)
    if not s:
        return None

    if s in {"-", "--", "—"}:
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

    s = s.replace(" ", "")
    s = re.sub(r"[^0-9,.-]", "", s)
    if not s:
        return None

    if s.count("-") > 1:
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        right = s.split(",")[-1]
        if len(right) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _row_has_summary_words(row: pd.Series) -> bool:
    row_text = " ".join(_norm_header(v) for v in row.tolist() if _norm_text(v))
    if not row_text:
        return False
    return any(word in row_text for word in _SUMMARY_ROW_WORDS)


def _is_realized_trade_row(
    row: pd.Series,
    type_col: str | None,
    entry_col: str | None,
    ticket_col: str | None,
    time_col: str | None,
    symbol_col: str | None,
) -> bool:
    if _row_has_summary_words(row):
        return False

    if symbol_col:
        symbol_val = _norm_header(row.get(symbol_col, ""))
        if any(tok in symbol_val for tok in _IGNORE_SYMBOL_TOKENS):
            return False

    if entry_col:
        entry_val = _norm_header(row.get(entry_col, ""))
        if entry_val:
            normalized_entry = entry_val.replace(" ", "")
            if normalized_entry not in {"out", "inout", "close", "closed", "exit"}:
                return False

    if type_col:
        type_val = _norm_header(row.get(type_col, ""))
        if any(token in type_val for token in _IGNORE_TYPE_TOKENS):
            return False
        if type_val and any(token in type_val for token in _TRADE_TYPE_TOKENS):
            return True

    if ticket_col:
        ticket_val = _parse_number(row.get(ticket_col, ""))
        if ticket_val is not None:
            return True

    if time_col:
        ts = pd.to_datetime([row.get(time_col, "")], errors="coerce")
        if len(ts) > 0 and pd.notna(ts[0]):
            return True

    return type_col is None and entry_col is None


def _extract_realized_pnl_from_table(table: pd.DataFrame) -> list[float]:
    columns = [_norm_header(c) for c in table.columns.tolist()]
    rename_map = {old: new for old, new in zip(table.columns.tolist(), columns)}
    df = table.rename(columns=rename_map)

    profit_col = _find_col(df.columns, _PROFIT_HINTS)
    if not profit_col:
        return []

    type_col = _find_col(df.columns, ["type", "action", "deal type"])
    entry_col = _find_col(df.columns, _ENTRY_HINTS)
    ticket_col = _find_col(df.columns, ["ticket", "deal", "order"])
    time_col = _find_col(df.columns, ["time", "date", "date and time"])
    symbol_col = _find_col(df.columns, ["symbol", "instrument", "asset"])

    pnls: list[float] = []
    for _, row in df.iterrows():
        pnl = _parse_number(row.get(profit_col, ""))
        if pnl is None:
            continue
        if not _is_realized_trade_row(row, type_col, entry_col, ticket_col, time_col, symbol_col):
            continue
        pnls.append(float(pnl))

    return pnls


def _extract_realized_trade_records_from_table(table: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [_norm_header(c) for c in table.columns.tolist()]
    rename_map = {old: new for old, new in zip(table.columns.tolist(), columns)}
    df = table.rename(columns=rename_map)

    profit_col = _find_col(df.columns, _PROFIT_HINTS)
    if not profit_col:
        return []

    type_col = _find_col(df.columns, ["type", "action", "deal type"])
    entry_col = _find_col(df.columns, _ENTRY_HINTS)
    ticket_col = _find_col(df.columns, ["ticket", "deal", "order"])
    time_cols = [col for col in df.columns if "time" in col or "date" in col]
    symbol_col = _find_col(df.columns, ["symbol", "instrument", "asset"])

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        pnl = _parse_number(row.get(profit_col, ""))
        if pnl is None:
            continue
        if not _is_realized_trade_row(row, type_col, entry_col, ticket_col, None, symbol_col):
            continue

        timestamp = None
        for col in reversed(time_cols):
            raw_ts = row.get(col, "")
            if not _norm_text(raw_ts):
                continue
            parsed = pd.to_datetime(raw_ts, errors="coerce")
            if pd.notna(parsed):
                timestamp = pd.Timestamp(parsed).isoformat()
                break

        records.append({
            "timestamp": timestamp,
            "pnl": float(pnl),
        })

    return records


def _detect_from_tokens(tokens: str, candidates: list[_CandidateTable]) -> ReportType:
    strategy_markers = (
        "strategy tester report",
        "expert",
        "symbol",
        "period",
        "model",
        "deals",
        "orders",
    )
    history_markers = (
        "trade history",
        "history",
        "closed transactions",
        "position",
        "positions",
    )

    strategy_score = sum(1 for m in strategy_markers if m in tokens)
    history_score = sum(1 for m in history_markers if m in tokens)

    for candidate in candidates:
        cols = set(_norm_header(c) for c in candidate.table.columns)
        if "direction" in cols or "entry" in cols or "deal" in cols:
            strategy_score += 2
        if "position" in cols and "type" in cols:
            history_score += 2
        if "date and time" in cols and "type" in cols:
            history_score += 2

    if strategy_score > history_score and strategy_score >= 2:
        return "strategy_tester"
    if history_score > strategy_score and history_score >= 2:
        return "trade_history"
    return "unknown"


def _collect_candidates(frames: list[pd.DataFrame]) -> tuple[list[_CandidateTable], str]:
    all_candidates: list[_CandidateTable] = []
    token_blocks: list[str] = []
    for raw in frames:
        token_blocks.append(_flatten_sample_tokens(raw))
        all_candidates.extend(_extract_candidate_tables(raw))
    return all_candidates, " ".join(token_blocks)


def detect_mt5_report_type(source: str | Path | bytes | IO[str] | IO[bytes]) -> ReportType:
    """Auto-detect MT5 report family.

    Returns
    -------
    "strategy_tester" | "trade_history" | "unknown"
    """
    frames = _source_to_raw_frames(source)
    candidates, tokens = _collect_candidates(frames)
    return _detect_from_tokens(tokens, candidates)


def parse_mt5_trade_results(source: str | Path | bytes | IO[str] | IO[bytes]) -> list[float]:
    """Parse an MT5 export and return realized trade PnL values.

    Parameters
    ----------
    source:
        Path, raw bytes, or file-like stream of the uploaded MT5 XLSX/CSV.

    Returns
    -------
    list[float]
        Clean realized trade PnL list suitable for Monte Carlo sampling.

    Raises
    ------
    MT5ParseError
        If no valid realized trade PnL values could be extracted.
    """
    frames = _source_to_raw_frames(source)
    candidates, _ = _collect_candidates(frames)

    if not candidates:
        raise MT5ParseError("No valid MT5 trade table detected in source file.")

    best_pnls: list[float] = []
    best_score = -1

    for candidate in candidates:
        pnls = _extract_realized_pnl_from_table(candidate.table)
        score = (len(pnls) * 100) + candidate.score
        if pnls and score > best_score:
            best_pnls = pnls
            best_score = score

    if not best_pnls:
        raise MT5ParseError(
            "No realized trade profit values found. "
            "Ensure the CSV contains a trade/deals table with a Profit/PnL column."
        )

    return [float(v) for v in best_pnls]


def parse_mt5_trade_records(source: str | Path | bytes | IO[str] | IO[bytes]) -> list[dict[str, Any]]:
    """Parse an MT5 export and return realized trade rows with timestamps.

    The returned list preserves the original trade order when timestamps are
    missing.  When timestamps are available, callers can sort chronologically.
    """
    frames = _source_to_raw_frames(source)
    candidates, _ = _collect_candidates(frames)

    if not candidates:
        raise MT5ParseError("No valid MT5 trade table detected in source file.")

    best_records: list[dict[str, Any]] = []
    best_score = -1

    for candidate in candidates:
        records = _extract_realized_trade_records_from_table(candidate.table)
        score = (len(records) * 100) + candidate.score
        if records and score > best_score:
            best_records = records
            best_score = score

    if not best_records:
        raise MT5ParseError(
            "No realized trade profit values found. "
            "Ensure the CSV contains a trade/deals table with a Profit/PnL column."
        )

    if any(record.get("timestamp") for record in best_records):
        best_records = sorted(
            best_records,
            key=lambda record: (
                record.get("timestamp") is None,
                record.get("timestamp") or "",
            ),
        )

    return best_records
