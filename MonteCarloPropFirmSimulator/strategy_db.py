"""
strategy_db.py
==============
SQLite-backed strategy registry for PassPlan.

Replaces the flat strategies/registry.json file with a proper SQLite database
at strategies/strategies.db.

Public API — Strategies
-----------------------
  initialize_db()                       — create DB + tables if missing
  insert_strategy(id, filename, path,   — add a new row
      uploaded_at, file_hash)
  get_strategy(strategy_id)             — fetch one row as dict (or None)
  find_by_hash(file_hash)               — deduplication: fetch by SHA-256 hash
  list_strategies()                     — all rows, newest first
  delete_strategy(strategy_id)          — remove DB row (caller deletes file)
  migrate_from_json(registry_path)      — one-shot import of old registry.json

Public API — Leaderboard
------------------------
  insert_simulation_result(...)         — persist one simulation run
  get_leaderboard(simulation_type,      — ranked rows for a given metric
      metric, limit)
  get_strategy_performance(strategy_id) — all simulation records for a strategy
  delete_simulation_results(strategy_id)— remove all records for a strategy

Public API — Strategy Features
------------------------------
  insert_strategy_features(strategy_id, features) — store extracted feature dict
  get_strategy_features(strategy_id)              — fetch features for one strategy
  list_all_strategy_features()                    — all feature rows, newest first

Thread safety
-------------
  SQLite itself handles concurrent access safely.  We pass
  check_same_thread=False so uvicorn worker threads can share the connection.
  All writes use context managers that commit/rollback atomically.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES_DIR = Path("strategies")
DB_PATH        = STRATEGIES_DIR / "strategies.db"

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Open (or reopen) a connection to the SQLite file."""
    STRATEGIES_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # safer for concurrent readers
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_db() -> None:
    """Create the database and all tables if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                path        TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                file_hash   TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_results (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id            TEXT    NOT NULL,
                simulation_type        TEXT    NOT NULL,
                pass_probability       REAL,
                fail_probability       REAL,
                expected_resets        REAL,
                expected_cost          REAL,
                expected_monthly_payout REAL,
                max_drawdown           REAL,
                sharpe                 REAL,
                profit_factor          REAL,
                num_simulations        INTEGER,
                created_at             TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_simres_strategy ON simulation_results (strategy_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_simres_type ON simulation_results (simulation_type)"
        )
        # ── Migrate: add recency columns to existing databases ────────────────
        for _migration in [
            "ALTER TABLE simulation_results ADD COLUMN recent_pass_probability REAL",
            "ALTER TABLE simulation_results ADD COLUMN probability_delta        REAL",
            "ALTER TABLE simulation_results ADD COLUMN recency_status           TEXT",
        ]:
            try:
                conn.execute(_migration)
            except sqlite3.OperationalError:
                pass   # column already exists — safe to ignore
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_features (
                strategy_id      TEXT PRIMARY KEY,
                num_trades       INTEGER,
                win_rate         REAL,
                avg_win          REAL,
                avg_loss         REAL,
                rr_ratio         REAL,
                expectancy       REAL,
                profit_factor    REAL,
                std_dev          REAL,
                variance         REAL,
                skew             REAL,
                kurtosis         REAL,
                max_drawdown     REAL,
                max_win_streak   INTEGER,
                max_loss_streak  INTEGER,
                created_at       TEXT NOT NULL
            )
            """
        )
    log.info("strategy_db: database ready at %s", DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insert_strategy(
    strategy_id: str,
    filename: str,
    path: str,
    uploaded_at: str,
    file_hash: Optional[str] = None,
) -> None:
    """Insert a new strategy row.  Raises sqlite3.IntegrityError on duplicate PK."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO strategies (strategy_id, filename, path, uploaded_at, file_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (strategy_id, filename, path, uploaded_at, file_hash),
        )


def get_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    """Return the strategy dict for *strategy_id*, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM strategies WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def find_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Return the first strategy whose file_hash matches, or None.
    Used for deduplication on upload.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM strategies WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_strategies() -> List[Dict[str, Any]]:
    """Return all strategies sorted by upload time (newest first)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY uploaded_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_strategy(strategy_id: str) -> bool:
    """
    Delete the DB row for *strategy_id*.
    Returns True if a row was deleted, False if it did not exist.
    The caller is responsible for deleting the CSV file on disk.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM strategies WHERE strategy_id = ?",
            (strategy_id,),
        )
    return cursor.rowcount > 0


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard — simulation results
# ─────────────────────────────────────────────────────────────────────────────

# Allowed metric columns and their sort direction (True = DESC, False = ASC)
_METRIC_CONFIG: Dict[str, tuple] = {
    "pass_probability":          ("pass_probability",          True),
    "expected_monthly_payout":   ("expected_monthly_payout",   True),
    "sharpe":                    ("sharpe",                    True),
    "profit_factor":             ("profit_factor",             True),
    "max_drawdown":               ("max_drawdown",               False),  # lower is better
    "recent_pass_probability":   ("recent_pass_probability",   True),   # recency: highest recent pass % first
    "probability_delta":          ("probability_delta",          True),   # recency: most improving first
}

_VALID_SIM_TYPES = {"until_payout", "full_period", "batch", "multi_account"}


def insert_simulation_result(
    strategy_id: str,
    simulation_type: str,
    *,
    pass_probability: Optional[float] = None,
    fail_probability: Optional[float] = None,
    expected_resets: Optional[float] = None,
    expected_cost: Optional[float] = None,
    expected_monthly_payout: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    sharpe: Optional[float] = None,
    profit_factor: Optional[float] = None,
    num_simulations: Optional[int] = None,
    recent_pass_probability: Optional[float] = None,
    probability_delta: Optional[float] = None,
    recency_status: Optional[str] = None,
    created_at: Optional[str] = None,
) -> int:
    """
    Persist one simulation run.  Returns the new row id.

    All metric fields are optional — pass only the ones your simulation type
    produces; the rest are stored as NULL.
    """
    if created_at is None:
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO simulation_results (
                strategy_id, simulation_type,
                pass_probability, fail_probability,
                expected_resets, expected_cost,
                expected_monthly_payout, max_drawdown,
                sharpe, profit_factor,
                num_simulations, created_at,
                recent_pass_probability, probability_delta, recency_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id, simulation_type,
                pass_probability, fail_probability,
                expected_resets, expected_cost,
                expected_monthly_payout, max_drawdown,
                sharpe, profit_factor,
                num_simulations, created_at,
                recent_pass_probability, probability_delta, recency_status,
            ),
        )
    return cursor.lastrowid


def get_leaderboard(
    simulation_type: str = "until_payout",
    metric: str = "pass_probability",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Return up to *limit* rows ranked by *metric* for *simulation_type*.

    Each strategy is represented by its most recent simulation result only.
    Invalid simulation_type / metric values return an empty list.
    """
    if simulation_type not in _VALID_SIM_TYPES:
        return []
    if metric not in _METRIC_CONFIG:
        return []

    col, desc = _METRIC_CONFIG[metric]
    direction = "DESC" if desc else "ASC"

    # One row per strategy — keep the most recent run
    sql = f"""
        SELECT
            s.strategy_id,
            s.filename,
            r.pass_probability,
            r.fail_probability,
            r.expected_monthly_payout,
            r.max_drawdown,
            r.sharpe,
            r.profit_factor,
            r.num_simulations,
            r.simulation_type,
            r.created_at,
            r.recent_pass_probability,
            r.probability_delta,
            r.recency_status
        FROM simulation_results r
        JOIN strategies s ON r.strategy_id = s.strategy_id
        WHERE r.simulation_type = ?
          AND r.{col} IS NOT NULL
          AND r.id = (
              SELECT id FROM simulation_results r2
              WHERE r2.strategy_id = r.strategy_id
                AND r2.simulation_type = r.simulation_type
              ORDER BY r2.id DESC LIMIT 1
          )
        ORDER BY r.{col} {direction}
        LIMIT ?
    """
    with _connect() as conn:
        rows = conn.execute(sql, (simulation_type, limit)).fetchall()

    result = []
    for rank, row in enumerate(rows, 1):
        d = _row_to_dict(row)
        d["rank"] = rank
        result.append(d)
    return result


def get_strategy_performance(strategy_id: str) -> List[Dict[str, Any]]:
    """
    Return all simulation records for a strategy, newest first.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM simulation_results
            WHERE strategy_id = ?
            ORDER BY id DESC
            """,
            (strategy_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_simulation_results(strategy_id: str) -> int:
    """Remove all simulation records for *strategy_id*.  Returns rows deleted."""
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM simulation_results WHERE strategy_id = ?",
            (strategy_id,),
        )
    return cursor.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# Strategy features
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_COLS = (
    "num_trades", "win_rate", "avg_win", "avg_loss", "rr_ratio",
    "expectancy", "profit_factor", "std_dev", "variance",
    "skew", "kurtosis", "max_drawdown", "max_win_streak", "max_loss_streak",
)


def insert_strategy_features(
    strategy_id: str,
    features: Dict[str, Any],
) -> None:
    """
    Upsert the extracted feature dict for *strategy_id*.

    Uses INSERT OR REPLACE so re-uploading an identical hash still works if the
    deduplication check is bypassed (e.g. during testing).
    """
    from datetime import datetime, timezone
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO strategy_features
                (strategy_id,
                 num_trades, win_rate, avg_win, avg_loss,
                 rr_ratio, expectancy, profit_factor,
                 std_dev, variance, skew, kurtosis,
                 max_drawdown, max_win_streak, max_loss_streak,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                features.get("num_trades"),
                features.get("win_rate"),
                features.get("avg_win"),
                features.get("avg_loss"),
                features.get("rr_ratio"),
                features.get("expectancy"),
                features.get("profit_factor"),
                features.get("std_dev"),
                features.get("variance"),
                features.get("skew"),
                features.get("kurtosis"),
                features.get("max_drawdown"),
                features.get("max_win_streak"),
                features.get("max_loss_streak"),
                created_at,
            ),
        )
    log.info("strategy_db: features stored for strategy %s (%d trades)",
             strategy_id, features.get("num_trades", 0))


def get_strategy_features(strategy_id: str) -> Optional[Dict[str, Any]]:
    """Return the feature dict for *strategy_id*, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM strategy_features WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_all_strategy_features() -> List[Dict[str, Any]]:
    """Return all feature rows, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM strategy_features ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# One-shot migration from legacy registry.json
# ─────────────────────────────────────────────────────────────────────────────

def migrate_from_json(registry_path: Path) -> None:
    """
    Import every entry from a legacy registry.json into SQLite, then rename
    the JSON file to registry_migrated_backup.json so it is never re-processed.

    Safe to call even when registry.json does not exist or is already migrated.
    """
    if not registry_path.exists():
        return

    backup_path = registry_path.parent / "registry_migrated_backup.json"
    if backup_path.exists():
        # Migration already ran; nothing to do.
        return

    log.info("strategy_db: migrating %s → SQLite …", registry_path)

    try:
        with registry_path.open(encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)
    except Exception as exc:
        log.warning("strategy_db: could not read registry.json (%s) — skipping migration", exc)
        return

    migrated = 0
    for strategy_id, entry in data.items():
        # Skip if already present (idempotent)
        if get_strategy(strategy_id) is not None:
            continue
        try:
            insert_strategy(
                strategy_id = strategy_id,
                filename    = entry.get("filename", "unknown.csv"),
                path        = entry.get("path", ""),
                uploaded_at = entry.get("uploaded_at", ""),
                file_hash   = entry.get("file_hash"),   # may not exist in old data
            )
            migrated += 1
        except Exception as exc:
            log.warning("strategy_db: skipping %s during migration (%s)", strategy_id, exc)

    # Rename so we never process it again
    registry_path.rename(backup_path)
    log.info("strategy_db: migrated %d entries; registry renamed to %s", migrated, backup_path.name)
