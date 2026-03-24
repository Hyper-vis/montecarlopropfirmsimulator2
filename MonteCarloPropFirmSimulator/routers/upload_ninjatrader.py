"""NinjaTrader CSV upload endpoint."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

import strategy_db
from ninjatrader_csv_ingestion import parse_ninjatrader_trade_rows


router = APIRouter(tags=["Upload"])


def _build_ninjatrader_canonical_csv(path: Path, rows: list[dict]) -> None:
    """Write NinjaTrader trades to canonical strategy CSV expected by engines."""
    known_times = [r.get("exit_time") for r in rows if r.get("exit_time") is not None]
    if known_times:
        start_fallback = min(known_times)
    else:
        start_fallback = (datetime.now(UTC) - timedelta(days=max(len(rows) - 1, 0))).replace(
            hour=12, minute=0, second=0, microsecond=0
        )

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Type", "Date and time", "Net P&L USD"])
        for idx, row in enumerate(rows):
            dt = row.get("exit_time") or (start_fallback + timedelta(days=idx))
            w.writerow(["Trade", dt.strftime("%Y-%m-%d %H:%M:%S"), f"{float(row['pnl']):.2f}"])


@router.post("/upload/ninjatrader", summary="Upload NinjaTrader trade report and register it for simulation")
async def upload_ninjatrader(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_report"
    extension = Path(filename).suffix.lower()

    if extension != ".csv":
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type. Please upload a NinjaTrader .csv trade report."},
        )

    temp_path: str | None = None
    try:
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        # Validate file size (100 MB max)
        file_size_mb = len(data) / (1024 * 1024)
        if file_size_mb > 100:
            return JSONResponse(
                status_code=413,
                content={"error": f"File is too large ({file_size_mb:.1f} MB). Maximum allowed: 100 MB."}
            )

        file_hash = hashlib.sha256(data).hexdigest()
        existing = strategy_db.find_by_hash_and_source(file_hash, "ninjatrader")
        if existing is not None:
            return {
                "strategy_id": existing["strategy_id"],
                "filename": existing["filename"],
                "path": existing["path"],
                "source": "ninjatrader",
                "duplicate": True,
                "num_trades": None,
                "status": "uploaded",
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(data)
            temp_path = tmp.name

        try:
            rows = parse_ninjatrader_trade_rows(temp_path)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        if not rows:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid trades found in this NinjaTrader report."},
            )

        strategy_id = f"ninjatrader_{uuid4().hex}"
        dest = strategy_db.STRATEGIES_DIR / f"{strategy_id}.csv"
        _build_ninjatrader_canonical_csv(dest, rows)
        strategy_db.insert_strategy(
            strategy_id=strategy_id,
            filename=filename,
            path=str(dest),
            uploaded_at=datetime.now(UTC).isoformat(),
            file_hash=file_hash,
        )

        return {
            "report_type": "ninjatrader",
            "strategy_id": strategy_id,
            "filename": filename,
            "path": str(dest),
            "source": "ninjatrader",
            "duplicate": False,
            "num_trades": len(rows),
            "status": "uploaded",
        }

    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not process this file. Please upload a valid NinjaTrader trade report."},
        )
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
