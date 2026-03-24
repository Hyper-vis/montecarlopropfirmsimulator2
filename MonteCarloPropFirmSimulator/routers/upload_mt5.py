"""MT5 upload endpoint."""

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
from services.mt5_parser_service import MT5ParserServiceError, parse_mt5_file


router = APIRouter(tags=["Upload"])

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm"}


def _build_mt5_canonical_csv(path: Path, trade_results: list[float]) -> None:
    """Write MT5 trades to canonical strategy CSV format used by existing engines.

    The analysis stack expects columns:
      Type, Date and time, Net P&L USD

    MT5 parser currently returns ordered realised trade PnL but not canonical timestamps,
    so we synthesize deterministic chronological dates (oldest -> newest).
    """
    start_date = (datetime.now(UTC) - timedelta(days=max(len(trade_results) - 1, 0))).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Type", "Date and time", "Net P&L USD"])
        for idx, pnl in enumerate(trade_results):
            dt = start_date + timedelta(days=idx)
            w.writerow(["Trade", dt.strftime("%Y-%m-%d %H:%M:%S"), f"{float(pnl):.2f}"])


@router.post("/upload/mt5", summary="Upload MT5 report and register it for simulation")
async def upload_mt5(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_report"
    extension = Path(filename).suffix.lower()

    if extension not in _ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid file type. Please upload .csv or .xlsx MT5 reports.",
            },
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
        existing = strategy_db.find_by_hash_and_source(file_hash, "mt5")
        if existing is not None:
            return {
                "strategy_id": existing["strategy_id"],
                "filename": existing["filename"],
                "path": existing["path"],
                "source": "mt5",
                "duplicate": True,
                "num_trades": None,
                "status": "uploaded",
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        parsed = parse_mt5_file(temp_path)
        trade_results = [float(v) for v in parsed["trade_results"]]

        strategy_id = f"mt5_{uuid4().hex}"
        dest = strategy_db.STRATEGIES_DIR / f"{strategy_id}.csv"
        _build_mt5_canonical_csv(dest, trade_results)
        strategy_db.insert_strategy(
            strategy_id=strategy_id,
            filename=filename,
            path=str(dest),
            uploaded_at=datetime.now(UTC).isoformat(),
            file_hash=file_hash,
        )

        return {
            "report_type": parsed["report_type"],
            "strategy_id": strategy_id,
            "filename": filename,
            "path": str(dest),
            "source": "mt5",
            "duplicate": False,
            "num_trades": len(trade_results),
            "status": "uploaded",
        }

    except MT5ParserServiceError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not process this file. Please upload a valid MT5 report."},
        )
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
