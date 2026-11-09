"""SQLite persistence for DefectX inspection and batch records."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "inspection_history.sqlite3"
)
_INITIALIZED_PATHS: set[str] = set()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_database(db_path: Path | str = DEFAULT_DATABASE_PATH) -> None:
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspection_batches (
                batch_id TEXT PRIMARY KEY,
                inspected_at TEXT NOT NULL,
                total_images INTEGER NOT NULL,
                successful_images INTEGER NOT NULL DEFAULT 0,
                pass_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0,
                review_count INTEGER NOT NULL DEFAULT 0,
                processing_errors INTEGER NOT NULL DEFAULT 0,
                defective_objects INTEGER NOT NULL DEFAULT 0,
                classified_objects INTEGER NOT NULL DEFAULT 0,
                defect_rate REAL,
                decision_confidence_threshold REAL NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('PROCESSING', 'COMPLETED'))
            );

            CREATE TABLE IF NOT EXISTS inspections (
                inspection_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES inspection_batches(batch_id),
                inspected_at TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                product_category TEXT NOT NULL,
                total_objects INTEGER NOT NULL,
                normal_objects INTEGER NOT NULL,
                defective_objects INTEGER NOT NULL,
                uncertain_objects INTEGER NOT NULL,
                product_decision TEXT NOT NULL CHECK (
                    product_decision IN ('PASS', 'FAIL', 'MANUAL REVIEW')
                ),
                decision_explanation TEXT NOT NULL,
                classification_confidences TEXT NOT NULL,
                object_details_json TEXT NOT NULL,
                yolo_ms REAL,
                mobilenet_ms REAL,
                total_inference_ms REAL,
                decision_confidence_threshold REAL NOT NULL,
                original_image_path TEXT,
                annotated_image_path TEXT,
                processing_error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_inspections_batch
                ON inspections(batch_id);
            CREATE INDEX IF NOT EXISTS idx_inspections_decision_time
                ON inspections(product_decision, inspected_at);
            CREATE INDEX IF NOT EXISTS idx_inspections_filename
                ON inspections(image_filename);
            """
        )
    _INITIALIZED_PATHS.add(str(path))


def _ensure_database(db_path: Path | str) -> Path:
    path = Path(db_path).resolve()
    if str(path) not in _INITIALIZED_PATHS or not path.is_file():
        initialize_database(path)
    return path


def create_batch(
    batch_id: str,
    inspected_at: str,
    total_images: int,
    decision_confidence_threshold: float,
    db_path: Path | str = DEFAULT_DATABASE_PATH,
) -> None:
    path = _ensure_database(db_path)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO inspection_batches (
                batch_id, inspected_at, total_images,
                decision_confidence_threshold, status
            ) VALUES (?, ?, ?, ?, 'PROCESSING')
            """,
            (
                batch_id,
                inspected_at,
                int(total_images),
                float(decision_confidence_threshold),
            ),
        )


def complete_batch(
    batch_id: str,
    *,
    successful_images: int,
    pass_count: int,
    fail_count: int,
    review_count: int,
    processing_errors: int,
    defective_objects: int,
    classified_objects: int,
    db_path: Path | str = DEFAULT_DATABASE_PATH,
) -> None:
    path = _ensure_database(db_path)
    defect_rate = (
        float(defective_objects) / classified_objects if classified_objects else None
    )
    with sqlite3.connect(path, timeout=30) as connection:
        cursor = connection.execute(
            """
            UPDATE inspection_batches
            SET successful_images = ?, pass_count = ?, fail_count = ?,
                review_count = ?, processing_errors = ?, defective_objects = ?,
                classified_objects = ?, defect_rate = ?, status = 'COMPLETED'
            WHERE batch_id = ? AND status = 'PROCESSING'
            """,
            (
                int(successful_images),
                int(pass_count),
                int(fail_count),
                int(review_count),
                int(processing_errors),
                int(defective_objects),
                int(classified_objects),
                defect_rate,
                batch_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Inspection batch is missing or already completed: {batch_id}")


def _persist_image(
    image: Image.Image | None, target: Path
) -> str | None:
    if image is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, format="PNG")
    return str(target.resolve())


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def save_inspection(
    record: dict[str, Any],
    *,
    original_image: Image.Image | None = None,
    annotated_image: Image.Image | None = None,
    db_path: Path | str = DEFAULT_DATABASE_PATH,
) -> str:
    path = _ensure_database(db_path)
    inspection_id = str(record.get("inspection_id") or uuid.uuid4())
    image_dir = path.parent / f"{path.stem}_images" / inspection_id
    original_path = None
    annotated_path = None
    try:
        original_path = _persist_image(original_image, image_dir / "original.png")
        annotated_path = _persist_image(annotated_image, image_dir / "annotated.png")
        details = record.get("object_details", [])
        if isinstance(details, str):
            details_json = details
        else:
            details_json = json.dumps(details, ensure_ascii=False, allow_nan=False)
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO inspections (
                    inspection_id, batch_id, inspected_at, image_filename,
                    product_category, total_objects, normal_objects,
                    defective_objects, uncertain_objects, product_decision,
                    decision_explanation, classification_confidences,
                    object_details_json, yolo_ms, mobilenet_ms,
                    total_inference_ms, decision_confidence_threshold,
                    original_image_path, annotated_image_path, processing_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_id,
                    str(record["batch_id"]),
                    str(record.get("inspected_at") or utc_timestamp()),
                    str(record["image_filename"]),
                    str(record.get("product_category") or ""),
                    int(record.get("total_objects", 0)),
                    int(record.get("normal_objects", 0)),
                    int(record.get("defective_objects", 0)),
                    int(record.get("uncertain_objects", 0)),
                    str(record["product_decision"]),
                    str(record["decision_explanation"]),
                    str(record.get("classification_confidences") or ""),
                    details_json,
                    _finite_float(record.get("yolo_ms")),
                    _finite_float(record.get("mobilenet_ms")),
                    _finite_float(record.get("total_inference_ms")),
                    float(record["decision_confidence_threshold"]),
                    original_path,
                    annotated_path,
                    record.get("processing_error"),
                ),
            )
    except Exception:
        if image_dir.exists():
            for item in image_dir.iterdir():
                item.unlink()
            image_dir.rmdir()
        raise
    return inspection_id


def _inspection_from_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["object_details"] = json.loads(result.pop("object_details_json"))
    return result


def get_inspection(
    inspection_id: str, db_path: Path | str = DEFAULT_DATABASE_PATH
) -> dict[str, Any] | None:
    path = _ensure_database(db_path)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)
        ).fetchone()
    return _inspection_from_row(row) if row else None


def list_inspections(
    search: str = "",
    decision: str = "ALL",
    *,
    limit: int = 1000,
    db_path: Path | str = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    path = _ensure_database(db_path)
    escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    decision_clause = "" if decision == "ALL" else "AND product_decision = ?"
    parameters: list[Any] = [pattern, pattern]
    if decision != "ALL":
        parameters.append(decision)
    parameters.append(max(1, min(int(limit), 10_000)))
    with sqlite3.connect(path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT * FROM inspections
            WHERE (inspection_id LIKE ? ESCAPE '\\' OR image_filename LIKE ? ESCAPE '\\')
            {decision_clause}
            ORDER BY inspected_at DESC, inspection_id DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    return [_inspection_from_row(row) for row in rows]


def get_batch(
    batch_id: str, db_path: Path | str = DEFAULT_DATABASE_PATH
) -> dict[str, Any] | None:
    path = _ensure_database(db_path)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM inspection_batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
    return dict(row) if row else None


def list_batches(
    *, limit: int = 1000, db_path: Path | str = DEFAULT_DATABASE_PATH
) -> list[dict[str, Any]]:
    path = _ensure_database(db_path)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM inspection_batches
            WHERE status = 'COMPLETED'
            ORDER BY inspected_at DESC, batch_id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 10_000)),),
        ).fetchall()
    return [dict(row) for row in rows]


def list_batch_inspections(
    batch_id: str, db_path: Path | str = DEFAULT_DATABASE_PATH
) -> list[dict[str, Any]]:
    path = _ensure_database(db_path)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM inspections
            WHERE batch_id = ?
            ORDER BY inspected_at, image_filename
            """,
            (batch_id,),
        ).fetchall()
    return [_inspection_from_row(row) for row in rows]
