"""
db.py — asyncpg connection pool + CRUD helpers for Face Re-ID API
"""
import os
import logging
import asyncpg

logger = logging.getLogger("db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://faceid_user:faceid_pass@localhost:5432/faceid_db"
)

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    try:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        # Auto-migrate: add image_path columns if they don't exist yet (for existing DBs)
        async with _pool.acquire() as conn:
            await conn.execute(
                "ALTER TABLE attendance_log ADD COLUMN IF NOT EXISTS image_path VARCHAR(500)"
            )
            await conn.execute(
                "ALTER TABLE unknown_log ADD COLUMN IF NOT EXISTS image_path VARCHAR(500)"
            )
        logger.info("PostgreSQL pool initialized.")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        _pool = None


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _pool_ok() -> bool:
    return _pool is not None


# ─────────────────────────────────────────────
# Attendance logging
# ─────────────────────────────────────────────

async def log_attendance(name: str, similarity: float, image_path: str | None = None) -> int | None:
    """Insert attendance log. Returns the new row id."""
    if not _pool_ok():
        return None
    try:
        async with _pool.acquire() as conn:
            row_id = await conn.fetchval(
                "INSERT INTO attendance_log (name, similarity, image_path) VALUES ($1, $2, $3) RETURNING id",
                name, float(similarity), image_path
            )
            return row_id
    except Exception as e:
        logger.error(f"log_attendance error: {e}")
        return None


async def log_unknown(image_path: str | None = None) -> int | None:
    """Insert unknown log. Returns the new row id."""
    if not _pool_ok():
        return None
    try:
        async with _pool.acquire() as conn:
            row_id = await conn.fetchval(
                "INSERT INTO unknown_log (image_path) VALUES ($1) RETURNING id",
                image_path
            )
            return row_id
    except Exception as e:
        logger.error(f"log_unknown error: {e}")
        return None


# ─────────────────────────────────────────────
# Query attendance
# ─────────────────────────────────────────────

async def get_attendance(limit: int = 200, name: str | None = None,
                         date_from: str | None = None, date_to: str | None = None):
    if not _pool_ok():
        return []
    try:
        conditions, params = [], []
        idx = 1
        if name:
            conditions.append(f"name ILIKE ${idx}"); params.append(f"%{name}%"); idx += 1
        if date_from:
            conditions.append(f"detected_at >= ${idx}"); params.append(date_from); idx += 1
        if date_to:
            conditions.append(f"detected_at <= ${idx}"); params.append(date_to); idx += 1
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = await _pool.fetch(
            f"SELECT * FROM attendance_log {where} ORDER BY detected_at DESC LIMIT ${idx}",
            *params
        )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_attendance error: {e}")
        return []


async def get_unknowns(limit: int = 200, date_from: str | None = None,
                       date_to: str | None = None):
    if not _pool_ok():
        return []
    try:
        conditions, params = [], []
        idx = 1
        if date_from:
            conditions.append(f"detected_at >= ${idx}"); params.append(date_from); idx += 1
        if date_to:
            conditions.append(f"detected_at <= ${idx}"); params.append(date_to); idx += 1
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = await _pool.fetch(
            f"SELECT * FROM unknown_log {where} ORDER BY detected_at DESC LIMIT ${idx}",
            *params
        )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_unknowns error: {e}")
        return []


async def get_stats():
    if not _pool_ok():
        return {}
    try:
        async with _pool.acquire() as conn:
            total_known   = await conn.fetchval("SELECT COUNT(*) FROM attendance_log")
            total_unknown = await conn.fetchval("SELECT COUNT(*) FROM unknown_log")
            today_known   = await conn.fetchval(
                "SELECT COUNT(*) FROM attendance_log WHERE detected_at::date = CURRENT_DATE")
            today_unknown = await conn.fetchval(
                "SELECT COUNT(*) FROM unknown_log WHERE detected_at::date = CURRENT_DATE")
            unique_today  = await conn.fetchval(
                "SELECT COUNT(DISTINCT name) FROM attendance_log WHERE detected_at::date = CURRENT_DATE")
        return {
            "total_known": total_known,
            "total_unknown": total_unknown,
            "today_known": today_known,
            "today_unknown": today_unknown,
            "unique_today": unique_today,
        }
    except Exception as e:
        logger.error(f"get_stats error: {e}")
        return {}


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────

async def load_settings() -> dict:
    if not _pool_ok():
        return {}
    try:
        rows = await _pool.fetch("SELECT key, value FROM app_settings")
        return {r["key"]: r["value"] for r in rows}
    except Exception as e:
        logger.error(f"load_settings error: {e}")
        return {}


async def save_setting(key: str, value: str) -> None:
    if not _pool_ok():
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO app_settings (key, value) VALUES ($1, $2)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                key, value
            )
    except Exception as e:
        logger.error(f"save_setting error: {e}")