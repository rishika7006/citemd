"""Database connection helpers.

``psycopg`` and ``pgvector`` are optional (the ``db`` extra). Imports happen lazily inside
the functions so that the pure-logic core of CiteMD can be imported and tested without the
database driver installed.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

from citemd.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg import Connection


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
        from pgvector.psycopg import register_vector  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "The Postgres backend needs the 'db' extra. Install with: pip install 'citemd[db]'"
        ) from exc
    return psycopg, register_vector


def connect(database_url: str | None = None, *, register: bool = True) -> "Connection":
    """Open a psycopg connection, optionally registering the pgvector type adapter.

    ``register`` must be False before the ``vector`` extension exists (i.e. during schema
    initialization), since registering the adapter requires the type to already be present.
    """
    psycopg, register_vector = _require_psycopg()
    url = database_url or get_settings().database_url
    conn = psycopg.connect(url)
    if register:
        register_vector(conn)
    return conn


def schema_sql() -> str:
    """Return the bundled schema DDL as a string."""
    return resources.files("citemd.db").joinpath("schema.sql").read_text(encoding="utf-8")


def init_schema(database_url: str | None = None) -> None:
    """Create the extension, tables, and indexes if they do not already exist."""
    with connect(database_url, register=False) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql())
        conn.commit()
