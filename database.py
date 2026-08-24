"""
database.py
------------
All MySQL operations for the eKYC app, in one place.

Fixes vs. the original mysqldb_operations.py:
  - Connection is no longer opened at import time (that crashed the whole
    app on import if MySQL wasn't reachable). It's created lazily and
    reused, with reconnect-on-drop handling.
  - Credentials come from environment variables / a .env file instead of
    being hardcoded in source.
  - Table is created automatically if it doesn't exist, so a fresh DB
    doesn't need manual setup.
  - Every DB call is wrapped in try/except so a DB hiccup shows a clean
    error in the UI instead of killing the Streamlit process.
"""

import os
import logging
import pandas as pd
import mysql.connector
from mysql.connector import Error as MySQLError

logger = logging.getLogger("ekyc")

DB_CONFIG = {
    "host": os.getenv("EKYC_DB_HOST", "localhost"),
    "user": os.getenv("EKYC_DB_USER", "root"),
    "password": os.getenv("EKYC_DB_PASSWORD", ""),
    "database": os.getenv("EKYC_DB_NAME", "ekyc"),
    "port": int(os.getenv("EKYC_DB_PORT", "3306")),
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) NOT NULL PRIMARY KEY,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    name VARCHAR(255),
    father_name VARCHAR(255),
    dob DATE,
    id_type VARCHAR(255) NOT NULL,
    embedding LONGTEXT
)
"""

_connection = None  # module-level singleton, created lazily


def get_connection():
    """Return a live MySQL connection, (re)creating it if needed."""
    global _connection
    try:
        if _connection is None or not _connection.is_connected():
            _connection = mysql.connector.connect(**DB_CONFIG)
            logger.info("MySQL connection established.")
            _ensure_table(_connection)
        return _connection
    except MySQLError as e:
        logger.error(f"Could not connect to MySQL: {e}")
        raise ConnectionError(
            f"Could not connect to the database ({DB_CONFIG['host']}:{DB_CONFIG['port']}). "
            f"Check EKYC_DB_* environment variables. Original error: {e}"
        )


def _ensure_table(conn):
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()
    cursor.close()


def insert_record(record: dict) -> None:
    """
    record must have keys: ID, Name, Father's Name, DOB (YYYY-MM-DD str),
    ID Type, Embedding (list/array of floats).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        sql = (
            "INSERT INTO users (id, name, father_name, dob, id_type, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        values = (
            record["ID"],
            record["Name"],
            record["Father's Name"],
            record["DOB"],
            record["ID Type"],
            str(record.get("Embedding", "")),
        )
        cursor.execute(sql, values)
        conn.commit()
        logger.info(f"Inserted record for ID={record['ID']}")
    except MySQLError as e:
        conn.rollback()
        logger.error(f"Insert failed: {e}")
        raise
    finally:
        cursor.close()


def fetch_record(record_id: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE id = %s", (record_id,))
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=cols)
    except MySQLError as e:
        logger.error(f"Fetch failed: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()


def is_duplicate(record_id: str) -> bool:
    return not fetch_record(record_id).empty
