"""MySQL persistence for verified eKYC records."""
from __future__ import annotations

import json

import mysql.connector
from mysql.connector import Error

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS kyc_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pan_number VARCHAR(10) NOT NULL UNIQUE,
    full_name VARCHAR(150) NOT NULL,
    father_name VARCHAR(150),
    dob DATE NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    cosine_distance FLOAT NOT NULL,
    face_embedding JSON,
    status ENUM('VERIFIED','REJECTED') NOT NULL DEFAULT 'VERIFIED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
"""


def _connect(with_db: bool = True):
    kwargs = dict(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        connection_timeout=5,
    )
    if with_db:
        kwargs["database"] = config.DB_NAME
    return mysql.connector.connect(**kwargs)


def init_db() -> None:
    """Create the database and table if they do not exist."""
    conn = _connect(with_db=False)
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}`")
        conn.database = config.DB_NAME
        cur.execute(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def pan_exists(pan: str) -> bool:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM kyc_records WHERE pan_number=%s", (pan,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def save_record(pan, name, father_name, dob_ddmmyyyy, similarity, distance, embedding) -> int:
    """Insert a verified record. dob must be DD/MM/YYYY. Returns the new row id."""
    from datetime import datetime

    dob = datetime.strptime(dob_ddmmyyyy, "%d/%m/%Y").date()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO kyc_records
               (pan_number, full_name, father_name, dob, cosine_similarity,
                cosine_distance, face_embedding, status)
               VALUES (%s,%s,%s,%s,%s,%s,%s,'VERIFIED')""",
            (
                pan,
                name,
                father_name,
                dob,
                float(similarity),
                float(distance),
                json.dumps([float(x) for x in embedding]),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_records(limit: int = 50):
    conn = _connect()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, pan_number, full_name, father_name, dob, "
            "ROUND(cosine_similarity,4) AS similarity, status, created_at "
            "FROM kyc_records ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def check_connection() -> tuple[bool, str]:
    try:
        init_db()
        return True, "Connected"
    except Error as e:
        return False, str(e)
