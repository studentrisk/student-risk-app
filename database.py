"""
database.py — จัดการ SQLite database สำหรับระบบ authentication
"""
import sqlite3
import bcrypt
from pathlib import Path

DB_PATH = Path(__file__).parent / "teachers.db"


def get_connection():
    """คืนค่า connection ไปยัง SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """สร้างตาราง teachers ถ้ายังไม่มี"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teachers (
                teacher_id   TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                department   TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        conn.commit()


def get_teacher(teacher_id: str) -> dict | None:
    """ดึงข้อมูลอาจารย์จาก database; คืน None ถ้าไม่พบ"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM teachers WHERE teacher_id = ?", (teacher_id,)
        ).fetchone()
    return dict(row) if row else None


def verify_password(plain_password: str, password_hash: str) -> bool:
    """ตรวจสอบรหัสผ่านกับ hash ที่เก็บใน database"""
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
