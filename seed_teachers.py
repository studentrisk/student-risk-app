"""
seed_teachers.py — สร้าง account อาจารย์ 6 คนใน database (รัน 1 ครั้ง)
รหัสผ่านจะถูก hash ด้วย bcrypt ก่อนเก็บลง database

วิธีใช้:
    python seed_teachers.py
"""
import bcrypt
from database import get_connection, init_db

# ======================================================
# ข้อมูล account อาจารย์ในสาขา (ชั่วคราว — เปลี่ยนภายหลังได้)
# ======================================================
TEACHERS_SEED = [
    {
        "teacher_id": "CS001",
        "name": "ผศ.ดร.วิชัย ประสิทธิ์ศักดิ์",
        "department": "สาขาวิทยาการคอมพิวเตอร์",
        "password": "Cs@2025_01",
    },
    {
        "teacher_id": "CS002",
        "name": "อ.สุภาพร เจริญสุข",
        "department": "สาขาวิทยาการคอมพิวเตอร์",
        "password": "Cs@2025_02",
    },
    {
        "teacher_id": "CS003",
        "name": "ผศ.ณัฐพล มั่นคง",
        "department": "สาขาวิทยาการคอมพิวเตอร์",
        "password": "Cs@2025_03",
    },
    {
        "teacher_id": "CS004",
        "name": "อ.พรรณิภา ศรีวิไล",
        "department": "สาขาวิทยาการคอมพิวเตอร์",
        "password": "Cs@2025_04",
    },
    {
        "teacher_id": "CS005",
        "name": "อ.กิตติพงษ์ อุดมทรัพย์",
        "department": "สาขาวิทยาการคอมพิวเตอร์",
        "password": "Cs@2025_05",
    },
    {
        "teacher_id": "CS006",
        "name": "ผศ.ดร.มาลินี ชัยวัฒน์",
        "department": "สาขาวิทยาการคอมพิวเตอร์",
        "password": "Cs@2025_06",
    },
]


def seed():
    init_db()
    with get_connection() as conn:
        inserted = 0
        skipped = 0
        for t in TEACHERS_SEED:
            # ตรวจสอบว่า teacher_id นี้มีอยู่แล้วหรือยัง
            existing = conn.execute(
                "SELECT teacher_id FROM teachers WHERE teacher_id = ?", (t["teacher_id"],)
            ).fetchone()

            if existing:
                print(f"  [ข้าม]   {t['teacher_id']} มีอยู่ใน database แล้ว")
                skipped += 1
                continue

            # Hash password ด้วย bcrypt (cost factor 12)
            password_hash = bcrypt.hashpw(
                t["password"].encode("utf-8"), bcrypt.gensalt(rounds=12)
            ).decode("utf-8")

            conn.execute(
                "INSERT INTO teachers (teacher_id, name, department, password_hash) VALUES (?, ?, ?, ?)",
                (t["teacher_id"], t["name"], t["department"], password_hash),
            )
            print(f"  [สร้าง]  {t['teacher_id']} — {t['name']}")
            inserted += 1

        conn.commit()

    print(f"\nเสร็จสิ้น: สร้างใหม่ {inserted} account | ข้าม {skipped} account")
    print("\n=== รายการ Account อาจารย์ (ชั่วคราว) ===")
    print(f"{'รหัสอาจารย์':<10} {'ชื่อ-นามสกุล':<30} {'รหัสผ่าน'}")
    print("-" * 65)
    for t in TEACHERS_SEED:
        print(f"{t['teacher_id']:<10} {t['name']:<30} {t['password']}")
    print("\n⚠️  กรุณาเปลี่ยนรหัสผ่านหลังจากเข้าระบบครั้งแรก")


if __name__ == "__main__":
    seed()
