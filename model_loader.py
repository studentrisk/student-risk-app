import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# 1. จำกัดจำนวน Thread เพื่อไม่ให้เซิร์ฟเวอร์ค้าง
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# 2. คลาส StudentRiskEncoder — ต้องตรงกับที่ใช้ train ใน ProjectV7.ipynb (Cell 15) ทุกประการ
#
# วิธีรับเข้า encoding (ปรับปรุงใหม่ — ตรวจตามลำดับ substring match):
#   0 = โควตา        — มีคำว่า 'โควตา'       (โควตา-เรียนดี, โควตา-เครือข่าย, TCAS โควตา ฯลฯ)
#   1 = สอบคัดเลือก  — มีคำว่า 'สอบคัดเลือก' (สอบตรง, รับตรง-สอบคัดเลือก ฯลฯ)
#   2 = TCAS         — มีคำว่า 'tcas' แต่ไม่ใช่โควตา (Portfolio, รับตรง, สถานศึกษาเครือข่าย)
#   3 = อื่นๆ        — ไม่ตรงกลุ่มใดข้างต้น (รับตรง ปวช./ปวส. ฯลฯ)
#
# วุฒิ (exact match):
#   0 = ปวช. หรือ ม.6 (ทั้งคู่ถูก encode เป็น 0 ในชุดข้อมูล)
#   1 = ปวส.
class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    DEG_MAP = {'ปวช.': 0, 'มัธยมศึกษาตอนปลาย (ม.6)': 0, 'ปวส.': 1}

    def __init__(self, school_lookup=None):
        self.school_lookup = school_lookup if school_lookup is not None else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient='records')
        else:
            rows = X

        result = []
        for row in rows:
            # วิธีรับเข้า encoding ใหม่: ตรวจตามลำดับ substring match
            # โควตา (0) → สอบคัดเลือก (1) → TCAS ไม่ใช่โควตา (2) → อื่นๆ (3)
            adm_s = str(row.get('วิธีรับเข้า', ''))
            if 'โควตา' in adm_s:
                adm = 0          # โควตา ทุกประเภท รวม TCAS โควตา
            elif 'สอบคัดเลือก' in adm_s:
                adm = 1          # สอบคัดเลือก (ทุกประเภท)
            elif 'tcas' in adm_s.lower():
                adm = 2          # TCAS Portfolio / รับตรง / สถานศึกษาเครือข่าย
            else:
                adm = 3          # อื่นๆ (รับตรง ปวช./ปวส. ฯลฯ)

            deg  = self.DEG_MAP.get(str(row.get('วุฒิ', '')), 0)
            size = self.school_lookup.get(str(row.get('จบการศึกษาจาก', '')), -1)
            gpa_pre  = float(row.get('คะแนนเฉลี่ยก่อนรับเข้า', 0))
            gpa_curr = float(row.get('GPA ปัจจุบัน', 0))
            period   = float(row.get('ปี/เทอม', 1.0))

            result.append([gpa_pre, adm, deg, size, gpa_curr, period])

        # คืนค่าเป็น numpy array เหมือนที่ train มา (shape: n_samples × 6)
        return np.array(result, dtype=float)

# 3. หลอก Pickle ให้มองเห็นคลาสนี้ใน __main__
sys.modules["__main__"].StudentRiskEncoder = StudentRiskEncoder

# 4. โหลดโมเดล
_pkl_files = sorted(glob.glob(os.path.join("models", "*.pkl")))
if not _pkl_files:
    raise FileNotFoundError("ไม่พบไฟล์ .pkl ใน folder models/")
_model_path = _pkl_files[0]
print(f"[model_loader] โหลดโมเดลสำเร็จ: {_model_path}")

with open(_model_path, "rb") as f:
    pipeline = pickle.load(f)

if hasattr(pipeline, "steps"):
    model_step = pipeline.steps[-1][1]
    if hasattr(model_step, "n_jobs"):
        model_step.n_jobs = 1

# 5. ชื่อคอลัมน์จริงที่ใช้เวลาเทรน (ตรึงไว้ที่นี่เพื่อใช้ทั้งไฟล์)
ACTUAL_COLS = [
    'คะแนนเฉลี่ยก่อนรับเข้า',
    'วิธีรับเข้า',
    'วุฒิ',
    'จบการศึกษาจาก',
    'GPA ปัจจุบัน',
    'ปี/เทอม',
]

# 6. ฟังก์ชันประเมินความเสี่ยง
def predict_risk_with_perturbation(
    gpa: float, admission: str, degree: str, school: str,
    study_year: float = 1.0, gpa_at_year: float = 0.0,
    n_perturbations: int = 30, gpa_noise_std: float = 0.05
) -> dict:

    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, gpa_noise_std, size=n_perturbations)
    gpa_values = np.clip(np.concatenate([[gpa], gpa + noise]), 0.0, 4.0)

    # สร้าง rows ด้วย ACTUAL_COLS เพื่อให้ชื่อคอลัมน์ตรงกับที่เทรนมา 100%
    rows = [
        {
            ACTUAL_COLS[0]: float(g),
            ACTUAL_COLS[1]: admission,
            ACTUAL_COLS[2]: degree,
            ACTUAL_COLS[3]: school,
            ACTUAL_COLS[4]: gpa_at_year,
            ACTUAL_COLS[5]: study_year,
        }
        for g in gpa_values
    ]
    X_batch = pd.DataFrame(rows)

    # บังคับ column order ให้ตรงเป๊ะกับที่โมเดลคาดหวัง
    X_batch = X_batch[ACTUAL_COLS]

    # ทายผล
    probs       = pipeline.predict_proba(X_batch)
    mean_probs  = probs.mean(axis=0)
    risk_mean   = float(mean_probs[0])
    safe_mean   = float(mean_probs[1])
    perturb_std = float(probs[:, 0].std())

    gap = abs(risk_mean - 0.5)
    if gap >= 0.25:
        confidence, confidence_en = "สูง", "high"
    elif gap >= 0.10:
        confidence, confidence_en = "ปานกลาง", "medium"
    else:
        confidence, confidence_en = "ต่ำ", "low"

    return {
        "risk_percent":    round(risk_mean * 100, 1),
        "success_percent": round(safe_mean * 100, 1),
        "label":           "เสี่ยงพ้นสภาพ" if risk_mean >= 0.5 else "ปลอดภัย",
        "confidence":      confidence,
        "confidence_en":   confidence_en,
        "perturb_std":     round(perturb_std * 100, 1),
        "n_perturbations": n_perturbations,
    }

def get_school_list() -> list:
    encoder = pipeline.named_steps["encoder"]
    return sorted(encoder.school_lookup.keys())