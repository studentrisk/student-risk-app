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

# 2. คลาส StudentRiskEncoder ฉบับสมบูรณ์
class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    ADM_MAP = {'โควตา': 1, 'สอบคัดเลือก': 2, 'อื่นๆ': 0}
    DEG_MAP = {'ปวช.': 1, 'มัธยมศึกษาตอนปลาย (ม.6)': 2, 'ปวส.': 3}

    def __init__(self, school_lookup=None):
        self.school_lookup = school_lookup if school_lookup is not None else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("👉 [DEBUG] คอลัมน์ที่ส่งให้ AI คือ:", X.columns.tolist())

        X_encoded = X.copy()

        if 'วิธีรับเข้า' in X_encoded.columns:
            X_encoded['วิธีรับเข้า'] = X_encoded['วิธีรับเข้า'].map(self.ADM_MAP).fillna(0)
        if 'วุฒิ' in X_encoded.columns:
            X_encoded['วุฒิ'] = X_encoded['วุฒิ'].map(self.DEG_MAP).fillna(0)
        if 'จบการศึกษาจาก' in X_encoded.columns:
            X_encoded['จบการศึกษาจาก'] = X_encoded['จบการศึกษาจาก'].map(self.school_lookup).fillna(0)

        return X_encoded

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