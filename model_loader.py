import os
import sys
import glob
import pickle

# 1. จำกัดจำนวน Thread เพื่อไม่ให้ Render/Hugging Face Server ค้าง
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# 2. คลาส StudentRiskEncoder ฉบับสมบูรณ์ (Pickle จะมาเรียกใช้ตัวนี้เพื่อแปลงค่า)
class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, school_lookup=None):
        self.school_lookup = school_lookup if school_lookup is not None else {}
        # แมปค่าให้ตรงกับ Colab ฉบับ 6 Features
        self.ADM_MAP = {'โควตา': 1, 'สอบคัดเลือก': 2, 'อื่นๆ': 0}
        self.DEG_MAP = {'ปวช.': 1, 'มัธยมศึกษาตอนปลาย (ม.6)': 2, 'ปวส.': 3}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

# 4. โหลดโมเดล — ใช้ไฟล์ .pkl แรกที่เจอใน folder models/
_pkl_files = sorted(glob.glob(os.path.join("models", "*.pkl")))
if not _pkl_files:
    raise FileNotFoundError("ไม่พบไฟล์ .pkl ใน folder models/ กรุณาวางไฟล์โมเดลไว้ใน folder นั้น")
_model_path = _pkl_files[0]
print(f"[model_loader] โหลดโมเดล: {_model_path}")

with open(_model_path, "rb") as f:
    pipeline = pickle.load(f)

# ป้องกันโมเดล (เช่น Random Forest) แตก Thread ไปแย่ง CPU กันเองจนค้าง
if hasattr(pipeline, "steps"):
    model_step = pipeline.steps[-1][1]
    if hasattr(model_step, "n_jobs"):
        model_step.n_jobs = 1

# 5. ฟังก์ชันแพ็กข้อมูลและประเมินผล
def predict_risk_with_perturbation(
    gpa: float,
    admission: str,
    degree: str,
    school: str,
    study_year: int = 1,
    gpa_at_year: float = 0.0,
    n_perturbations: int = 30,
    gpa_noise_std: float = 0.05,
) -> dict:
    
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, gpa_noise_std, size=n_perturbations)
    gpa_values = np.clip(np.concatenate([[gpa], gpa + noise]), 0.0, 4.0)

    # 🎯 ชื่อคอลัมน์ภาษาไทย 6 ตัว ตรงตามที่ AI ถูกสอนมาเป๊ะๆ 100%
    rows = [
        {
            "คะแนนเฉลี่ยก่อนรับเข้า": float(g),
            "วิธีรับเข้า":            admission,
            "วุฒิ":                    degree,
            "จบการศึกษาจาก":           school,
            "GPA ปัจจุบัน":            gpa_at_year,
            "ปี/เทอม":                study_year,
        }
        for g in gpa_values
    ]
    X_batch = pd.DataFrame(rows)

    # ส่งเข้าท่อ Pipeline (มันจะวิ่งผ่าน transform ด้านบน แล้วไปเข้าโมเดลเอง)
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

# 6. ฟังก์ชันดึงรายชื่อโรงเรียน
def get_school_list() -> list:
    encoder = pipeline.named_steps["encoder"]
    return sorted(encoder.school_lookup.keys())