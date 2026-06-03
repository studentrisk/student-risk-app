import dill
import sys
import glob
import os

# จำกัดจำนวน Thread เพื่อไม่ให้ Render Server ค้าง (503 Service Unavailable)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    """
    Stub class — ต้องมีเพื่อให้ dill โหลด pkl ได้
    (transform จริงอยู่ใน pkl แล้ว ชื่อ column จริงคือ 'GPA ปัจจุบัน' และ 'ปี/เทอม')
    """
    ADM_MAP = {"โควตา": 0, "สอบคัดเลือก": 1}
    DEG_MAP = {"ปวช.": 0, "มัธยมศึกษาตอนปลาย (ม.6)": 0, "ปวส.": 1}

    def __init__(self, school_lookup=None):
        self.school_lookup = school_lookup or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# ลงทะเบียน class ให้ dill หา __main__.StudentRiskEncoder ได้ตอน load
sys.modules["__main__"].StudentRiskEncoder = StudentRiskEncoder

# โหลดโมเดล — ใช้ไฟล์ .pkl แรกที่เจอใน folder models/
_pkl_files = sorted(glob.glob(os.path.join("models", "*.pkl")))
if not _pkl_files:
    raise FileNotFoundError("ไม่พบไฟล์ .pkl ใน folder models/ กรุณาวางไฟล์โมเดลไว้ใน folder นั้น")
_model_path = _pkl_files[0]
print(f"[model_loader] โหลดโมเดล: {_model_path}")

with open(_model_path, "rb") as f:
    pipeline = dill.load(f)

# ป้องกันโมเดล (เช่น Random Forest) แตก Thread ไปแย่ง CPU กันเองจนค้าง
if hasattr(pipeline, "steps"):
    model_step = pipeline.steps[-1][1]
    if hasattr(model_step, "n_jobs"):
        model_step.n_jobs = 1

# ──────────────────────────────────────────────────────────────────────────────
# PATCH: inject pd และ np เข้า globals ของ transform จริงใน pkl
# (Colab บันทึก transform โดยอ้างอิง pd/np จาก namespace ที่รัน แต่ไม่ได้ bundle ไว้)
# ──────────────────────────────────────────────────────────────────────────────
try:
    enc = pipeline.named_steps["encoder"]
    _raw_t     = type(enc).__dict__["transform"]
    _inner1    = _raw_t.__closure__[0].cell_contents
    _real_fn   = _inner1.__closure__[0].cell_contents
    _real_fn.__globals__["pd"] = pd
    _real_fn.__globals__["np"] = np
except Exception:
    pass   # ถ้า pkl เวอร์ชันอื่นไม่ต้อง patch ก็ผ่านไป


def predict_risk_with_perturbation(
    gpa: float,
    admission: str,
    degree: str,
    school: str,
    study_year: int = 1,       # ส่งเป็น 'ปี/เทอม' เข้าโมเดล
    gpa_at_year: float = 0.0,  # ส่งเป็น 'GPA ปัจจุบัน' เข้าโมเดล
    n_perturbations: int = 30,
    gpa_noise_std: float = 0.05,
) -> dict:
    """
    Perturbation-based Smoothing
    ────────────────────────────
    สร้าง n_perturbations ตัวอย่างโดยบวก Gaussian noise ที่ GPA ก่อนรับเข้า
    แล้วเฉลี่ยความน่าจะเป็นเพื่อลดการแกว่งของผลลัพธ์
    """
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, gpa_noise_std, size=n_perturbations)
    gpa_values = np.clip(np.concatenate([[gpa], gpa + noise]), 0.0, 4.0)

    rows = [
        {
            "คะแนนเฉลี่ยก่อนรับเข้า": float(g),
            "วิธีรับเข้า":             admission,
            "วุฒิ":                    degree,
            "จบการศึกษาจาก":           school,
            "ปี/เทอม":                 study_year,   # ← ชื่อจริงใน pkl
            "GPA ปัจจุบัน":            gpa_at_year,  # ← ชื่อจริงใน pkl
        }
        for g in gpa_values
    ]
    X_batch = pd.DataFrame(rows)

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