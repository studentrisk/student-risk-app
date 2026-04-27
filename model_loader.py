import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    ADM_MAP = {"โควตา": 0, "สอบคัดเลือก": 1}
    DEG_MAP = {"ปวช.": 0, "มัธยมศึกษาตอนปลาย (ม.6)": 0, "ปวส.": 1}

    def __init__(self, school_lookup=None):
        self.school_lookup = school_lookup or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient="records")
        else:
            rows = X
        result = []
        for row in rows:
            adm = self.ADM_MAP.get(str(row.get("วิธีรับเข้า", "")), 2)
            deg = self.DEG_MAP.get(str(row.get("วุฒิ", "")), 0)
            size = self.school_lookup.get(str(row.get("จบการศึกษาจาก", "")), -1)
            gpa = float(row.get("คะแนนเฉลี่ยก่อนรับเข้า", 0))
            grade_y1s1 = float(row.get("เกรดปีแรกเทอมแรก", 0))
            result.append([gpa, adm, deg, size, grade_y1s1])
        return np.array(result, dtype=float)


sys.modules["__main__"].StudentRiskEncoder = StudentRiskEncoder

with open("models/SMOTE_Gradient_Boosting_SMOTE_5_Feature.pkl", "rb") as f:
    pipeline = pickle.load(f)


def predict_risk(gpa: float, admission: str, degree: str, school: str, grade_y1s1: float = 0.0) -> dict:
    """ทำนายแบบเดิม (ไม่มี perturbation) — เก็บไว้เพื่อ backward-compat"""
    X = pd.DataFrame([{
        "คะแนนเฉลี่ยก่อนรับเข้า": gpa,
        "วิธีรับเข้า": admission,
        "วุฒิ": degree,
        "จบการศึกษาจาก": school,
        "เกรดปีแรกเทอมแรก": grade_y1s1,
    }])
    prob = pipeline.predict_proba(X)[0]
    result = pipeline.predict(X)[0]
    return {
        "risk_percent": round(float(prob[0]) * 100, 1),
        "success_percent": round(float(prob[1]) * 100, 1),
        "label": "เสี่ยงพ้นสภาพ" if result == 0 else "ปลอดภัย"
    }


def predict_risk_with_perturbation(
    gpa: float,
    admission: str,
    degree: str,
    school: str,
    grade_y1s1: float = 0.0,
    n_perturbations: int = 30,
    gpa_noise_std: float = 0.05,
) -> dict:
    """
    Perturbation-based Smoothing
    ─────────────────────────────
    สร้าง n_perturbations ตัวอย่างโดยบวก Gaussian noise เล็กน้อยที่ค่า GPA
    แล้วเฉลี่ยความน่าจะเป็นจากทุกตัวอย่างเพื่อลดการแกว่งของผลลัพธ์

    Parameters
    ----------
    gpa              : ค่า GPA ต้นฉบับ
    admission        : วิธีรับเข้า
    degree           : วุฒิการศึกษา
    school           : โรงเรียน / สถาบัน
    grade_y1s1       : เกรดเฉลี่ยปีแรกเทอมแรก (0.00 – 4.00)
    n_perturbations  : จำนวนตัวอย่าง perturbed (ไม่รวม original)
    gpa_noise_std    : ค่าเบี่ยงเบนมาตรฐานของ noise ที่ GPA
    """
    rng = np.random.default_rng(seed=42)

    # สร้าง GPA values: 1 ต้นฉบับ + n_perturbations ที่มี noise
    noise = rng.normal(0.0, gpa_noise_std, size=n_perturbations)
    gpa_values = np.clip(
        np.concatenate([[gpa], gpa + noise]),
        0.0, 4.0
    )  # shape: (n_perturbations + 1,)

    # สร้าง DataFrame หลายแถว (batch) เพื่อเรียก predict_proba ครั้งเดียว
    rows = [
        {
            "คะแนนเฉลี่ยก่อนรับเข้า": float(g),
            "วิธีรับเข้า": admission,
            "วุฒิ": degree,
            "จบการศึกษาจาก": school,
            "เกรดปีแรกเทอมแรก": grade_y1s1,
        }
        for g in gpa_values
    ]
    X_batch = pd.DataFrame(rows)

    # ── Batch predict ──────────────────────────────────────────────────────
    probs = pipeline.predict_proba(X_batch)  # shape: (n+1, 2)

    # ── เฉลี่ยความน่าจะเป็นข้ามทุก sample ────────────────────────────────
    mean_probs = probs.mean(axis=0)          # [p_risk, p_safe]
    risk_mean  = float(mean_probs[0])
    safe_mean  = float(mean_probs[1])

    # ── Stability: std ของ risk prob ข้ามตัวอย่าง perturbed ──────────────
    perturb_std = float(probs[:, 0].std())   # ยิ่งน้อย ยิ่งเสถียร

    # ── Confidence: ระยะห่างจาก boundary (0.5) ───────────────────────────
    gap = abs(risk_mean - 0.5)
    if gap >= 0.25:
        confidence = "สูง"
        confidence_en = "high"
    elif gap >= 0.10:
        confidence = "ปานกลาง"
        confidence_en = "medium"
    else:
        confidence = "ต่ำ"
        confidence_en = "low"

    label = "เสี่ยงพ้นสภาพ" if risk_mean >= 0.5 else "ปลอดภัย"

    return {
        "risk_percent":    round(risk_mean * 100, 1),
        "success_percent": round(safe_mean * 100, 1),
        "label":           label,
        "confidence":      confidence,
        "confidence_en":   confidence_en,
        "perturb_std":     round(perturb_std * 100, 1),  # หน่วยเป็น %
        "n_perturbations": n_perturbations,
    }


def get_school_list() -> list:
    encoder = pipeline.named_steps["encoder"]
    schools = sorted(encoder.school_lookup.keys())
    return schools