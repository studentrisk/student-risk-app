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
            result.append([gpa, adm, deg, size])
        return np.array(result, dtype=float)


sys.modules["__main__"].StudentRiskEncoder = StudentRiskEncoder

with open("models/SMOTE_Gradient_Boosting_SMOTE.pkl", "rb") as f:
    pipeline = pickle.load(f)


def predict_risk(gpa: float, admission: str, degree: str, school: str) -> dict:
    X = pd.DataFrame([{
        "คะแนนเฉลี่ยก่อนรับเข้า": gpa,
        "วิธีรับเข้า": admission,
        "วุฒิ": degree,
        "จบการศึกษาจาก": school
    }])
    prob = pipeline.predict_proba(X)[0]
    result = pipeline.predict(X)[0]
    return {
        "risk_percent": round(float(prob[0]) * 100, 1),
        "success_percent": round(float(prob[1]) * 100, 1),
        "label": "เสี่ยงพ้นสภาพ" if result == 0 else "ปลอดภัย"
    }


def get_school_list() -> list:
    encoder = pipeline.named_steps["encoder"]
    schools = sorted(encoder.school_lookup.keys())
    return schools