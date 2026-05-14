from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    ADM_MAP = {'โควตา': 0, 'สอบคัดเลือก': 1}
    DEG_MAP = {'ปวช.': 0, 'มัธยมศึกษาตอนปลาย (ม.6)': 0, 'ปวส.': 1}

    def __init__(self, school_lookup=None):
        self.school_lookup = school_lookup or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient='records')
        else:
            rows = X
        result = []
        for row in rows:
            adm  = self.ADM_MAP.get(str(row.get('วิธีรับเข้า', '')), 2)
            deg  = self.DEG_MAP.get(str(row.get('วุฒิ', '')), 0)
            size = self.school_lookup.get(str(row.get('จบการศึกษาจาก', '')), -1)
            gpa  = float(row.get('คะแนนเฉลี่ยก่อนรับเข้า', 0))
            result.append([gpa, adm, deg, size])
        return np.array(result, dtype=float)