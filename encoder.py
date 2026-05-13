# ==============================================================================
# encoder.py — Custom Encoder สำรอง (ไม่ได้ถูกเรียกจาก main.py โดยตรง)
# ------------------------------------------------------------------------------
# หน้าที่ : กำหนด class StudentRiskEncoder แบบ standalone
#           ใช้เป็น reference หรือ import ในกรณีที่ต้องการแยก encoder ออกมา
# หมายเหตุ: model_loader.py มี class เดียวกันนี้ฝังอยู่แล้ว
#           ไฟล์นี้เป็นเวอร์ชันดั้งเดิม (4 feature) ก่อนเพิ่ม grade_y1s1
# ทำงานร่วมกับ :
#   - model_loader.py → มี class เวอร์ชันใหม่ (5 feature) แทนที่ใช้งานจริง
# ==============================================================================

from sklearn.base import BaseEstimator, TransformerMixin  # Base class ของ scikit-learn สำหรับสร้าง Custom Transformer
import numpy as np   # NumPy: ใช้แปลง list → array ตัวเลข
import pandas as pd  # Pandas: ใช้รับ DataFrame เป็น input


# ==============================================================================
# CLASS: StudentRiskEncoder — แปลงข้อมูลดิบเป็นตัวเลขก่อนส่งเข้าโมเดล
# ------------------------------------------------------------------------------
# BaseEstimator    : ทำให้ class นี้ใช้ร่วมกับ scikit-learn Pipeline ได้
#                   และมี get_params() / set_params() ให้อัตโนมัติ
# TransformerMixin : เพิ่ม fit_transform() ให้อัตโนมัติ (เรียก fit().transform())
# ==============================================================================
class StudentRiskEncoder(BaseEstimator, TransformerMixin):

    # ADM_MAP : dict mapping วิธีรับเข้า → ตัวเลข
    #   "โควตา"       → 0 (รับตรงโดยไม่ต้องสอบ)
    #   "สอบคัดเลือก" → 1 (ผ่านการสอบแข่งขัน)
    #   ค่าไม่รู้จัก  → 2 (default ใน .get())
    ADM_MAP = {'โควตา': 0, 'สอบคัดเลือก': 1}

    # DEG_MAP : dict mapping วุฒิการศึกษา → ตัวเลข
    #   "ปวช." และ "ม.6" → 0 (ถือว่าระดับเดียวกันในการ train โมเดล)
    #   "ปวส."           → 1 (วุฒิสูงกว่า มีทักษะวิชาชีพมากกว่า)
    DEG_MAP = {'ปวช.': 0, 'มัธยมศึกษาตอนปลาย (ม.6)': 0, 'ปวส.': 1}

    def __init__(self, school_lookup=None):
        """
        school_lookup : dict ชื่อโรงเรียน → ขนาด (จำนวนนักศึกษาในชุดข้อมูล train)
                        ใช้แทนชื่อโรงเรียน (string) ด้วยขนาด (int) เพื่อให้โมเดลรับได้
                        ถ้าไม่ส่ง → ใช้ {} (dict เปล่า)
        """
        self.school_lookup = school_lookup or {}

    def fit(self, X, y=None):
        """
        ไม่ทำอะไร — มีไว้เพราะ scikit-learn Pipeline บังคับให้ Transformer มี fit()
        mapping ทั้งหมดกำหนดไว้ตายตัวใน class แล้ว
        """
        return self

    def transform(self, X):
        """
        แปลงข้อมูล input เป็น numpy array ตัวเลข 4 คอลัมน์:
          [gpa, adm, deg, size]
        
        ⚠️ หมายเหตุ: เวอร์ชันนี้มีแค่ 4 feature (ไม่มี grade_y1s1)
           ใช้กับโมเดลเก่าที่ train ก่อนเพิ่ม feature ที่ 5
           เวอร์ชันที่ใช้งานจริงอยู่ใน model_loader.py (5 feature)
        
        Parameters:
          X : pd.DataFrame หรือ list of dict
        
        Returns:
          np.array shape (n_rows, 4) dtype float
        """
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient='records')   # แปลง DataFrame → list of dict
        else:
            rows = X
        result = []
        for row in rows:
            # adm  : แปลงวิธีรับเข้าจากข้อความ → ตัวเลข
            adm  = self.ADM_MAP.get(str(row.get('วิธีรับเข้า', '')), 2)
            # deg  : แปลงวุฒิการศึกษาจากข้อความ → ตัวเลข
            deg  = self.DEG_MAP.get(str(row.get('วุฒิ', '')), 0)
            # size : ขนาดโรงเรียน ค้นจาก school_lookup
            #        ถ้าโรงเรียนไม่อยู่ในรายการ → -1 (สถาบันนอก dataset)
            size = self.school_lookup.get(str(row.get('จบการศึกษาจาก', '')), -1)
            # gpa  : คะแนนเฉลี่ยก่อนรับเข้า (float 0.00–4.00)
            gpa  = float(row.get('คะแนนเฉลี่ยก่อนรับเข้า', 0))
            result.append([gpa, adm, deg, size])
        return np.array(result, dtype=float)   # คืน array พร้อมส่งเข้าโมเดล