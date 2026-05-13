# ==============================================================================
# model_loader.py — โหลดโมเดล AI และทำการพยากรณ์ความเสี่ยง
# ------------------------------------------------------------------------------
# หน้าที่ : 1. โหลดไฟล์ .pkl (โมเดล ML ที่ train มาแล้ว) เข้าสู่หน่วยความจำ
#           2. มี Encoder แปลงข้อมูลดิบเป็นตัวเลขก่อนส่งเข้าโมเดล
#           3. ฟังก์ชันพยากรณ์แบบปกติ และแบบ Perturbation Smoothing
#           4. ฟังก์ชันดึงรายชื่อโรงเรียนทั้งหมดจาก pipeline
# ทำงานร่วมกับ :
#   - main.py  → import predict_risk_with_perturbation, get_school_list
#   - models/SMOTE_Gradient_Boosting_SMOTE_5_Feature.pkl → ไฟล์โมเดลที่ train ไว้
# ==============================================================================

import pickle   # ใช้โหลดไฟล์ .pkl (Python object ที่บันทึกไว้บน disk)
import sys      # ใช้ลงทะเบียน class ลงใน sys.modules เพื่อให้ pickle โหลดได้
import numpy as np   # NumPy: คำนวณ array, noise, mean, std
import pandas as pd  # Pandas: จัดการข้อมูลในรูปแบบตาราง (DataFrame)
from sklearn.base import BaseEstimator, TransformerMixin  # Base class ของ scikit-learn สำหรับสร้าง Custom Transformer


# ==============================================================================
# CLASS: StudentRiskEncoder — แปลงข้อมูลดิบเป็นตัวเลขก่อนส่งเข้าโมเดล
# ------------------------------------------------------------------------------
# โมเดล ML ต้องรับข้อมูลเป็นตัวเลขเท่านั้น
# Encoder นี้จะแปลงข้อความ เช่น "โควตา" → 0, "ปวส." → 1 ฯลฯ
# ถูกฝังอยู่ใน pipeline ของโมเดล (step แรก ก่อน Gradient Boosting)
# ==============================================================================
class StudentRiskEncoder(BaseEstimator, TransformerMixin):

    # ADM_MAP : ตาราง mapping วิธีรับเข้า (ข้อความ) → ตัวเลข
    #   "โควตา"       → 0
    #   "สอบคัดเลือก" → 1
    #   (ค่าอื่น ๆ ที่ไม่รู้จัก) → 2 (default ใน .get())
    ADM_MAP = {"โควตา": 0, "สอบคัดเลือก": 1}

    # DEG_MAP : ตาราง mapping วุฒิการศึกษา (ข้อความ) → ตัวเลข
    #   "ปวช." และ "ม.6" → 0  (ระดับเดียวกันในโมเดล)
    #   "ปวส."           → 1
    DEG_MAP = {"ปวช.": 0, "มัธยมศึกษาตอนปลาย (ม.6)": 0, "ปวส.": 1}

    def __init__(self, school_lookup=None):
        """
        school_lookup : dict ที่ map ชื่อโรงเรียน → ขนาด (จำนวนนักศึกษา) 
                        โหลดมาจากข้อมูลที่ใช้ train โมเดล
                        ถ้าไม่ส่งมา จะใช้ dict เปล่า {}
        """
        self.school_lookup = school_lookup or {}

    def fit(self, X, y=None):
        """ต้องมีไว้เพื่อเป็น scikit-learn Transformer แต่ไม่ทำอะไร (เพราะ mapping คงที่)"""
        return self

    def transform(self, X):
        """
        แปลงข้อมูล input เป็น array ตัวเลข 5 คอลัมน์:
          [gpa, adm, deg, size, grade_y1s1]
        
        Parameters:
          X : DataFrame หรือ list of dict ที่มี column ชื่อภาษาไทย
        
        Returns:
          numpy array shape (n_samples, 5) dtype float
          → ส่งต่อให้ Gradient Boosting classifier ทำการพยากรณ์
        """
        if isinstance(X, pd.DataFrame):
            rows = X.to_dict(orient="records")   # แปลง DataFrame เป็น list of dict
        else:
            rows = X                              # ถ้าเป็น list of dict อยู่แล้ว ใช้ตรงๆ
        result = []
        for row in rows:
            # adm  : วิธีรับเข้า แปลงข้อความ → ตัวเลข (0,1,2)
            adm = self.ADM_MAP.get(str(row.get("วิธีรับเข้า", "")), 2)
            # deg  : วุฒิการศึกษา แปลงข้อความ → ตัวเลข (0,1)
            deg = self.DEG_MAP.get(str(row.get("วุฒิ", "")), 0)
            # size : ขนาดโรงเรียน (จำนวนนักศึกษา) ค้นจาก school_lookup
            #        ถ้าไม่พบโรงเรียน → -1 (โมเดลตีความว่าเป็นสถาบันเอกชนขนาดเล็ก)
            size = self.school_lookup.get(str(row.get("จบการศึกษาจาก", "")), -1)
            # gpa  : คะแนนเฉลี่ยก่อนรับเข้า (ตัวเลข 0.00–4.00)
            gpa = float(row.get("คะแนนเฉลี่ยก่อนรับเข้า", 0))
            # grade_y1s1 : เกรดเฉลี่ยปีที่ 1 เทอมที่ 1 (0.00–4.00) — feature ที่ 5
            grade_y1s1 = float(row.get("เกรดปีแรกเทอมแรก", 0))
            result.append([gpa, adm, deg, size, grade_y1s1])
        return np.array(result, dtype=float)   # คืน numpy array พร้อมส่งเข้าโมเดล


# ──────────────────────────────────────────────────────────────────────────────
# ลงทะเบียน Class ใน sys.modules["__main__"]
# เหตุผล : ไฟล์ .pkl ถูก save ตอน train โดยอ้างชื่อ class จาก __main__
#          พอมาโหลดใน module อื่น (model_loader.py) Python จะหา class ไม่เจอ
#          จึงต้องลงทะเบียนไว้ก่อน เพื่อให้ pickle.load() ทำงานได้
# ──────────────────────────────────────────────────────────────────────────────
sys.modules["__main__"].StudentRiskEncoder = StudentRiskEncoder

# โหลด pipeline จากไฟล์ .pkl
# pipeline : scikit-learn Pipeline ที่ประกอบด้วย 2 ขั้นตอน:
#   1. "encoder" → StudentRiskEncoder (แปลง input เป็นตัวเลข)
#   2. "classifier" → Gradient Boosting model (พยากรณ์ความเสี่ยง)
# ไฟล์นี้ train มาจาก SMOTE (Synthetic Minority Over-sampling Technique)
# เพื่อแก้ปัญหาข้อมูลไม่สมดุล (นักศึกษาพ้นสภาพ vs ไม่พ้น)
with open("models/SMOTE_Gradient_Boosting_SMOTE_5_Feature.pkl", "rb") as f:
    pipeline = pickle.load(f)   # pipeline: object โมเดลที่พร้อมใช้งาน


# ==============================================================================
# FUNCTION: predict_risk — พยากรณ์แบบปกติ (ไม่มี Perturbation)
# ------------------------------------------------------------------------------
# เก็บไว้เพื่อ backward compatibility (ใช้งานได้แต่ไม่ได้ถูกเรียกจาก main.py)
# ==============================================================================
def predict_risk(gpa: float, admission: str, degree: str, school: str, grade_y1s1: float = 0.0) -> dict:
    """ทำนายแบบเดิม (ไม่มี perturbation) — เก็บไว้เพื่อ backward-compat"""
    # สร้าง DataFrame 1 แถวจาก input → ส่งให้ pipeline แปลงและพยากรณ์
    X = pd.DataFrame([{
        "คะแนนเฉลี่ยก่อนรับเข้า": gpa,
        "วิธีรับเข้า": admission,
        "วุฒิ": degree,
        "จบการศึกษาจาก": school,
        "เกรดปีแรกเทอมแรก": grade_y1s1,
    }])
    # predict_proba คืน [[p_risk, p_safe]] → prob[0] = p_risk, prob[1] = p_safe
    prob = pipeline.predict_proba(X)[0]
    # predict คืน label (0 = พ้นสภาพ, 1 = ปลอดภัย)
    result = pipeline.predict(X)[0]
    return {
        "risk_percent": round(float(prob[0]) * 100, 1),     # ความเสี่ยง (%)
        "success_percent": round(float(prob[1]) * 100, 1),  # โอกาสสำเร็จ (%)
        "label": "เสี่ยงพ้นสภาพ" if result == 0 else "ปลอดภัย"
    }


# ==============================================================================
# FUNCTION: predict_risk_with_perturbation — พยากรณ์แบบ Perturbation Smoothing
# ------------------------------------------------------------------------------
# ทำงานร่วมกับ: main.py → เรียกจาก route POST /predict
#               index.html → ผล dict ถูกส่งเป็น context "result"
# ==============================================================================
def predict_risk_with_perturbation(
    gpa: float,
    admission: str,
    degree: str,
    school: str,
    grade_y1s1: float = 0.0,
    n_perturbations: int = 30,       # จำนวนตัวอย่างที่เพิ่ม noise (30 รอบ + 1 ต้นฉบับ = 31 การทำนาย)
    gpa_noise_std: float = 0.05,     # ค่าเบี่ยงเบนมาตรฐาน (std) ของ noise ที่ GPA เช่น 0.05 = ±0.05
) -> dict:
    """
    Perturbation-based Smoothing
    ─────────────────────────────
    แนวคิด: แทนที่จะทำนาย 1 ครั้ง ให้ทำนาย 31 ครั้ง
            โดยแต่ละครั้งบวก noise เล็กน้อยที่ GPA
            แล้วเฉลี่ยผลลัพธ์ → ลดการแกว่งของผล (stable result)

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
    # rng : Random Number Generator ที่กำหนด seed=42 ไว้
    #       seed ทำให้ผลลัพธ์คงที่ทุกครั้งที่รันด้วย input เดิม (reproducible)
    rng = np.random.default_rng(seed=42)

    # สร้าง GPA values: 1 ต้นฉบับ + n_perturbations ที่มี noise
    # noise : array shape (30,) → ค่า noise แบบ Gaussian (종อยู่รอบ 0, std=0.05)
    noise = rng.normal(0.0, gpa_noise_std, size=n_perturbations)
    # gpa_values : รวม GPA ต้นฉบับ + 30 ค่าที่บวก noise
    #              np.clip ตัดค่าให้อยู่ในช่วง 0.0–4.00 เสมอ
    gpa_values = np.clip(
        np.concatenate([[gpa], gpa + noise]),
        0.0, 4.0
    )  # shape: (n_perturbations + 1,) = (31,)

    # สร้าง DataFrame 31 แถว — แต่ละแถวมี GPA ต่างกัน ส่วน feature อื่นเหมือนกัน
    rows = [
        {
            "คะแนนเฉลี่ยก่อนรับเข้า": float(g),   # GPA ที่แตกต่างกันในแต่ละรอบ
            "วิธีรับเข้า": admission,
            "วุฒิ": degree,
            "จบการศึกษาจาก": school,
            "เกรดปีแรกเทอมแรก": grade_y1s1,
        }
        for g in gpa_values   # วนลูป 31 ครั้ง
    ]
    X_batch = pd.DataFrame(rows)   # X_batch: DataFrame 31 แถว × 5 คอลัมน์

    # ── Batch predict ─────────────────────────────────────────────────────────
    # probs : array shape (31, 2) 
    #         คอลัมน์ 0 = ความน่าจะเป็นพ้นสภาพ
    #         คอลัมน์ 1 = ความน่าจะเป็นสำเร็จ
    probs = pipeline.predict_proba(X_batch)  # shape: (n+1, 2)

    # ── เฉลี่ยความน่าจะเป็นข้ามทุก sample ───────────────────────────────────
    # mean_probs : เฉลี่ยแนวแกน axis=0 (เฉลี่ยข้ามทั้ง 31 แถว)
    mean_probs = probs.mean(axis=0)          # [p_risk, p_safe]
    risk_mean  = float(mean_probs[0])        # ค่าเฉลี่ยความน่าจะเป็นพ้นสภาพ (0.0–1.0)
    safe_mean  = float(mean_probs[1])        # ค่าเฉลี่ยความน่าจะเป็นปลอดภัย (0.0–1.0)

    # ── Stability: วัดความเสถียรของผล ────────────────────────────────────────
    # perturb_std : ค่าเบี่ยงเบนมาตรฐานของ risk probability ข้าม 31 ตัวอย่าง
    #               ยิ่งน้อย = ผลเสถียร, ยิ่งมาก = ผลแกว่ง (ไม่แน่นอน)
    perturb_std = float(probs[:, 0].std())   # ยิ่งน้อย ยิ่งเสถียร

    # ── Confidence: ระยะห่างจาก Decision Boundary ────────────────────────────
    # gap : ระยะห่างระหว่าง risk_mean กับค่า 0.5 (จุดตัดสินใจ)
    #       เช่น risk_mean=0.8 → gap=0.3 (มั่นใจสูง)
    #            risk_mean=0.55 → gap=0.05 (มั่นใจต่ำ — ใกล้ boundary)
    gap = abs(risk_mean - 0.5)
    if gap >= 0.25:
        confidence = "สูง"        # ห่างจาก boundary มากกว่า 25% → มั่นใจสูง
        confidence_en = "high"
    elif gap >= 0.10:
        confidence = "ปานกลาง"   # ห่าง 10–25% → มั่นใจปานกลาง
        confidence_en = "medium"
    else:
        confidence = "ต่ำ"       # ห่างน้อยกว่า 10% → มั่นใจต่ำ ควรระวัง
        confidence_en = "low"

    # label : ผลสรุปขั้นสุดท้าย
    #         risk_mean >= 0.5 → พ้นสภาพ, risk_mean < 0.5 → ปลอดภัย
    label = "เสี่ยงพ้นสภาพ" if risk_mean >= 0.5 else "ปลอดภัย"

    # คืน dict ผลลัพธ์ทั้งหมด → ส่งไปแสดงใน index.html ผ่าน context "result"
    return {
        "risk_percent":    round(risk_mean * 100, 1),          # ความเสี่ยง (%) แสดงเป็นตัวใหญ่
        "success_percent": round(safe_mean * 100, 1),          # โอกาสสำเร็จ (%)
        "label":           label,                              # "เสี่ยงพ้นสภาพ" / "ปลอดภัย"
        "confidence":      confidence,                         # ความมั่นใจ (ภาษาไทย)
        "confidence_en":   confidence_en,                      # ความมั่นใจ (ภาษาอังกฤษ) → ใช้เป็น CSS class
        "perturb_std":     round(perturb_std * 100, 1),        # ความเสถียรของผล (%) → แสดงเป็น ±X%
        "n_perturbations": n_perturbations,                    # จำนวนรอบที่ทดสอบ → แสดงใน result card
    }


# ==============================================================================
# FUNCTION: get_school_list — ดึงรายชื่อโรงเรียนทั้งหมดจาก pipeline
# ------------------------------------------------------------------------------
# ทำงานร่วมกับ: main.py → เรียกตอนเริ่มต้น เก็บไว้ใน SCHOOL_LIST
#               index.html → ใช้สร้าง dropdown ค้นหาโรงเรียน (combobox)
# ==============================================================================
def get_school_list() -> list:
    """
    ดึง dict school_lookup จาก Encoder ใน pipeline แล้วคืนรายชื่อโรงเรียนเรียงตามตัวอักษร
    
    pipeline.named_steps["encoder"] : เข้าถึง StudentRiskEncoder ที่อยู่ใน pipeline
    encoder.school_lookup           : dict ชื่อโรงเรียน → ขนาด ที่โหลดมาพร้อม .pkl
    """
    encoder = pipeline.named_steps["encoder"]   # ดึง Encoder จาก pipeline
    schools = sorted(encoder.school_lookup.keys())   # เรียงชื่อโรงเรียนตามตัวอักษร
    return schools   # คืน list of string ส่งไปเก็บใน SCHOOL_LIST ของ main.py