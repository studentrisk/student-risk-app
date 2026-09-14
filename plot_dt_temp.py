#gdrivefile = 'https://drive.google.com/file/d/1wMGZZLn972iV4gGiYNOwbMGEbouKawXd/view?usp=sharing'
filecsv = 'https://docs.google.com/uc?id='+'1wMGZZLn972iV4gGiYNOwbMGEbouKawXd'
import pandas as pd
import numpy as np

df = pd.read_csv(filecsv)
gpa_cols = sorted([c for c in df.columns if c.startswith('GPA')], key=lambda x: x.split(' ')[1])

def extract_features(row):
    valid_periods = []
    for col in gpa_cols:
        val = row[col]
        if pd.notna(val) and val != 0:
            valid_periods.append((col.split(' ')[1], float(val)))
    if not valid_periods: return float('nan'), float('nan')

    all_years = sorted(list(set([p[0].split('/')[0] for p in valid_periods])))
    year_map = {year: i+1 for i, year in enumerate(all_years)}

    last_p, last_g = valid_periods[-1]
    year_str, term_str = last_p.split('/')
    rel_year = float(year_map[year_str])
    if term_str == '2': rel_year += 0.5
    return last_g, rel_year

res = df.apply(extract_features, axis=1)
df['GPA ปัจจุบัน'] = res.apply(lambda x: x[0])
df['ปี/เทอม'] = res.apply(lambda x: x[1])
df = df.drop(columns=gpa_cols)
df.isnull().sum()
# Drop missing values สำหรับ 6 features ใหม่
df = df.dropna(subset=['คะแนนเฉลี่ยก่อนรับเข้า', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'GPA ปัจจุบัน', 'ปี/เทอม'])
df.isnull().sum()
df[df.isnull().any(axis=1)]
df_completed = df[df['สถานภาพ'] == 'สำเร็จการศึกษา']
df_status_only = df[['สถานภาพ']]
df_status_only.head()
df[df.apply(lambda row: row.astype(str).str.contains("พ้นสภาพนักศึกษา").any(), axis=1)]
df_filtered_status = df[df['สถานภาพ'].isin(['พ้นสภาพนักศึกษา', 'สำเร็จการศึกษา'])]
df_filtered_status
from sklearn.base import BaseEstimator, TransformerMixin

class StudentRiskEncoder(BaseEstimator, TransformerMixin):
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
            # วิธีรับเข้า: encoding ใหม่ (4 กลุ่ม)
            #   0 = โควตา        — มีคำว่า 'โควตา' (ทุกประเภท รวม TCAS ที่มีโควตา)
            #   1 = สอบคัดเลือก  — มีคำว่า 'สอบคัดเลือก' (ไม่ว่าประเภทใด)
            #   2 = TCAS         — มีคำว่า 'tcas' แต่ไม่ใช่โควตา (Portfolio, รับตรง, เครือข่าย)
            #   3 = อื่นๆ        — ไม่ตรงกับข้างต้น
            adm_s = str(row.get('วิธีรับเข้า', ''))
            if 'โควตา' in adm_s:
                adm = 0                        # โควตา (ทุกประเภท รวม TCAS โควตา)
            elif 'สอบคัดเลือก' in adm_s:
                adm = 1                        # สอบคัดเลือก
            elif 'tcas' in adm_s.lower():
                adm = 2                        # TCAS (Portfolio / รับตรง / เครือข่าย)
            else:
                adm = 3                        # อื่นๆ
            deg  = self.DEG_MAP.get(str(row.get('วุฒิ', '')), 0)
            size = self.school_lookup.get(str(row.get('จบการศึกษาจาก', '')), -1)
            gpa_pre  = float(row.get('คะแนนเฉลี่ยก่อนรับเข้า', 0))
            gpa_curr = float(row.get('GPA ปัจจุบัน', 0))
            period   = float(row.get('ปี/เทอม', 1.0))
            result.append([gpa_pre, adm, deg, size, gpa_curr, period])
        return np.array(result, dtype=float)


class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    ADM_MAP = {'โควตา': 0, 'สอบคัดเลือก': 1}
    # ปรับปรุงให้รองรับ 6 columns
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
            # วิธีรับเข้า: encoding ใหม่ (4 กลุ่ม)
            #   0 = โควตา        — มีคำว่า 'โควตา' (ทุกประเภท รวม TCAS ที่มีโควตา)
            #   1 = สอบคัดเลือก  — มีคำว่า 'สอบคัดเลือก' (ไม่ว่าประเภทใด)
            #   2 = TCAS         — มีคำว่า 'tcas' แต่ไม่ใช่โควตา (Portfolio, รับตรง, เครือข่าย)
            #   3 = อื่นๆ        — ไม่ตรงกับข้างต้น
            adm_s = str(row.get('วิธีรับเข้า', ''))
            if 'โควตา' in adm_s:
                adm = 0                        # โควตา (ทุกประเภท รวม TCAS โควตา)
            elif 'สอบคัดเลือก' in adm_s:
                adm = 1                        # สอบคัดเลือก
            elif 'tcas' in adm_s.lower():
                adm = 2                        # TCAS (Portfolio / รับตรง / เครือข่าย)
            else:
                adm = 3                        # อื่นๆ
            deg  = self.DEG_MAP.get(str(row.get('วุฒิ', '')), 0)
            size = self.school_lookup.get(str(row.get('จบการศึกษาจาก', '')), -1)
            gpa_pre  = float(row.get('คะแนนเฉลี่ยก่อนรับเข้า', 0))
            gpa_curr = float(row.get('GPA ปัจจุบัน', 0))
            period   = float(row.get('ปี/เทอม', 1.0))
            result.append([gpa_pre, adm, deg, size, gpa_curr, period])
        return np.array(result, dtype=float)


status_mapping = {
    'พ้นสภาพนักศึกษา': 0,
    'สำเร็จการศึกษา': 1
}
df_filtered_status['สถานภาพ_encoded'] = df_filtered_status['สถานภาพ'].map(status_mapping)
# Encoding 'วิธีรับเข้า' column using label mapping
if 'วิธีรับเข้า' in df_filtered_status.columns:
    unique_methods = df_filtered_status['วิธีรับเข้า'].dropna().unique()
    method_mapping = {method: i for i, method in enumerate(unique_methods)}
    df_filtered_status['วิธีรับเข้า_encoded'] = df_filtered_status['วิธีรับเข้า'].map(method_mapping)

# Encoding 'วุฒิ' column with specified grouping
if 'วุฒิ' in df_filtered_status.columns:
    degree_custom_mapping = {
        'ปวช.': 0,
        'มัธยมศึกษาตอนปลาย (ม.6)': 0,
        'ปวส.': 1
    }

    unique_degrees_not_mapped = [
        deg for deg in df_filtered_status['วุฒิ'].dropna().unique()
        if deg not in degree_custom_mapping
    ]

    next_label = 2
    for deg in unique_degrees_not_mapped:
        degree_custom_mapping[deg] = next_label
        next_label += 1

    df_filtered_status['วุฒิ_encoded'] = df_filtered_status['วุฒิ'].map(degree_custom_mapping)

df_filtered_status
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_filtered_status['คะแนนเฉลี่ยก่อนรับเข้า_normalized'] = scaler.fit_transform(
    df_filtered_status[['คะแนนเฉลี่ยก่อนรับเข้า']]
)

# Save scaler ไว้ใช้บนเว็บ
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

with open('data_school/COLLEGE_DATA.json', 'r', encoding='utf-8') as f:
    COLLEGE_DATA = json.load(f)

with open('data_school/OBEC_SCHOOL_007.json', 'r', encoding='utf-8') as f:
    OBEC_SCHOOL_007 = json.load(f)

def get_college_encoded(total_students):
    if total_students < 5000:    return 0
    elif total_students < 12000: return 1
    elif total_students < 30000: return 2
    else:                        return 3

OBEC_SIZE_MAP = {'เล็ก': 0, 'กลาง': 1, 'ใหญ่': 2, 'ใหญ่พิเศษ': 3}

school_size_lookup = {}
for item in OBEC_SCHOOL_007:
    name = item.get('schoolName', '')
    size_raw = item.get('schoolSize', '')
    encoded = -1
    for key in OBEC_SIZE_MAP:
        if key in size_raw:
            encoded = OBEC_SIZE_MAP[key]
            break
    school_size_lookup[name] = encoded

college_records = COLLEGE_DATA.get('records', [])
college_student_sums = {}
for record in college_records:
    if isinstance(record, list) and len(record) > 9:
        name = record[3]
        try:    students = int(record[9])
        except: students = 0
        college_student_sums[name] = college_student_sums.get(name, 0) + students

for name, total in college_student_sums.items():
    if name not in school_size_lookup:
        school_size_lookup[name] = get_college_encoded(total)

class StudentRiskEncoder(BaseEstimator, TransformerMixin):
    # วิธีรับเข้า encoding (ปรับปรุงใหม่):
    #   0 = โควตา        — มีคำว่า 'โควตา' (ทุกประเภท รวม TCAS ที่มีโควตา)
    #   1 = สอบคัดเลือก  — มีคำว่า 'สอบคัดเลือก' (ไม่ว่าประเภทใด)
    #   2 = TCAS         — มีคำว่า 'tcas' แต่ไม่ใช่โควตา (Portfolio, รับตรง, เครือข่าย)
    #   3 = อื่นๆ        — ไม่ตรงกับข้างต้นเลย
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
            # วิธีรับเข้า: encoding ใหม่ (4 กลุ่ม)
            #   0 = โควตา        — มีคำว่า 'โควตา' (ทุกประเภท รวม TCAS ที่มีโควตา)
            #   1 = สอบคัดเลือก  — มีคำว่า 'สอบคัดเลือก' (ไม่ว่าประเภทใด)
            #   2 = TCAS         — มีคำว่า 'tcas' แต่ไม่ใช่โควตา (Portfolio, รับตรง, เครือข่าย)
            #   3 = อื่นๆ        — ไม่ตรงกับข้างต้น
            adm_s = str(row.get('วิธีรับเข้า', ''))
            if 'โควตา' in adm_s:
                adm = 0                        # โควตา (ทุกประเภท รวม TCAS โควตา)
            elif 'สอบคัดเลือก' in adm_s:
                adm = 1                        # สอบคัดเลือก
            elif 'tcas' in adm_s.lower():
                adm = 2                        # TCAS (Portfolio / รับตรง / เครือข่าย)
            else:
                adm = 3                        # อื่นๆ
            deg  = self.DEG_MAP.get(str(row.get('วุฒิ', '')), 0)
            size = self.school_lookup.get(str(row.get('จบการศึกษาจาก', '')), -1)
            gpa_pre  = float(row.get('คะแนนเฉลี่ยก่อนรับเข้า', 0))
            gpa_curr = float(row.get('GPA ปัจจุบัน', 0))
            period   = float(row.get('ปี/เทอม', 1.0))
            result.append([gpa_pre, adm, deg, size, gpa_curr, period])
        return np.array(result, dtype=float)



columns_to_drop = ['หน่วยกิตที่ลงทะเบียน', 'หน่วยกิตที่ผ่าน', 'หน่วยกิตของหลักสูตร', 'คะแนนเฉลี่ยสะสม']
df_filtered_status = df_filtered_status.drop(columns=columns_to_drop, errors='ignore')
# ตรวจสอบข้อมูลหลังการจัดกลุ่ม feature ใหม่
import pandas as pd

# 1. Restore original columns from 'df'
original_cols = ['สถานภาพ', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก']
for col in original_cols:
    if col not in df_filtered_status.columns:
        df_filtered_status[col] = df.loc[df_filtered_status.index, col]

# 2. Custom Admission Method Encoding
def encode_admission_method_custom(method):
    if pd.isna(method):
        return 2
    method_lower = str(method).lower()
    if 'โควตา' in method_lower:
        return 0
    elif 'สอบคัดเลือก' in method_lower:
        return 1
    else:
        return 2

df_filtered_status['วิธีรับเข้า_custom_encoded'] = df_filtered_status['วิธีรับเข้า'].apply(encode_admission_method_custom)

# 3. ใช้ school_size_lookup จาก Block 11 แทน results เดิม
df_filtered_status['ขนาดโรงเรียน_encoded'] = (
    df_filtered_status['จบการศึกษาจาก']
    .map(lambda name: school_size_lookup.get(name, -1))
)

# 4. Clean up: Drop temporary or original string columns
final_drop = ['สถานภาพ', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'ขนาดโรงเรียน', 'ประเภทโรงเรียน']
df_filtered_status = df_filtered_status.drop(columns=final_drop, errors='ignore')

columns_to_remove_originals = ['สถานภาพ', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก']
df_filtered_status = df_filtered_status.drop(columns=columns_to_remove_originals, errors='ignore')

if 'วิธีรับเข้า_encoded' in df_filtered_status.columns:
    df_filtered_status = df_filtered_status.drop(columns=['วิธีรับเข้า_encoded'])

columns = df_filtered_status.columns.tolist()

# Find the index of 'สถานภาพ_encoded'
status_encoded_index = columns.index('สถานภาพ_encoded')

# Remove 'วิธีรับเข้า_custom_encoded' from its current position if it exists
if 'วิธีรับเข้า_custom_encoded' in columns:
    columns.remove('วิธีรับเข้า_custom_encoded')

# Insert 'วิธีรับเข้า_custom_encoded' after 'สถานภาพ_encoded'
columns.insert(status_encoded_index + 1, 'วิธีรับเข้า_custom_encoded')

df_filtered_status = df_filtered_status[columns]

# แสดงผลข้อมูลที่ใช้ 6 features (แก้ไขให้ดึงเฉพาะ column ที่มีอยู่จริงใน DataFrame ล่าสุด)
valid_cols = [c for c in ['id', 'คะแนนเฉลี่ยก่อนรับเข้า', 'GPA ปัจจุบัน', 'ปี/เทอม', 'วิธีรับเข้า_encoded', 'วุฒิ_encoded'] if c in df_filtered_status.columns]
df_filtered_status.isnull().sum()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

# เตรียมข้อมูลโดยใช้ 6 features แทนที่เกรดปีแรกเทอมแรก
df_raw_lr = df[df['สถานภาพ'].isin(['พ้นสภาพนักศึกษา', 'สำเร็จการศึกษา'])].copy()
df_raw_lr = df_raw_lr.dropna(subset=['คะแนนเฉลี่ยก่อนรับเข้า', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'GPA ปัจจุบัน', 'ปี/เทอม'])
y_lr = df_raw_lr['สถานภาพ'].map({'พ้นสภาพนักศึกษา': 0, 'สำเร็จการศึกษา': 1})

features_lr = ['คะแนนเฉลี่ยก่อนรับเข้า', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'GPA ปัจจุบัน', 'ปี/เทอม']
X_lr = df_raw_lr[features_lr]

pipeline_lr = ImbPipeline([
    ('encoder', StudentRiskEncoder(school_lookup=school_size_lookup)),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(random_state=42, max_iter=1000))
])

scores = cross_val_score(pipeline_lr, X_lr, y_lr, cv=5)
from sklearn.model_selection import train_test_split

# กรองข้อมูลเฉพาะสถานภาพที่ต้องการ
df_model = df[df['สถานภาพ'].isin(['พ้นสภาพนักศึกษา', 'สำเร็จการศึกษา'])].copy()
df_model = df_model.dropna(subset=['คะแนนเฉลี่ยก่อนรับเข้า', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'GPA ปัจจุบัน', 'ปี/เทอม'])

features = ['คะแนนเฉลี่ยก่อนรับเข้า', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'GPA ปัจจุบัน', 'ปี/เทอม']
X = df_model[features]
y = df_model['สถานภาพ'].map({'พ้นสภาพนักศึกษา': 0, 'สำเร็จการศึกษา': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


df_raw = df[df['สถานภาพ'].isin(['พ้นสภาพนักศึกษา', 'สำเร็จการศึกษา'])].copy()
features_list = ['คะแนนเฉลี่ยก่อนรับเข้า', 'วิธีรับเข้า', 'วุฒิ', 'จบการศึกษาจาก', 'GPA ปัจจุบัน', 'ปี/เทอม']
df_raw = df_raw.dropna(subset=features_list)

X = df_raw[features_list]
y = df_raw['สถานภาพ'].map({'พ้นสภาพนักศึกษา': 0, 'สำเร็จการศึกษา': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import urllib.request
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from matplotlib import font_manager

# ==========================================
# Thai Font — ใช้ urllib แทน !wget (รองรับ Windows)
# ==========================================
def setup_thai_font():
    font_path = 'Sarabun-Regular.ttf'
    if not os.path.exists(font_path):
        try:
            url = 'https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf'
            print("⏬ กำลังดาวน์โหลดฟอนต์ Sarabun...")
            urllib.request.urlretrieve(url, font_path)
            print("✅ ดาวน์โหลดสำเร็จ")
        except Exception as e:
            print(f"⚠️ ดาวน์โหลดไม่ได้: {e}")
            font_path = None

    if font_path and os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Sarabun'
        print("✅ ตั้งค่าฟอนต์ Sarabun เรียบร้อย")
    else:
        # ลองใช้ฟอนต์ไทยที่มีบน Windows
        win_font = 'C:/Windows/Fonts/THSarabunNew.ttf'
        if os.path.exists(win_font):
            font_manager.fontManager.addfont(win_font)
            prop = font_manager.FontProperties(fname=win_font)
            plt.rcParams['font.family'] = prop.get_name()
            print("✅ ใช้ฟอนต์ THSarabunNew จาก Windows")
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("⚠️ ใช้ฟอนต์ DejaVu Sans (ภาษาไทยอาจแสดงเป็น □)")

setup_thai_font()

# ==========================================
# Normal Pipeline (ไม่มี SMOTE, ไม่มี PCA)
# ==========================================
def make_normal_pipeline(classifier):
    return Pipeline([
        ('encoder', StudentRiskEncoder(school_lookup=school_size_lookup)),
        ('scaler',  StandardScaler()),
        ('model',   classifier)
    ])

models_to_run = {
    'Decision Tree':       make_normal_pipeline(DecisionTreeClassifier(random_state=42)),
    'Random Forest':       make_normal_pipeline(RandomForestClassifier(random_state=42)),
    'Gradient Boosting':   make_normal_pipeline(GradientBoostingClassifier(random_state=42)),
}

def evaluate_normal_model(name, pipeline):
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"--- {name} ---")
    print(f"5-Fold CV Accuracy (Mean): {cv_scores.mean():.4f}")
    print(f"Test Accuracy:             {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy:         {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:                 {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:                    {recall_score(y_test, y_pred):.4f}")
    print(f"AUC:                       {roc_auc_score(y_test, y_prob):.4f}\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['พ้นสภาพ', 'สำเร็จ'],
                     yticklabels=['พ้นสภาพ', 'สำเร็จ'])
    cbar = ax.collections[0].colorbar
    cbar.set_label('จำนวนคน', rotation=270, labelpad=15)
    plt.title(f'Confusion Matrix (Normal)\n{name}', fontsize=14)
    plt.ylabel('ค่าจริง')
    plt.xlabel('ค่าทำนาย')
    plt.tight_layout()
    plt.show()

for name, pipeline in models_to_run.items():
    evaluate_normal_model(name, pipeline)


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

dt_pipeline = models_to_run['Decision Tree']
dt_pipeline.fit(X_train, y_train)
dt_model = dt_pipeline.named_steps['model']

plt.figure(figsize=(24, 12), dpi=300)
plot_tree(dt_model, 
          feature_names=['GPA ก่อนเข้า', 'วิธีรับเข้า', 'วุฒิ', 'ขนาด ร.ร.', 'GPA ปัจจุบัน', 'ปี/เทอม'],
          class_names=['พ้นสภาพนักศึกษา', 'สำเร็จการศึกษา'],
          filled=True, 
          rounded=True, 
          max_depth=3,
          fontsize=10)
plt.title('Decision Tree (Normal) - 6 Features (Max Depth 3 displayed)', fontsize=18)
plt.savefig('decision_tree_v7.png', bbox_inches='tight')
print('SUCCESS: Saved decision_tree_v7.png')
