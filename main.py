import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
import json
import firebase_admin
from firebase_admin import credentials, auth

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from model_loader import predict_risk_with_perturbation, get_school_list

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# โหลดรายชื่อโรงเรียนครั้งเดียวตอนเริ่มต้น
SCHOOL_LIST = get_school_list()

# ==========================================
# 🔑 โค้ดเช็คกุญแจแบบ 2 ระบบ (Local & Render)
# ==========================================
firebase_cred_json = os.environ.get("FIREBASE_CREDENTIALS")

if firebase_cred_json:
    # 🌐 กรณีรันอยู่บน Render (อ่านจาก Environment)
    print("กำลังใช้งาน Firebase จาก Render Environment")
    cred_dict = json.loads(firebase_cred_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
elif os.path.exists("firebase-key.json"):
    # 💻 กรณีรันอยู่บนเครื่อง Local (อ่านจากไฟล์)
    print("กำลังใช้งาน Firebase จากไฟล์ firebase-key.json (Local)")
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
else:
    print("❌ ERROR: หากุญแจ Firebase ไม่เจอ ทั้งบน Render และ Local!")
# ==========================================

def get_current_user(request: Request):
    """ตรวจสอบว่ามี session cookie และให้ Firebase ยืนยันความถูกต้อง"""
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        # ส่ง Token ให้ Firebase ตรวจสอบ (ป้องกันการปลอมแปลง Cookie)
        decoded_token = auth.verify_id_token(token)
        return decoded_token # คืนค่าข้อมูล User กลับไป (เช่น อีเมล)
    except Exception as e:
        print(f"Token invalid or expired: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    # ถ้า login อยู่แล้ว (และ Token ถูกต้อง) ให้ redirect ไป dashboard
    if get_current_user(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse(request, "login.html")

@app.get("/logout")
async def logout(request: Request):
    # ลบ cookie session และบังคับ redirect ไปหน้า login
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session")
    # ป้องกัน browser cache หน้า logout
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/")
    response = templates.TemplateResponse(request, "index.html", {
        "schools": SCHOOL_LIST,
        "user": user  # ส่งข้อมูล user ไปเผื่อใช้แสดงชื่อ/อีเมลใน index.html
    })
    # ป้องกัน browser เก็บ cache หน้า dashboard ไว้ (ป้องกันกดปุ่มย้อนกลับหลัง logout)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    return response


# แปลง period_label → study_year (ตัวเลข) ที่โมเดลต้องการ
PERIOD_MAP = {
    "ปี 1 เทอม 1": 1, "ปี 1 เทอม 2": 2,
    "ปี 2 เทอม 1": 3, "ปี 2 เทอม 2": 4,
    "ปี 3 เทอม 1": 5, "ปี 3 เทอม 2": 6,
    "ปี 4 เทอม 1": 7, "ปี 4 เทอม 2": 8,
}

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    gpa: float = Form(...),
    admission: str = Form(...),
    degree: str = Form(...),
    school: str = Form(...),
    gpa_at_year: float = Form(...),
    period_label: str = Form(...),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/", status_code=303)

    study_year = PERIOD_MAP.get(period_label, 1)
    result = predict_risk_with_perturbation(
        gpa, admission, degree, school,
        study_year=study_year,
        gpa_at_year=gpa_at_year,
    )
    response = templates.TemplateResponse(request, "index.html", {
        "result": result,
        "schools": SCHOOL_LIST,
        "user": user,
        "form": {
            "gpa": gpa,
            "admission": admission,
            "degree": degree,
            "school": school,
            "gpa_at_year": gpa_at_year,
            "period_label": period_label,
        }
    })
    # ป้องกัน browser เก็บ cache ผลการประเมิน
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    return response