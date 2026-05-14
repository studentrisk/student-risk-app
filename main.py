import os
import json
import firebase_admin
from firebase_admin import credentials, auth

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
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
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "index.html", {
        "schools": SCHOOL_LIST,
        "user": user  # ส่งข้อมูล user ไปเผื่อใช้แสดงชื่อ/อีเมลใน index.html
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gpa: float = Form(...),
    admission: str = Form(...),
    degree: str = Form(...),
    school: str = Form(...),
    grade_y1s1: float = Form(...),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/", status_code=303)
        
    result = predict_risk_with_perturbation(gpa, admission, degree, school, grade_y1s1)
    return templates.TemplateResponse(request, "index.html", {
        "result": result,
        "schools": SCHOOL_LIST,
        "user": user,
        "form": {"gpa": gpa, "admission": admission, "degree": degree, "school": school, "grade_y1s1": grade_y1s1}
    })