# ==============================================================================
# main.py — ไฟล์หลักของเว็บแอปพลิเคชัน (Backend / Server)
# ------------------------------------------------------------------------------
# หน้าที่ : เป็น "สมองกลาง" ของระบบ ทำหน้าที่รับ HTTP Request จากเบราว์เซอร์
#           แล้วตอบกลับด้วย HTML หรือ Redirect
# ทำงานร่วมกับ :
#   - model_loader.py  → เรียกฟังก์ชันพยากรณ์ความเสี่ยง + ดึงรายชื่อโรงเรียน
#   - templates/index.html  → หน้า Dashboard ที่แสดงให้อาจารย์ใช้งาน
#   - templates/login.html  → หน้าเข้าสู่ระบบ
#   - Firebase (cloud)      → ใช้ยืนยันตัวตนผู้ใช้ผ่าน ID Token
# ==============================================================================

import os       # ใช้อ่านตัวแปร Environment เช่น FIREBASE_CREDENTIALS บน Render
import json     # ใช้แปลง JSON string (จาก Environment) เป็น dict ของ Python
import firebase_admin                        # Firebase Admin SDK — จัดการ Auth ฝั่ง Server
from firebase_admin import credentials, auth # credentials: โหลดกุญแจ, auth: ตรวจสอบ Token

from fastapi import FastAPI, Request, Form   # FastAPI: framework เว็บ, Request: ข้อมูล HTTP, Form: รับค่าจาก HTML form
from fastapi.responses import HTMLResponse, RedirectResponse  # HTMLResponse: ตอบกลับเป็น HTML, RedirectResponse: เปลี่ยนหน้า
from fastapi.templating import Jinja2Templates  # Jinja2: ระบบ template ที่ฝัง Python ลงใน HTML ได้ (ใช้ {{ }} และ {% %})

# นำเข้าฟังก์ชันจาก model_loader.py:
#   predict_risk_with_perturbation → ทำนายความเสี่ยงแบบ Perturbation Smoothing
#   get_school_list                → ดึงรายชื่อโรงเรียนทั้งหมดจาก model pipeline
from model_loader import predict_risk_with_perturbation, get_school_list

# สร้าง instance ของ FastAPI — เป็นตัวแทนของทั้งเว็บแอป ใช้ลงทะเบียน route ต่าง ๆ
app = FastAPI()

# กำหนดโฟลเดอร์ที่เก็บไฟล์ HTML template (templates/index.html, templates/login.html)
# Jinja2Templates จะค้นหา .html จากโฟลเดอร์นี้เวลาเรียก TemplateResponse
templates = Jinja2Templates(directory="templates")

# โหลดรายชื่อโรงเรียนจาก pipeline ครั้งเดียวตอนเซิร์ฟเวอร์เริ่มทำงาน
# เก็บไว้ใน SCHOOL_LIST (list ของ string) เพื่อไม่ต้องโหลดซ้ำทุก request
# ส่งต่อไปยัง index.html ผ่าน context เพื่อสร้าง dropdown ค้นหาโรงเรียน
SCHOOL_LIST = get_school_list()

# ==========================================
# 🔑 โค้ดเช็คกุญแจแบบ 2 ระบบ (Local & Render)
# ==========================================
# firebase_cred_json : string JSON ที่อ่านจาก Environment Variable ชื่อ "FIREBASE_CREDENTIALS"
#   - บน Render.com → ตั้งค่า Environment Variable ไว้ที่หน้า Dashboard ของ Render
#   - บนเครื่อง Local → ไม่มีตัวแปรนี้ ให้ใช้ไฟล์ firebase-key.json แทน
firebase_cred_json = os.environ.get("FIREBASE_CREDENTIALS")

if firebase_cred_json:
    # 🌐 กรณีรันอยู่บน Render (อ่านจาก Environment)
    # json.loads() แปลง JSON string → dict แล้วใช้สร้าง Certificate Object
    print("กำลังใช้งาน Firebase จาก Render Environment")
    cred_dict = json.loads(firebase_cred_json)   # cred_dict: dict ที่มี private_key, client_email ฯลฯ
    cred = credentials.Certificate(cred_dict)    # cred: กุญแจ Service Account ของ Firebase
    firebase_admin.initialize_app(cred)          # เริ่มต้น Firebase Admin ด้วยกุญแจนั้น
elif os.path.exists("firebase-key.json"):
    # 💻 กรณีรันอยู่บนเครื่อง Local (อ่านจากไฟล์)
    # ไฟล์ firebase-key.json ต้องอยู่ใน root ของโปรเจกต์ (ห้าม push ขึ้น Git!)
    print("กำลังใช้งาน Firebase จากไฟล์ firebase-key.json (Local)")
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
else:
    print("❌ ERROR: หากุญแจ Firebase ไม่เจอ ทั้งบน Render และ Local!")
# ==========================================

def get_current_user(request: Request):
    """
    ตรวจสอบว่าผู้ใช้ login อยู่หรือไม่
    ─────────────────────────────────────
    อ่าน Cookie ชื่อ "session" จาก HTTP Request
    แล้วส่ง ID Token นั้นให้ Firebase ยืนยัน (ป้องกันการปลอมแปลง Cookie)
    
    คืนค่า:
      - decoded_token (dict) : ถ้า Token ถูกต้อง มีข้อมูลเช่น uid, email
      - None                 : ถ้าไม่มี Cookie หรือ Token หมดอายุ/ถูกแก้ไข
    
    ทำงานร่วมกับ: login.html → JavaScript เขียน Cookie "session" หลัง login สำเร็จ
    """
    token = request.cookies.get("session")   # token: ID Token ที่ Firebase ออกให้หลัง login
    if not token:
        return None
    try:
        # ส่ง Token ให้ Firebase ตรวจสอบ (ป้องกันการปลอมแปลง Cookie)
        decoded_token = auth.verify_id_token(token)
        return decoded_token # คืนค่าข้อมูล User กลับไป (เช่น อีเมล)
    except Exception as e:
        print(f"Token invalid or expired: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Route: GET "/" — หน้าแรก (เปลี่ยนเส้นทางไปหน้า Login)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    หน้าแรกของเว็บ (/)
    - ถ้า login อยู่แล้ว → redirect ไป /dashboard โดยอัตโนมัติ
    - ถ้ายังไม่ login    → แสดงหน้า login.html
    ทำงานร่วมกับ: templates/login.html
    """
    if get_current_user(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse(request, "login.html")

# ──────────────────────────────────────────────────────────────────────────────
# Route: GET "/logout" — ออกจากระบบ
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/logout")
async def logout(request: Request):
    """
    ลบ Cookie "session" แล้วพาไปหน้า Login
    เมื่อกดปุ่ม "ออกจากระบบ" ใน index.html จะเรียก route นี้
    ทำงานร่วมกับ: templates/index.html (ปุ่ม "ออกจากระบบ")
    """
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session")   # ลบ Cookie เพื่อยกเลิก session ที่ login ไว้
    return response

# ──────────────────────────────────────────────────────────────────────────────
# Route: GET "/dashboard" — หน้า Dashboard หลัก
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    แสดงหน้า Dashboard (index.html) สำหรับอาจารย์ที่ login แล้ว
    ส่ง context ไปให้ Jinja2 ใน index.html ได้ใช้:
      - schools : list ชื่อโรงเรียนทั้งหมด → ใช้สร้าง dropdown ค้นหา
      - user    : dict ข้อมูลผู้ใช้จาก Firebase (uid, email ฯลฯ)
    ทำงานร่วมกับ: templates/index.html
    """
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/")   # ถ้าไม่ได้ login → กลับไปหน้า Login
    return templates.TemplateResponse(request, "index.html", {
        "schools": SCHOOL_LIST,
        "user": user  # ส่งข้อมูล user ไปเผื่อใช้แสดงชื่อ/อีเมลใน index.html
    })

# ──────────────────────────────────────────────────────────────────────────────
# Route: POST "/predict" — รับข้อมูลฟอร์มแล้วพยากรณ์ความเสี่ยง
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gpa: float = Form(...),           # GPA ก่อนรับเข้า (0.00–4.00) — รับจาก input ชื่อ "gpa" ใน index.html
    admission: str = Form(...),       # วิธีรับเข้า เช่น "โควตา" / "สอบคัดเลือก" — จาก <select name="admission">
    degree: str = Form(...),          # วุฒิการศึกษา เช่น "ม.6" / "ปวช." / "ปวส." — จาก <select name="degree">
    school: str = Form(...),          # ชื่อโรงเรียน/สถาบัน — จาก hidden input #sc-val ใน index.html
    grade_y1s1: float = Form(...),    # เกรดเฉลี่ยปีที่ 1 เทอมที่ 1 (0.00–4.00) — จาก input ชื่อ "grade_y1s1"
):
    """
    รับข้อมูลนักศึกษาจากฟอร์ม แล้วส่งให้ model_loader.py พยากรณ์
    
    result : dict ผลลัพธ์จาก predict_risk_with_perturbation() มีเช่น
              - risk_percent    : ความเสี่ยง (%)
              - success_percent : โอกาสสำเร็จ (%)
              - label           : "เสี่ยงพ้นสภาพ" หรือ "ปลอดภัย"
              - confidence      : ความมั่นใจ "สูง"/"ปานกลาง"/"ต่ำ"
              - perturb_std     : ค่าความเสถียรของผล (%)
              - n_perturbations : จำนวนรอบที่ทดสอบ
    
    form  : dict เก็บค่าที่ผู้ใช้กรอกไว้ ส่งกลับไปให้ index.html แสดงใหม่
            (เพื่อไม่ให้ช่องกรอกข้อมูลล้างหลังกด submit)
    
    ทำงานร่วมกับ: model_loader.py → predict_risk_with_perturbation()
                  templates/index.html → แสดงผล result และ form
    """
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/", status_code=303)   # ถ้า session หมด → บังคับ login ใหม่
        
    # เรียกฟังก์ชันพยากรณ์จาก model_loader.py พร้อมส่งข้อมูล input ทั้งหมด
    result = predict_risk_with_perturbation(gpa, admission, degree, school, grade_y1s1)
    return templates.TemplateResponse(request, "index.html", {
        "result": result,           # ผลพยากรณ์ → ใช้ใน {% if result %} ของ index.html
        "schools": SCHOOL_LIST,     # รายชื่อโรงเรียน → สร้าง dropdown
        "user": user,               # ข้อมูลผู้ใช้ → แสดงชื่อใน nav
        "form": {"gpa": gpa, "admission": admission, "degree": degree, "school": school, "grade_y1s1": grade_y1s1}
        # form → เก็บค่าที่กรอกไว้ เพื่อ pre-fill ช่องกรอกข้อมูลหลัง submit
    })