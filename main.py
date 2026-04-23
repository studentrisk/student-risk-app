from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from model_loader import predict_risk, get_school_list

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# โหลดรายชื่อโรงเรียนครั้งเดียวตอนเริ่มต้น
SCHOOL_LIST = get_school_list()

TEACHERS = {
    "T001": {"password": "1234", "name": "อ.สมชาย ใจดี", "department": "สาขาวิทยาการคอมพิวเตอร์"},
    "T002": {"password": "5678", "name": "อ.สมหญิง รักเรียน", "department": "สาขาวิทยาการคอมพิวเตอร์"},
}

sessions = {}

def get_current_user(request: Request):
    token = request.cookies.get("session")
    return sessions.get(token)

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.post("/login")
async def login(request: Request, teacher_id: str = Form(...), password: str = Form(...)):
    teacher = TEACHERS.get(teacher_id)
    if teacher and teacher["password"] == password:
        import secrets
        token = secrets.token_hex(16)
        sessions[token] = {"id": teacher_id, **teacher}
        response = RedirectResponse("/dashboard", status_code=302)
        response.set_cookie("session", token)
        return response
    return templates.TemplateResponse(request, "login.html", {"error": "รหัสอาจารย์หรือรหัสผ่านไม่ถูกต้อง"})

@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    sessions.pop(token, None)
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "index.html", {
        "user": user,
        "schools": SCHOOL_LIST
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gpa: float = Form(...),
    admission: str = Form(...),
    degree: str = Form(...),
    school: str = Form(...)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/", status_code=303)
    result = predict_risk(gpa, admission, degree, school)
    return templates.TemplateResponse(request, "index.html", {
        "user": user,
        "result": result,
        "schools": SCHOOL_LIST,
        "form": {"gpa": gpa, "admission": admission, "degree": degree, "school": school}
    })