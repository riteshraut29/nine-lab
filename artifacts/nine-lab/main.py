import os, uuid, asyncio, time, json, re, textwrap
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Nine Lab")
app.mount("/ninelab/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PORT = int(os.getenv("PORT", "22451"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

jobs: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=8)

# ── Supabase usage tracking ─────────────────────────────────────────────────

def check_usage_limit(ip: str) -> bool:
    """Return True if the IP is allowed to generate (under limit), False if blocked."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        # No Supabase configured → no rate limiting (open for self-hosted deployments)
        return True
    try:
        import httpx
        today = date.today().isoformat()
        url = f"{SUPABASE_URL}/rest/v1/usage_tracking"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        params = {"ip_address": f"eq.{ip}", "created_at": f"gte.{today}T00:00:00"}
        r = httpx.get(url, headers=headers, params=params, timeout=5)
        if r.status_code != 200:
            # Supabase error — fail-closed to protect against quota abuse when configured
            return False
        return len(r.json()) < 1
    except Exception:
        # Network/connection error — fail-closed when Supabase is explicitly configured
        return False


def record_usage(ip: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        import httpx
        url = f"{SUPABASE_URL}/rest/v1/usage_tracking"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        httpx.post(url, headers=headers, json={"ip_address": ip}, timeout=5)
    except Exception:
        pass

# ── Gemini helper ────────────────────────────────────────────────────────────

def gemini_call(prompt: str, retries: int = 1) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]
    last_err = None
    for model_name in models_to_try:
        for attempt in range(retries + 1):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "quota" in err_str or "rate" in err_str or "429" in err_str or "exhausted" in err_str:
                    if attempt < retries:
                        time.sleep(3)
                    else:
                        break  # Try next model
                elif attempt < retries:
                    time.sleep(2)
                else:
                    break  # Try next model
    raise last_err

# ── Tavily helper ────────────────────────────────────────────────────────────

def tavily_search(query: str, retries: int = 1) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    for attempt in range(retries + 1):
        try:
            result = client.search(query, max_results=5)
            return result.get("results", [])
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                raise e

# ── Agents ───────────────────────────────────────────────────────────────────

def agent_research(company: str, jd: str) -> dict:
    try:
        results = tavily_search(f"{company} company culture hiring process interview 2024 2025", retries=1)
        snippets = "\n".join([f"- {r.get('title','')}: {r.get('content','')[:300]}" for r in results[:4]])
        prompt = f"""You are a placement expert. Analyze this company and job description.

Company: {company}
Job Description: {jd}

Web search results about the company:
{snippets}

Provide a structured analysis with these sections:
1. COMPANY OVERVIEW (2-3 sentences about culture, size, domain)
2. ROLE REQUIREMENTS (top 5 must-have skills/qualifications)
3. INTERVIEW PROCESS (typical rounds at this company)
4. COMPANY RED FLAGS OR GREEN FLAGS (based on the JD and search results)
5. SALARY RANGE ESTIMATE (for Indian market, entry/mid level)

Be concise, factual, and actionable. Format each section clearly."""

        text = gemini_call(prompt, retries=1)
        return {"success": True, "data": text, "source": "tavily+gemini"}
    except Exception as e:
        err_msg = str(e)[:60]
        return {"success": False, "data": f"⚠️ Note: Company research data unavailable ({err_msg}). Proceeding with JD-only analysis.", "source": "fallback"}


def agent_analysis(resume: str, jd: str, company: str) -> dict:
    try:
        prompt = f"""You are a brutally honest career coach for Indian placements. Analyze this candidate.

RESUME:
{resume[:3000]}

JOB DESCRIPTION:
{jd[:2000]}

COMPANY: {company}

Give a REALITY CHECK with these sections:
1. MATCH SCORE: X/10 (be honest, explain in 1 sentence)
2. STRENGTHS (what the resume does well for this JD — top 3)
3. CRITICAL GAPS (what's missing that the JD demands — top 5, be specific)
4. RESUME RED FLAGS (ATS killers, formatting issues, weak language — top 3)
5. SKILLS TO ACQUIRE (concrete list with priority: HIGH/MEDIUM/LOW)
6. HONEST VERDICT (1-2 sentences: should they apply now or upskill first?)

Be direct. Indian students need honest feedback, not flattery."""

        text = gemini_call(prompt, retries=1)
        return {"success": True, "data": text}
    except Exception as e:
        err_msg = str(e)[:60]
        return {"success": False, "data": f"⚠️ Note: Resume analysis unavailable ({err_msg}). Some sections may be incomplete.", "source": "fallback"}


def agent_plan(resume: str, jd: str, company: str, analysis: str, research: str) -> dict:
    try:
        prompt = f"""You are a placement coach creating a detailed preparation plan for an Indian student.

COMPANY: {company}
JOB DESCRIPTION SUMMARY: {jd[:1000]}
CANDIDATE ANALYSIS: {analysis[:1500]}
COMPANY RESEARCH: {research[:1000]}

Create a detailed PREP PLAN with:
1. STRUCTURED STUDY SCHEDULE (phase-by-phase breakdown with specific tasks, time allocation, and priorities)
2. MOCK INTERVIEW QUESTIONS (10 role-specific technical questions with brief answers)
3. HR/BEHAVIORAL QUESTIONS (5 questions tailored to {company}'s culture)
4. RESOURCES (free resources: YouTube channels, websites, GitHub repos)
5. PRE-INTERVIEW CHECKLIST (10 items)

Format clearly with headers. Be specific to the role and company. Include Indian context (FAANG India, startups, service companies as relevant)."""

        text = gemini_call(prompt, retries=1)
        return {"success": True, "data": text}
    except Exception as e:
        err_msg = str(e)[:60]
        return {"success": False, "data": f"⚠️ Note: Prep plan generation encountered issues ({err_msg}). Partial plan provided.", "source": "fallback"}


def agent_resume(resume: str, jd: str, company: str) -> dict:
    try:
        prompt = f"""You are an expert ATS-optimized resume writer for Indian job market.

ORIGINAL RESUME:
{resume[:3000]}

TARGET JOB DESCRIPTION:
{jd[:2000]}

TARGET COMPANY: {company}

Rewrite this resume to be ATS-optimized and tailored for this specific role. Output a COMPLETE revised resume with:

1. CONTACT SECTION (name, email, phone, LinkedIn, GitHub — use placeholders if not in original)
2. PROFESSIONAL SUMMARY (3-4 lines, keyword-rich, tailored to this JD)
3. TECHNICAL SKILLS (organized by category, match JD keywords exactly)
4. WORK EXPERIENCE (each role with 3-5 bullet points starting with strong action verbs, quantified where possible)
5. PROJECTS (2-3 most relevant, with tech stack and impact)
6. EDUCATION (standard format)
7. CERTIFICATIONS / ACHIEVEMENTS (if any in original)

RULES:
- Use keywords from the JD naturally
- Start bullets with: Developed, Implemented, Optimized, Led, Built, Reduced, Increased, etc.
- Quantify everything possible (%, numbers, scale)
- Remove irrelevant information
- Keep it to 1 page worth of content
- Do NOT include photos, objective statements, or "References available"

Output the complete resume text, ready to paste."""

        text = gemini_call(prompt, retries=1)
        return {"success": True, "data": text}
    except Exception as e:
        err_msg = str(e)[:60]
        return {"success": False, "data": f"⚠️ Note: Resume revision unavailable ({err_msg}). Original resume content preserved.", "source": "fallback"}

# ── PDF Generation ───────────────────────────────────────────────────────────

def make_pdf_reality(job_id: str, company: str, analysis: dict, research: dict) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas as pdfcanvas

    PURPLE = HexColor("#6C63FF")
    DARK = HexColor("#1a1a2e")
    LIGHT_BG = HexColor("#f8f7ff")

    filename = f"{job_id}_reality.pdf"
    filepath = PDF_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath), pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=25*mm, bottomMargin=25*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("NLTitle", parent=styles["Title"],
        fontSize=22, textColor=PURPLE, spaceAfter=4, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle("NLSub", parent=styles["Normal"],
        fontSize=11, textColor=DARK, spaceAfter=12)
    h2_style = ParagraphStyle("NLH2", parent=styles["Normal"],
        fontSize=13, textColor=PURPLE, spaceAfter=6, spaceBefore=12, fontName="Helvetica-Bold")
    body_style = ParagraphStyle("NLBody", parent=styles["Normal"],
        fontSize=10, textColor=DARK, spaceAfter=4, leading=15)
    warning_style = ParagraphStyle("NLWarn", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#FF6B35"), spaceAfter=4, leading=15)

    story = []

    story.append(Paragraph("Nine Lab", ParagraphStyle("Brand", parent=styles["Normal"],
        fontSize=10, textColor=PURPLE, fontName="Helvetica-Bold")))
    story.append(Paragraph("Reality Report", title_style))
    story.append(Paragraph(f"Company: <b>{company}</b> · Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=12))

    def add_section(title, content):
        story.append(Paragraph(title, h2_style))
        if isinstance(content, str):
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 3))
                    continue
                style = warning_style if line.startswith("⚠️") else body_style
                safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                try:
                    story.append(Paragraph(safe_line, style))
                except Exception:
                    story.append(Paragraph(safe_line[:200], style))
        story.append(Spacer(1, 6))

    add_section("📊 Resume Analysis & Gap Report", analysis.get("data", "Data unavailable"))
    add_section("🏢 Company Intelligence", research.get("data", "Data unavailable"))

    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=6))
    story.append(Paragraph(
        "Nine Lab · Built for Indian students · ninelab.app",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=HexColor("#999999"), alignment=1)
    ))

    def add_page_number(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(HexColor("#999999"))
        canvas_obj.drawRightString(
            A4[0] - 20*mm, 15*mm,
            f"Page {doc_obj.page}"
        )
        canvas_obj.restoreState()

    doc.build(story, onLaterPages=add_page_number, onFirstPage=add_page_number)
    return filename


def make_pdf_plan(job_id: str, company: str, plan: dict) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.colors import HexColor

    PURPLE = HexColor("#6C63FF")
    DARK = HexColor("#1a1a2e")

    filename = f"{job_id}_plan.pdf"
    filepath = PDF_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath), pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=25*mm, bottomMargin=25*mm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"],
        fontSize=22, textColor=PURPLE, spaceAfter=4, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle("S", parent=styles["Normal"],
        fontSize=11, textColor=DARK, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Normal"],
        fontSize=13, textColor=PURPLE, spaceAfter=6, spaceBefore=12, fontName="Helvetica-Bold")
    body_style = ParagraphStyle("B", parent=styles["Normal"],
        fontSize=10, textColor=DARK, spaceAfter=4, leading=15)
    warning_style = ParagraphStyle("W", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#FF6B35"), spaceAfter=4, leading=15)

    story = []
    story.append(Paragraph("Nine Lab", ParagraphStyle("Brand", parent=styles["Normal"],
        fontSize=10, textColor=PURPLE, fontName="Helvetica-Bold")))
    story.append(Paragraph("Prep Plan", title_style))
    story.append(Paragraph(f"Company: <b>{company}</b> · Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=12))

    content = plan.get("data", "Data unavailable")
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        is_header = (line.startswith("DAY ") or line.startswith("##") or
                     line.startswith("MOCK") or line.startswith("HR/") or
                     line.startswith("RESOURCES") or line.startswith("DAY-OF") or
                     line.startswith("1.") and len(line) < 60 or
                     line.isupper() and len(line) < 80)
        style = h2_style if is_header else (warning_style if line.startswith("⚠️") else body_style)
        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        try:
            story.append(Paragraph(safe_line, style))
        except Exception:
            story.append(Paragraph(safe_line[:200], body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=6))
    story.append(Paragraph(
        "Nine Lab · Built for Indian students · ninelab.app",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=HexColor("#999999"), alignment=1)
    ))

    def add_page_number(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(HexColor("#999999"))
        canvas_obj.drawRightString(A4[0] - 20*mm, 15*mm, f"Page {doc_obj.page}")
        canvas_obj.restoreState()

    doc.build(story, onLaterPages=add_page_number, onFirstPage=add_page_number)
    return filename


def make_pdf_resume(job_id: str, company: str, resume_data: dict) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.colors import HexColor

    PURPLE = HexColor("#6C63FF")
    DARK = HexColor("#1a1a2e")

    filename = f"{job_id}_resume.pdf"
    filepath = PDF_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath), pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"],
        fontSize=20, textColor=DARK, spaceAfter=2, fontName="Helvetica-Bold")
    watermark_style = ParagraphStyle("WM", parent=styles["Normal"],
        fontSize=8, textColor=PURPLE, spaceAfter=8, fontName="Helvetica-Bold")
    h2_style = ParagraphStyle("H2", parent=styles["Normal"],
        fontSize=12, textColor=PURPLE, spaceAfter=3, spaceBefore=10,
        fontName="Helvetica-Bold", borderPad=2)
    body_style = ParagraphStyle("B", parent=styles["Normal"],
        fontSize=10, textColor=DARK, spaceAfter=3, leading=14)
    bullet_style = ParagraphStyle("BL", parent=styles["Normal"],
        fontSize=10, textColor=DARK, spaceAfter=2, leading=14, leftIndent=12)
    warning_style = ParagraphStyle("W", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#FF6B35"), spaceAfter=4, leading=15)

    story = []
    story.append(Paragraph(f"✨ Revised Resume · Tailored for {company}", watermark_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=8))

    content = resume_data.get("data", "Data unavailable")
    lines = content.split("\n")

    section_keywords = ["CONTACT", "SUMMARY", "PROFESSIONAL SUMMARY", "SKILLS", "TECHNICAL SKILLS",
                        "EXPERIENCE", "WORK EXPERIENCE", "PROJECTS", "EDUCATION", "CERTIFICATIONS",
                        "ACHIEVEMENTS", "AWARDS", "LANGUAGES", "INTERESTS"]

    first_line = True
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            story.append(Spacer(1, 3))
            continue

        if first_line and not any(kw in line_stripped.upper() for kw in section_keywords):
            safe = line_stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            try:
                story.append(Paragraph(safe, title_style))
            except Exception:
                story.append(Paragraph(safe[:100], title_style))
            first_line = False
            continue
        first_line = False

        is_section = any(line_stripped.upper().startswith(kw) for kw in section_keywords)
        is_bullet = line_stripped.startswith("•") or line_stripped.startswith("-") or line_stripped.startswith("*")
        is_warning = line_stripped.startswith("⚠️")

        if is_section:
            story.append(HRFlowable(width="100%", thickness=0.5, color=PURPLE, spaceAfter=3))
            style = h2_style
        elif is_bullet:
            style = bullet_style
        elif is_warning:
            style = warning_style
        else:
            style = body_style

        safe = line_stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        try:
            story.append(Paragraph(safe, style))
        except Exception:
            story.append(Paragraph(safe[:200], body_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=4))
    story.append(Paragraph(
        "Generated by Nine Lab · Built for Indian students",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7, textColor=HexColor("#999999"), alignment=1)
    ))

    def add_page_number(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(HexColor("#999999"))
        canvas_obj.drawRightString(A4[0] - 20*mm, 12*mm, f"Page {doc_obj.page}")
        canvas_obj.restoreState()

    doc.build(story, onLaterPages=add_page_number, onFirstPage=add_page_number)
    return filename

# ── Pipeline runner ──────────────────────────────────────────────────────────

def run_pipeline(job_id: str, resume: str, jd: str, company: str):
    def update(stage: str, pct: int, msg: str):
        jobs[job_id].update({"stage": stage, "progress": pct, "message": msg})

    try:
        update("research", 5, "🔍 Researching company and role...")

        loop = asyncio.new_event_loop()

        async def parallel_stage1():
            r_task = loop.run_in_executor(executor, agent_research, company, jd)
            a_task = loop.run_in_executor(executor, agent_analysis, resume, jd, company)
            return await asyncio.gather(r_task, a_task)

        research_result, analysis_result = loop.run_until_complete(parallel_stage1())
        update("research", 35, "✅ Company research done! Analyzing your resume...")

        update("analysis", 40, "🧠 Deep-analyzing resume vs JD...")
        time.sleep(0.5)
        update("analysis", 55, "📝 Crafting your 9-day prep plan...")

        async def parallel_stage2():
            p_task = loop.run_in_executor(executor, agent_plan, resume, jd, company,
                                          analysis_result["data"], research_result["data"])
            r_task = loop.run_in_executor(executor, agent_resume, resume, jd, company)
            return await asyncio.gather(p_task, r_task)

        plan_result, resume_result = loop.run_until_complete(parallel_stage2())
        loop.close()

        update("pdf", 75, "📄 Generating your Reality Report PDF...")
        reality_file = make_pdf_reality(job_id, company, analysis_result, research_result)

        update("pdf", 82, "📄 Generating your Prep Plan PDF...")
        plan_file = make_pdf_plan(job_id, company, plan_result)

        update("pdf", 90, "📄 Generating your Revised Resume PDF...")
        resume_file = make_pdf_resume(job_id, company, resume_result)

        jobs[job_id].update({
            "stage": "done",
            "progress": 100,
            "message": "🎉 Tera placement ready! Download your PDFs below.",
            "files": {
                "reality": reality_file,
                "plan": plan_file,
                "resume": resume_file,
            }
        })

    except Exception as e:
        jobs[job_id].update({
            "stage": "error",
            "progress": 0,
            "message": f"❌ Error: {str(e)[:200]}. Check your API keys and try again.",
        })

# ── Request/Response models ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    resume: str
    jd: str
    company: str

# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/ninelab/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Sirf PDF files allowed hain. Please upload a .pdf file.")
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, detail="File bahut bada hai! Max 10MB allowed hai.")
    try:
        from pypdf import PdfReader
        import io
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(400, detail="File bahut bada hai! Max 10MB allowed hai.")
        reader = PdfReader(io.BytesIO(contents))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise HTTPException(422, detail="Is PDF se text extract nahi ho saka. PDF mein scannable text hona chahiye (image-based PDFs work nahi karte).")
        word_count = len(full_text.split())
        return JSONResponse({"text": full_text, "pages": len(reader.pages), "words": word_count})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"PDF read karne mein error aaya: {str(e)}")


@app.get("/", response_class=RedirectResponse)
async def root_redirect():
    return RedirectResponse(url="/ninelab/", status_code=302)

@app.get("/ninelab", response_class=HTMLResponse)
@app.get("/ninelab/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Nine Lab loading...</h1>", status_code=200)


@app.post("/ninelab/generate")
async def generate(req: GenerateRequest, request: Request):
    # Use X-Forwarded-For when behind a proxy (Replit, nginx, etc.)
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"

    if not req.resume.strip():
        raise HTTPException(400, detail="Please provide your resume.")
    if not req.jd.strip():
        raise HTTPException(400, detail="Please provide the job description.")
    if len(req.company.strip()) < 2:
        raise HTTPException(400, detail="Please enter the company name.")

    if not GEMINI_API_KEY:
        raise HTTPException(400, detail="GEMINI_API_KEY not configured. Add it to your environment variables.")
    if not TAVILY_API_KEY:
        raise HTTPException(400, detail="TAVILY_API_KEY not configured. Add it to your environment variables.")

    if not check_usage_limit(ip):
        raise HTTPException(429, detail=(
            "Aaj ka free use ho gaya! Kal wapas aao. (1 free use per day per IP)\n\n"
            "Want unlimited access? Self-host Nine Lab with your own API keys — it's free and open source. "
            "Copy the artifacts/nine-lab/ folder, add GEMINI_API_KEY and TAVILY_API_KEY, and run: "
            "pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000"
        ))

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "stage": "queued",
        "progress": 0,
        "message": "⏳ Starting your placement pipeline...",
        "files": None,
        "created_at": time.time(),
    }

    record_usage(ip)
    executor.submit(run_pipeline, job_id, req.resume, req.jd, req.company)

    return JSONResponse({"job_id": job_id})


@app.get("/ninelab/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, detail="Job not found. Shayad expire ho gaya.")
    return JSONResponse(jobs[job_id])


@app.get("/ninelab/pdf/{filename}")
async def download_pdf(filename: str):
    if not filename.endswith(".pdf") or "/" in filename or ".." in filename:
        raise HTTPException(400, detail="Invalid filename")
    filepath = PDF_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, detail="PDF not found")
    return FileResponse(str(filepath), media_type="application/pdf",
                        filename=filename, headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/ninelab/health")
async def health():
    return {"status": "ok", "service": "Nine Lab", "keys": {
        "gemini": bool(GEMINI_API_KEY),
        "tavily": bool(TAVILY_API_KEY),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
    }}
