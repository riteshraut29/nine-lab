import os, uuid, asyncio, time, json, re, textwrap
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Header
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

jobs: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=8)
user_daily_usage: dict[str, dict] = {}  # {user_id: {"date": "YYYY-MM-DD", "count": N}}

# ── Supabase Auth helpers ─────────────────────────────────────────────────────

def supabase_auth_request(method: str, path: str, payload: dict = None, token: str = None) -> dict:
    import httpx
    key = token or SUPABASE_KEY
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    url = f"{SUPABASE_URL}/auth/v1{path}"
    try:
        if method == "POST":
            r = httpx.post(url, headers=headers, json=payload, timeout=10)
        elif method == "GET":
            r = httpx.get(url, headers=headers, timeout=10)
        else:
            r = httpx.delete(url, headers=headers, json=payload, timeout=10)
        return {"status": r.status_code, "data": r.json() if r.text else {}}
    except Exception as e:
        return {"status": 500, "data": {"error": str(e)}}


def get_user_from_token(token: str) -> Optional[dict]:
    if not token or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    result = supabase_auth_request("GET", "/user", token=token)
    if result["status"] == 200 and "id" in result["data"]:
        return result["data"]
    return None


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

# ── AI helper (Groq primary, Gemini fallback) ─────────────────────────────────

def gemini_call(prompt: str, retries: int = 3, temperature: float = 0.7,
                system_prompt: str = None) -> str:
    last_err = None

    # ── Try Groq first (faster, higher limits) ───────────────────────────────
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model_name in groq_models:
            for attempt in range(retries + 1):
                try:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    response = groq_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=4096,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    is_rate_limit = "rate" in err_str or "429" in err_str or "limit" in err_str
                    if attempt < retries:
                        time.sleep((2 ** attempt) * 3 if is_rate_limit else 2)
                    else:
                        break  # Try next model

    # ── Fallback to Gemini if Groq unavailable or all retries exhausted ───────
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        for model_name in gemini_models:
            for attempt in range(retries + 1):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        full_prompt,
                        generation_config={"temperature": temperature}
                    )
                    return response.text
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    is_rate_limit = "quota" in err_str or "rate" in err_str or "429" in err_str or "exhausted" in err_str
                    if attempt < retries:
                        time.sleep((2 ** attempt) * 5 if is_rate_limit else 3)
                    else:
                        break

    raise last_err or RuntimeError("No AI provider configured.")

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

# ── Text helpers ──────────────────────────────────────────────────────────────

def strip_md(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'`([^`]+?)`', r'\1', text)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\s', '\u2022 ', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s', '\u2022 ', text, flags=re.MULTILINE)
    return text.strip()

def safe_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def extract_section(text: str, header: str) -> str:
    pattern = rf'(?:^|\n)\s*(?:\d+\.\s*)?{re.escape(header)}[:\s]*\n?(.*?)(?=\n\s*(?:\d+\.\s*)?[A-Z][A-Z\s/&]{{3,}}\s*:|$)'
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return strip_md(m.group(1).strip()) if m else ""

def section_bullets(text: str) -> list:
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith(('===', '---'))]
    return [l.lstrip('\u2022•-*0123456789. ') for l in lines if len(l) > 3]

# ── Agents ───────────────────────────────────────────────────────────────────

def agent_research(company: str, jd: str) -> dict:
    """Tavily-only web search pass — returns raw snippets for other agents."""
    try:
        results = tavily_search(
            f"{company} company culture interview process hiring 2024 2025 India", retries=1
        )
        snippets = "\n".join([
            f"[{r.get('title', '')}] {r.get('content', '')[:400]}"
            for r in results[:5]
        ])
        return {"success": True, "data": snippets, "source": "tavily"}
    except Exception as e:
        return {"success": False, "data": f"Web research unavailable: {str(e)[:60]}", "source": "fallback"}


def agent_company_report(company: str, jd: str, research_snippets: str) -> dict:
    """NEXUS — elite company intelligence analyst."""
    # Extract job title from first non-empty line of JD
    job_title = next((l.strip() for l in jd.split('\n') if l.strip()), "the role")[:80]

    NEXUS_SYSTEM = """You are NEXUS — an elite corporate intelligence analyst with 15 years of experience researching companies for candidates at McKinsey, BCG, and top Indian placement cells. You have deep knowledge of Indian IT, startup, and corporate ecosystems.

Your ONE job: give this candidate an unfair advantage before their interview.

THINKING PROCESS (follow this exactly):
Step 1: Understand the company's core business model
Step 2: Identify what type of people they hire and why
Step 3: Map their interview process based on known patterns
Step 4: Find the real red flags most candidates miss
Step 5: Craft the 3 most impressive things the candidate can say

OUTPUT RULES:
- Every fact must be specific to THIS company. Zero generic filler.
- Salary in Indian Rupees only. Use realistic Indian market data.
- Interview process: name each round specifically
- If data unavailable, say exactly: "Data not publicly available" — never guess or hallucinate
- Tone: insider friend who works there, not a Wikipedia article
- Zero markdown symbols (###, **, --) in output

WHAT GOOD LOOKS LIKE:
"Senwell's technical round focuses heavily on JavaScript fundamentals — they specifically ask about closures and async/await."

WHAT BAD LOOKS LIKE (never do this):
"The company has a good interview process with multiple rounds."

OUTPUT FORMAT — use these exact section headers:
COMPANY OVERVIEW
CULTURE AND ENVIRONMENT
RECENT DEVELOPMENTS
INTERVIEW PROCESS
WHAT THEY TEST IN EACH ROUND
TOP 5 INTERVIEW QUESTIONS AT THIS COMPANY
TECH STACK THEY USE
SALARY RANGE
GREEN FLAGS
RED FLAGS
YOUR INTERVIEW ADVANTAGE"""

    user_prompt = f"""Research this company thoroughly for a job candidate:

Company: {company}
Role Applied For: {job_title}
Web Research Data: {research_snippets[:3000]}

The candidate has an upcoming interview. Give them everything they need to walk in with confidence and insider knowledge. Think step by step before writing each section."""

    try:
        text = gemini_call(user_prompt, retries=2, temperature=0.2, system_prompt=NEXUS_SYSTEM)
        return {"success": True, "data": text}
    except Exception as e:
        return {"success": False, "data": f"Company report unavailable ({str(e)[:60]})."}


def agent_analysis(resume: str, jd: str, company: str, research: str = "") -> dict:
    """ARIA — senior placement mentor with brutal honesty and deep care."""
    ARIA_SYSTEM = """You are ARIA — a senior placement mentor who combines the analytical precision of a Fortune 500 HR director with the warmth of a trusted elder sibling. You have personally reviewed 50,000 resumes and placed 10,000+ Indian students.

Your philosophy: Brutal honesty delivered with deep care. Every problem you identify, you solve.

THINKING PROCESS (follow this exactly):
Step 1: Read the JD carefully — what are the TOP 3 non-negotiable requirements?
Step 2: Read the resume — does it prove these 3 requirements? Score each 0-10.
Step 3: Calculate match score honestly (average x 10 = %)
Step 4: Identify gaps in PRIORITY ORDER — what is costing them the job most?
Step 5: For each gap, think: what is the fastest specific fix?
Step 6: Separate company analysis completely from candidate analysis
Step 7: Write closing message as if this is your younger sibling

MATCH SCORE CALIBRATION:
90-100: Almost perfect, minor polish needed
70-89: Strong candidate, 2-3 gaps to fix
50-69: Potential candidate, significant work needed
30-49: Major gaps, needs focused preparation
0-29: Fundamental mismatch, needs honest redirect

TONE RULES (non-negotiable):
GOOD: "Your Python skills are strong, but the JD needs Node.js — here is how to bridge this in 2 weeks"
BAD: "You do not know Node.js which is a critical requirement"
GOOD: "Fix your LinkedIn URL first — it takes 30 seconds and shows attention to detail"
BAD: "You left a placeholder URL which looks unprofessional"

The candidate should finish reading feeling: capable + clear on next steps + motivated
Never make them feel: defeated, embarrassed, overwhelmed

OUTPUT FORMAT — use these exact section headers:
MATCH SCORE: [0-100]
VERDICT: [one honest sentence]
YOUR TOP 3 STRENGTHS
YOUR TOP 3 PRIORITY GAPS
DETAILED STRENGTHS ANALYSIS
GAPS WITH PRIORITY AND SPECIFIC FIX
RESUME RED FLAGS WITH EXACT FIXES
ABOUT THE COMPANY
WHAT THIS COMPANY LOOKS FOR
THEIR INTERVIEW PROCESS
SALARY RANGE
YOUR PRIORITY ACTION LIST
NEXT STEPS CHECKLIST
CLOSING MESSAGE"""

    user_prompt = f"""Analyze this candidate's fit for the role. Think through each step carefully before writing.

RESUME:
{resume[:3000]}

JOB DESCRIPTION:
{jd[:2000]}

COMPANY: {company}
COMPANY RESEARCH: {research[:1500]}

Evaluate honestly. Remember: this person's career depends on accurate feedback. Give them the truth, but give them hope too."""

    try:
        text = gemini_call(user_prompt, retries=2, temperature=0.3, system_prompt=ARIA_SYSTEM)
        return {"success": True, "data": text}
    except Exception as e:
        return {"success": False, "data": f"Analysis unavailable ({str(e)[:60]}). Please try again."}


def _extract_match_score(analysis_text: str) -> int:
    """Pull numeric match score from ARIA's output."""
    for line in analysis_text.split('\n'):
        if 'MATCH SCORE' in line.upper():
            nums = re.findall(r'\d+', line)
            if nums:
                return min(int(nums[0]), 100)
    return 50


def _extract_gap_summary(analysis_text: str) -> str:
    """Pull the top gaps section from ARIA's output for FORGE."""
    gaps = extract_section(analysis_text, "YOUR TOP 3 PRIORITY GAPS")
    if not gaps:
        gaps = extract_section(analysis_text, "GAPS WITH PRIORITY AND SPECIFIC FIX")
    return gaps[:800] if gaps else "Improve skill alignment with job description requirements."


def agent_plan(resume: str, jd: str, company: str, analysis: str, research: str,
               match_score: int = 50) -> dict:
    """ATLAS — world-class placement coach for Indian students."""
    ATLAS_SYSTEM = """You are ATLAS — a world-class placement coach who has trained 5,000+ Indian students from tier-2 and tier-3 colleges to crack placements at top companies. You know exactly what Indian companies test, what resources work best in the Indian context, and how to build skills fast.

Your superpower: You never give generic advice. Every word you write is specific to this exact candidate, this exact role, this exact company.

THINKING PROCESS (follow this exactly):
Step 1: Assess current level from resume — what do they actually know vs what they claim?
Step 2: Map JD requirements — what does this specific role actually need?
Step 3: Find the GAP — rank from most critical to least critical
Step 4: For each phase, identify the FASTEST path to demonstrable skill
Step 5: Pull questions this specific company is known to ask
Step 6: Find resources that are FREE, Indian context, and actually effective

PRIORITY FRAMEWORK:
CRITICAL = Without this, they get rejected in screening round
IMPORTANT = Without this, they fail technical round
GOOD TO HAVE = Separates from other candidates in final round

ANTI-GENERIC RULES (follow strictly):
WRONG: "Learn React"
RIGHT: "Build a Task Manager app in React with CRUD operations — deploy on Vercel with GitHub link"
WRONG: "Practice DSA"
RIGHT: "Solve 20 Array and String problems on GFG — focus on Two Pointer and Sliding Window patterns"
WRONG: "Prepare for HR"
RIGHT: "For this company's HR round: prepare why you want this specific company — mention specific things about their work culture"
NEVER USE: "Day 1", "Day 2", "Week 1", "9 days", "9-day" — use Phase 1, Phase 2, Phase 3 only

RESOURCE RULES:
Indian creators first: CodeWithHarry, Apna College, Striver, Hitesh Choudhary
Then international: freeCodeCamp, MDN Docs, official documentation
Free resources only unless paid is genuinely much better

OUTPUT FORMAT — use these exact section headers:
CURRENT LEVEL ASSESSMENT
PRIORITY 1 — CRITICAL
PRIORITY 2 — IMPORTANT
PRIORITY 3 — GOOD TO HAVE
PHASE 1: FOUNDATION
PHASE 2: CORE SKILLS
PHASE 3: COMPANY-SPECIFIC POLISH
TOP 10 TECHNICAL INTERVIEW QUESTIONS WITH ANSWERS
TOP 5 HR QUESTIONS WITH ANSWERS
FREE RESOURCES
INTERVIEW DAY CHECKLIST"""

    user_prompt = f"""Create a preparation roadmap for this candidate. Think step by step — assess their level first, then build the plan.

RESUME: {resume[:2500]}
JOB DESCRIPTION: {jd[:1500]}
COMPANY: {company}
COMPANY RESEARCH: {research[:1500]}
MATCH SCORE FROM ANALYSIS: {match_score}

Build a plan that meets this candidate exactly where they are. Make every task specific and completable. This plan should feel like it was written personally for them — because it was."""

    try:
        text = gemini_call(user_prompt, retries=2, temperature=0.4, system_prompt=ATLAS_SYSTEM)
        return {"success": True, "data": text}
    except Exception as e:
        return {"success": False, "data": f"Prep plan generation encountered issues ({str(e)[:60]})."}


def agent_resume(resume: str, jd: str, company: str, company_research: str = "",
                 gap_summary: str = "") -> dict:
    """FORGE — elite resume architect for ATS and human readers."""
    FORGE_SYSTEM = """You are FORGE — an elite resume architect who has crafted resumes that landed candidates at Google, Microsoft, Goldman Sachs, Zomato, Flipkart, and 200+ top Indian companies. You are an expert in ATS optimization and understand exactly how both machines and humans read resumes in 2025.

Your standard: Every resume you write passes ATS AND impresses a senior recruiter in 6 seconds.

THINKING PROCESS (follow this exactly):
Step 1: Extract the TOP 10 keywords from the JD — exact phrases not synonyms
Step 2: Identify what THIS company values most from the research
Step 3: Find every achievement in the resume and quantify them
Step 4: Map candidate projects to what this company cares about
Step 5: Write Professional Summary last — after you know what to highlight

ATS 2025 HARD RULES:
- Single column ONLY — two columns break ATS parsers
- No tables, no text boxes, no graphics, no icons
- Section headers: exact standard names in CAPS
- Keywords must appear EXACTLY as written in JD
- Job title from JD must appear in Professional Summary

BULLET POINT FORMULA (mandatory for every single bullet):
[Strong Action Verb] + [What you did specifically] + [Result with number]
GOOD: "Developed a Python-based resume analyzer that processed 500+ applications and reduced screening time by 40%"
BAD: "Worked on Python project for resume analysis"

BANNED PHRASES — delete and replace with proof:
"Quick learner" — show a project built fast instead
"Team player" — show collaboration with specific outcome instead
"Hardworking" — show quantified result instead
"Passionate about" — show project or contribution instead
"Good communication" — show presentation or documentation achievement instead
"Detail-oriented" — show debugging or QA work instead

PROFESSIONAL SUMMARY FORMULA:
[Job Title from JD] with [experience level] in [top 2 skills from JD]. [One specific achievement]. Seeking to [contribute specific value] at [Company Name].

OUTPUT RULES:
- Clean plain text with section headers in CAPS
- Bullet points start with a dash
- ZERO Nine Lab branding anywhere
- ZERO AI generation mentions
- Must read like a human senior developer wrote it
- Zero markdown symbols

OUTPUT FORMAT:
[CANDIDATE FULL NAME]
[Phone] | [Email] | [LinkedIn] | [GitHub] | [City, State]

PROFESSIONAL SUMMARY

TECHNICAL SKILLS

WORK EXPERIENCE

PROJECTS

EDUCATION

ACHIEVEMENTS"""

    user_prompt = f"""Rewrite this resume to be ATS-optimized and tailored for this specific role and company.

ORIGINAL RESUME: {resume[:3000]}
JOB DESCRIPTION: {jd[:2000]}
COMPANY: {company}
WHAT THIS COMPANY VALUES: {company_research[:800]}
CANDIDATE GAP AREAS: {gap_summary}

Think step by step:
1. First extract all keywords from the JD
2. Then identify all achievements to quantify
3. Then write each section
4. Finally write the Professional Summary

Make this resume feel like it was written by someone who knows exactly what {company} is looking for — because it was."""

    try:
        text = gemini_call(user_prompt, retries=2, temperature=0.1, system_prompt=FORGE_SYSTEM)
        return {"success": True, "data": text}
    except Exception as e:
        return {"success": False, "data": f"Resume revision unavailable ({str(e)[:60]}). Original resume content preserved."}

# ── PDF Generation ───────────────────────────────────────────────────────────

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas as pdfcanvas

PDF_PURPLE = HexColor("#6C63FF")
PDF_DARK = HexColor("#1A1A2E")
PDF_LIGHT = HexColor("#F0EEFF")
PDF_GREEN = HexColor("#22C55E")
PDF_RED = HexColor("#EF4444")
PDF_AMBER = HexColor("#F59E0B")
PDF_GREY = HexColor("#D1D5DB")
PDF_LIGHT_GREEN = HexColor("#DCFCE7")
PDF_LIGHT_RED = HexColor("#FEE2E2")
PDF_LIGHT_AMBER = HexColor("#FEF3C7")
PDF_WHITE = HexColor("#FFFFFF")
PDF_MUTED = HexColor("#888888")

def _pdf_styles():
    s = getSampleStyleSheet()
    return s, {
        "title": ParagraphStyle("NLTitle", parent=s["Normal"], fontSize=18, textColor=PDF_DARK, fontName="Helvetica-Bold", spaceAfter=4, leading=22),
        "h2": ParagraphStyle("NLH2", parent=s["Normal"], fontSize=13, textColor=PDF_PURPLE, fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=10, leading=16),
        "h3": ParagraphStyle("NLH3", parent=s["Normal"], fontSize=11, textColor=PDF_DARK, fontName="Helvetica-Bold", spaceAfter=4, leading=14),
        "body": ParagraphStyle("NLBody", parent=s["Normal"], fontSize=10.5, textColor=PDF_DARK, spaceAfter=4, leading=14),
        "small": ParagraphStyle("NLSmall", parent=s["Normal"], fontSize=9, textColor=PDF_DARK, spaceAfter=3, leading=12),
        "subtitle": ParagraphStyle("NLSub", parent=s["Normal"], fontSize=10, textColor=PDF_MUTED, spaceAfter=8, leading=13),
        "brand": ParagraphStyle("NLBrand", parent=s["Normal"], fontSize=10, textColor=PDF_PURPLE, fontName="Helvetica-Bold", spaceAfter=1),
    }

def _footer_handler(canvas_obj, doc_obj):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.setFillColor(PDF_PURPLE)
    canvas_obj.drawCentredString(A4[0] / 2, 20, "Nine Lab \u00b7 Your Placement Partner \u00b7 ninelab.in")
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.setFillColor(PDF_MUTED)
    canvas_obj.drawRightString(A4[0] - 72, 20, f"Page {doc_obj.page}")
    canvas_obj.restoreState()

def _no_footer(canvas_obj, doc_obj):
    pass

def _make_progress_bar(score, width=400, height=14):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PDF_GREY, strokeColor=None, rx=4, ry=4))
    fill_w = max(2, (score / 100) * width)
    d.add(Rect(0, 0, fill_w, height, fillColor=PDF_PURPLE, strokeColor=None, rx=4, ry=4))
    return d

def _colored_box(flowables, bg_color=PDF_LIGHT, border_color=None, left_border_color=None, col_width=None):
    w = col_width or 450
    t = Table([[flowables]], colWidths=[w])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if border_color:
        style_cmds.append(("BOX", (0, 0), (-1, -1), 1, border_color))
    if left_border_color:
        style_cmds.append(("LINEBEFORE", (0, 0), (0, -1), 3, left_border_color))
    t.setStyle(TableStyle(style_cmds))
    return t

def _render_lines(text, styles_dict, story_list):
    clean = strip_md(text)
    for line in clean.split('\n'):
        line = line.strip()
        if not line:
            story_list.append(Spacer(1, 3))
            continue
        is_header = (re.match(r'^[A-Z][A-Z\s/&]{3,}:', line) or
                     (line.isupper() and len(line) < 80 and len(line) > 3))
        st = styles_dict["h2"] if is_header else styles_dict["body"]
        try:
            story_list.append(Paragraph(safe_text(line), st))
        except Exception:
            story_list.append(Paragraph(safe_text(line[:200]), styles_dict["body"]))


def make_pdf_company_report(job_id: str, company: str, report_data: dict) -> str:
    _, st = _pdf_styles()
    filename = f"{job_id}_company.pdf"
    filepath = PDF_DIR / filename
    doc = SimpleDocTemplate(str(filepath), pagesize=A4, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=50)

    story = []
    text = strip_md(report_data.get("data", ""))

    # PAGE 1
    story.append(Paragraph("Nine Lab", st["brand"]))
    story.append(Paragraph(f"{safe_text(company)}", ParagraphStyle("CompH", fontName="Helvetica-Bold", fontSize=20, textColor=PDF_PURPLE, spaceAfter=4, leading=24)))
    story.append(Paragraph("Company Intelligence Report", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=PDF_PURPLE, spaceAfter=12))

    overview = extract_section(text, "COMPANY OVERVIEW") or f"Intelligence report for {company}."
    culture = (extract_section(text, "CULTURE AND ENVIRONMENT") or
               extract_section(text, "CULTURE AND WORK ENVIRONMENT") or
               extract_section(text, "CULTURE"))
    news = (extract_section(text, "RECENT DEVELOPMENTS") or
            extract_section(text, "RECENT NEWS") or
            extract_section(text, "NEWS"))

    story.append(Paragraph("What They Do", st["h2"]))
    for line in section_bullets(overview) or [overview]:
        story.append(Paragraph(safe_text(line), st["body"]))
    story.append(Spacer(1, 8))

    if culture:
        story.append(_colored_box([
            Paragraph("Culture &amp; Work Environment", st["h3"]),
            *[Paragraph(safe_text(b), st["body"]) for b in section_bullets(culture) or [culture]]
        ], bg_color=PDF_LIGHT))
        story.append(Spacer(1, 8))

    if news:
        story.append(_colored_box([
            Paragraph("Recent News", st["h3"]),
            *[Paragraph(safe_text(b), st["body"]) for b in section_bullets(news) or [news]]
        ], bg_color=PDF_LIGHT_AMBER))
    story.append(PageBreak())

    # PAGE 2
    story.append(Paragraph("Interview Intelligence", st["h2"]))
    story.append(Spacer(1, 4))

    interview = extract_section(text, "INTERVIEW PROCESS")
    testing = (extract_section(text, "WHAT THEY TEST IN EACH ROUND") or
               extract_section(text, "WHAT THEY TEST"))
    questions = (extract_section(text, "TOP 5 INTERVIEW QUESTIONS AT THIS COMPANY") or
                 extract_section(text, "TOP 5 INTERVIEW QUESTIONS") or
                 extract_section(text, "INTERVIEW QUESTIONS"))
    tech = extract_section(text, "TECH STACK THEY USE") or extract_section(text, "TECH STACK")

    if interview:
        story.append(Paragraph("Interview Stages", st["h3"]))
        for b in section_bullets(interview) or [interview]:
            story.append(Paragraph(safe_text(b), st["body"]))
        story.append(Spacer(1, 6))

    if testing:
        story.append(Paragraph("What They Test", st["h3"]))
        for b in section_bullets(testing) or [testing]:
            story.append(Paragraph(safe_text(b), st["body"]))
        story.append(Spacer(1, 6))

    if questions:
        story.append(_colored_box([
            Paragraph("Top Interview Questions", st["h3"]),
            *[Paragraph(safe_text(b), st["body"]) for b in section_bullets(questions) or [questions]]
        ], bg_color=PDF_LIGHT, border_color=PDF_PURPLE))
        story.append(Spacer(1, 8))

    if tech:
        story.append(Paragraph("Tech Stack", st["h3"]))
        for b in section_bullets(tech) or [tech]:
            story.append(Paragraph(safe_text(b), st["body"]))
    story.append(PageBreak())

    # PAGE 3
    salary = extract_section(text, "SALARY RANGE") or extract_section(text, "SALARY")
    green = extract_section(text, "GREEN FLAGS")
    red = extract_section(text, "RED FLAGS")
    advantage = (extract_section(text, "YOUR INTERVIEW ADVANTAGE") or
                 extract_section(text, "INTERVIEW ADVANTAGE"))

    if salary:
        story.append(Paragraph("Salary Range", st["h2"]))
        story.append(Paragraph(safe_text(salary.split('\n')[0] if salary else "Not available"), st["body"]))
        story.append(_make_progress_bar(65, width=400, height=14))
        story.append(Spacer(1, 12))

    green_items = section_bullets(green) if green else ["Positive work culture"]
    red_items = section_bullets(red) if red else ["High workload during deadlines"]

    left_col = [Paragraph("Green Flags", st["h3"])]
    for g in green_items[:5]:
        left_col.append(Paragraph(f"\u2713 {safe_text(g)}", st["body"]))

    right_col = [Paragraph("Red Flags", st["h3"])]
    for r in red_items[:5]:
        right_col.append(Paragraph(f"\u26a0 {safe_text(r)}", st["body"]))

    flags_table = Table([
        [_colored_box(left_col, bg_color=PDF_LIGHT_GREEN, left_border_color=PDF_GREEN, col_width=200),
         _colored_box(right_col, bg_color=PDF_LIGHT_RED, left_border_color=PDF_RED, col_width=200)]
    ], colWidths=[220, 220])
    flags_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(flags_table)
    story.append(Spacer(1, 12))

    if advantage:
        adv_items = section_bullets(advantage) or [advantage]
        story.append(_colored_box([
            Paragraph("Your Interview Advantage", st["h3"]),
            *[Paragraph(f"{i+1}. {safe_text(a)}", st["body"]) for i, a in enumerate(adv_items[:3])]
        ], bg_color=PDF_LIGHT, border_color=PDF_PURPLE))

    doc.build(story, onFirstPage=_footer_handler, onLaterPages=_footer_handler)
    return filename


def make_pdf_reality(job_id: str, company: str, analysis: dict, research: dict) -> str:
    _, st = _pdf_styles()
    filename = f"{job_id}_reality.pdf"
    filepath = PDF_DIR / filename
    doc = SimpleDocTemplate(str(filepath), pagesize=A4, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=50)

    story = []
    text = strip_md(analysis.get("data", ""))
    research_text = strip_md(research.get("data", ""))

    match_score = 50
    match_line = [l for l in text.split('\n') if 'MATCH SCORE' in l.upper() or '/100' in l]
    if match_line:
        nums = re.findall(r'\d+', match_line[0])
        if nums:
            match_score = min(int(nums[0]), 100)

    # PAGE 1 - EXECUTIVE SUMMARY
    story.append(Paragraph("Nine Lab", st["brand"]))
    story.append(Paragraph("Reality Report", st["title"]))
    story.append(Paragraph(f"{safe_text(company)} \u00b7 {datetime.now().strftime('%d %b %Y')}", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=PDF_PURPLE, spaceAfter=10))

    story.append(Paragraph("Match Score", st["h3"]))
    story.append(_make_progress_bar(match_score, width=400))
    story.append(Spacer(1, 4))

    if match_score >= 70:
        verdict = f"You are {match_score}% aligned with this role. You are a strong candidate!"
    elif match_score >= 50:
        verdict = f"You are {match_score}% there. With focused prep, you can absolutely crack this!"
    else:
        verdict = f"You are {match_score}% aligned. These gaps are completely fixable with the right plan."
    story.append(Paragraph(safe_text(verdict), ParagraphStyle("Verdict", fontName="Helvetica-Bold", fontSize=11, textColor=PDF_DARK, spaceAfter=12, leading=14)))
    story.append(Spacer(1, 6))

    strengths_text = (extract_section(text, "YOUR TOP 3 STRENGTHS") or
                      extract_section(text, "TOP 3 STRENGTHS"))
    gaps_text = (extract_section(text, "YOUR TOP 3 PRIORITY GAPS") or
                 extract_section(text, "TOP 3 PRIORITY GAPS") or
                 extract_section(text, "PRIORITY GAPS"))

    s_bullets = section_bullets(strengths_text)[:3] if strengths_text else ["Strong technical foundation"]
    g_bullets = section_bullets(gaps_text)[:3] if gaps_text else ["Areas for improvement identified"]

    left = [Paragraph("Your Strengths", st["h3"])] + [Paragraph(f"\u2713 {safe_text(s)}", st["body"]) for s in s_bullets]
    right = [Paragraph("Priority Gaps", st["h3"])] + [Paragraph(f"{i+1}. {safe_text(g)}", st["body"]) for i, g in enumerate(g_bullets)]

    t = Table([
        [_colored_box(left, bg_color=PDF_LIGHT_GREEN, left_border_color=PDF_GREEN, col_width=200),
         _colored_box(right, bg_color=PDF_LIGHT_RED, left_border_color=PDF_RED, col_width=200)]
    ], colWidths=[220, 220])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    story.append(PageBreak())

    # PAGE 2 - DEEP DIVE
    story.append(Paragraph("Deep Dive Analysis", st["h2"]))
    story.append(Spacer(1, 4))

    det_strengths = (extract_section(text, "DETAILED STRENGTHS ANALYSIS") or
                     extract_section(text, "DETAILED STRENGTHS") or strengths_text or "")
    det_gaps = (extract_section(text, "GAPS WITH PRIORITY AND SPECIFIC FIX") or
                extract_section(text, "DETAILED GAPS") or gaps_text or "")
    red_flags = (extract_section(text, "RESUME RED FLAGS WITH EXACT FIXES") or
                 extract_section(text, "RESUME RED FLAGS") or "")
    co_overview = (extract_section(text, "ABOUT THE COMPANY") or
                   extract_section(research_text, "COMPANY OVERVIEW") or
                   extract_section(text, "COMPANY OVERVIEW") or f"Analysis for {company}.")
    what_look = (extract_section(text, "WHAT THIS COMPANY LOOKS FOR") or
                 extract_section(research_text, "WHAT THEY LOOK FOR") or
                 extract_section(text, "WHAT THEY LOOK FOR") or "")
    interview_proc = (extract_section(text, "THEIR INTERVIEW PROCESS") or
                      extract_section(research_text, "INTERVIEW PROCESS") or "")
    salary = extract_section(text, "SALARY RANGE") or extract_section(research_text, "SALARY RANGE") or ""

    left_col = [Paragraph("About You", ParagraphStyle("LH", fontName="Helvetica-Bold", fontSize=12, textColor=PDF_PURPLE, spaceAfter=6, leading=15))]
    left_col.append(Paragraph("Resume Strengths", st["h3"]))
    for b in section_bullets(det_strengths)[:5] or ["Strong fundamentals shown"]:
        left_col.append(Paragraph(f"\u2022 {safe_text(b)}", st["small"]))
    left_col.append(Spacer(1, 6))
    left_col.append(Paragraph("Gaps (Priority Order)", st["h3"]))
    for i, b in enumerate(section_bullets(det_gaps)[:4] or ["Review needed"], 1):
        left_col.append(Paragraph(f"{i}. {safe_text(b)}", st["small"]))
    left_col.append(Spacer(1, 6))
    if red_flags:
        left_col.append(Paragraph("Red Flags to Fix", st["h3"]))
        for b in section_bullets(red_flags)[:3]:
            left_col.append(Paragraph(f"\u26a0 {safe_text(b)}", st["small"]))

    right_col = [Paragraph("About The Company", ParagraphStyle("RH", fontName="Helvetica-Bold", fontSize=12, textColor=PDF_PURPLE, spaceAfter=6, leading=15))]
    right_col.append(Paragraph("Overview", st["h3"]))
    for b in section_bullets(co_overview)[:3] or [co_overview[:200]]:
        right_col.append(Paragraph(safe_text(b), st["small"]))
    right_col.append(Spacer(1, 6))
    if what_look:
        right_col.append(Paragraph("What They Value", st["h3"]))
        for b in section_bullets(what_look)[:5]:
            right_col.append(Paragraph(f"\u2022 {safe_text(b)}", st["small"]))
        right_col.append(Spacer(1, 6))
    if interview_proc:
        right_col.append(Paragraph("Interview Process", st["h3"]))
        for i, b in enumerate(section_bullets(interview_proc)[:5], 1):
            right_col.append(Paragraph(f"{i}. {safe_text(b)}", st["small"]))
    if salary:
        right_col.append(Spacer(1, 6))
        right_col.append(Paragraph(f"Salary: {safe_text(salary.split(chr(10))[0])}", st["small"]))

    deep_t = Table([[left_col, right_col]], colWidths=[220, 220])
    deep_t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (1, 0), (1, 0), 0.5, PDF_GREY),
    ]))
    story.append(deep_t)
    story.append(PageBreak())

    # PAGE 3 - ACTION PLAN
    story.append(Paragraph("Your Action Plan", st["h2"]))
    story.append(Spacer(1, 4))

    priority_actions = (extract_section(text, "YOUR PRIORITY ACTION LIST") or
                        extract_section(text, "PRIORITY ACTIONS") or
                        extract_section(text, "PRIORITY ACTION"))
    next_steps = extract_section(text, "NEXT STEPS CHECKLIST") or extract_section(text, "NEXT STEPS")
    closing = extract_section(text, "CLOSING MESSAGE") or extract_section(text, "CLOSING")

    story.append(Paragraph("Priority Fix List", st["h3"]))
    for i, b in enumerate(section_bullets(priority_actions)[:3] if priority_actions else ["Review gaps and create study plan"], 1):
        story.append(Paragraph(f"<b>{i}.</b> {safe_text(b)}", st["body"]))
    story.append(Spacer(1, 10))

    if next_steps:
        story.append(Paragraph("Next Steps", st["h3"]))
        for b in section_bullets(next_steps)[:5]:
            story.append(Paragraph(f"\u25a1 {safe_text(b)}", st["body"]))
        story.append(Spacer(1, 10))

    closing_msg = closing or f"You have real strengths for this role at {company}. Every gap here is fixable. Focus on the priorities, put in the work, and you will be ready. You have got this!"
    story.append(_colored_box([
        Paragraph(safe_text(closing_msg.split('\n')[0] if closing_msg else "You've got this!"), ParagraphStyle("Close", fontName="Helvetica-Bold", fontSize=10.5, textColor=PDF_DARK, leading=14, alignment=1))
    ], bg_color=PDF_LIGHT, border_color=PDF_PURPLE))

    doc.build(story, onFirstPage=_footer_handler, onLaterPages=_footer_handler)
    return filename


def make_pdf_plan(job_id: str, company: str, plan: dict) -> str:
    _, st = _pdf_styles()
    filename = f"{job_id}_plan.pdf"
    filepath = PDF_DIR / filename
    doc = SimpleDocTemplate(str(filepath), pagesize=A4, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=50)

    story = []
    text = strip_md(plan.get("data", ""))

    # PAGE 1
    story.append(Paragraph("Nine Lab", st["brand"]))
    story.append(Paragraph("Prep Plan", st["title"]))
    story.append(Paragraph(f"{safe_text(company)} \u00b7 {datetime.now().strftime('%d %b %Y')}", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=PDF_PURPLE, spaceAfter=10))

    current = (extract_section(text, "CURRENT LEVEL ASSESSMENT") or
               extract_section(text, "CURRENT LEVEL"))
    p1 = (extract_section(text, "PRIORITY 1") or
          extract_section(text, "PRIORITY 1 CRITICAL"))
    p2 = (extract_section(text, "PRIORITY 2") or
          extract_section(text, "PRIORITY 2 IMPORTANT"))
    p3 = (extract_section(text, "PRIORITY 3") or
          extract_section(text, "PRIORITY 3 GOOD TO HAVE"))

    if current:
        story.append(_colored_box([
            Paragraph("Current Level Assessment", st["h3"]),
            Paragraph(safe_text(current.split('\n')[0]), st["body"])
        ], bg_color=HexColor("#F3F4F6")))
        story.append(Spacer(1, 8))

    for label, content, color in [("Priority 1: Critical", p1, PDF_RED), ("Priority 2: Important", p2, PDF_AMBER), ("Priority 3: Good to Have", p3, PDF_GREEN)]:
        if content:
            items = [Paragraph(f"<b>{label}</b>", st["h3"])]
            for b in section_bullets(content)[:3]:
                items.append(Paragraph(safe_text(b), st["body"]))
            story.append(_colored_box(items, bg_color=PDF_WHITE, left_border_color=color))
            story.append(Spacer(1, 6))
    story.append(PageBreak())

    # PAGE 2 - PHASES
    story.append(Paragraph("Your Learning Phases", st["h2"]))
    story.append(Spacer(1, 4))

    for phase_name, phase_key, color in [
        ("Phase 1: Foundation", "PHASE 1: FOUNDATION", PDF_PURPLE),
        ("Phase 2: Core Skills", "PHASE 2: CORE SKILLS", PDF_PURPLE),
        ("Phase 3: Company-Specific Polish", "PHASE 3: COMPANY-SPECIFIC POLISH", PDF_PURPLE),
    ]:
        phase_text = (extract_section(text, phase_key) or
                      extract_section(text, phase_key.split(":")[0]))
        if phase_text:
            story.append(_colored_box([
                Paragraph(phase_name, st["h3"]),
                *[Paragraph(safe_text(b), st["small"]) for b in section_bullets(phase_text)[:6]]
            ], bg_color=PDF_LIGHT, left_border_color=color))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(phase_name, st["h3"]))
            story.append(Paragraph("See full plan details above.", st["body"]))
            story.append(Spacer(1, 6))
    story.append(PageBreak())

    # PAGE 3 - Q&A + Resources
    tech_q = (extract_section(text, "TOP 10 TECHNICAL INTERVIEW QUESTIONS WITH ANSWERS") or
              extract_section(text, "TOP 10 TECHNICAL INTERVIEW QUESTIONS") or
              extract_section(text, "TECHNICAL QUESTIONS"))
    hr_q = (extract_section(text, "TOP 5 HR QUESTIONS WITH ANSWERS") or
            extract_section(text, "TOP 5 HR QUESTIONS") or
            extract_section(text, "HR QUESTIONS"))
    resources = extract_section(text, "FREE RESOURCES") or extract_section(text, "RESOURCES")
    checklist = extract_section(text, "INTERVIEW DAY CHECKLIST") or extract_section(text, "CHECKLIST")

    if tech_q:
        story.append(Paragraph("Technical Interview Questions", st["h2"]))
        for i, b in enumerate(section_bullets(tech_q)[:10], 1):
            bg = HexColor("#F9FAFB") if i % 2 == 0 else PDF_WHITE
            story.append(_colored_box([Paragraph(f"{i}. {safe_text(b)}", st["small"])], bg_color=bg))
        story.append(Spacer(1, 8))

    if hr_q:
        story.append(Paragraph("HR Interview Questions", st["h2"]))
        for i, b in enumerate(section_bullets(hr_q)[:5], 1):
            bg = HexColor("#F9FAFB") if i % 2 == 0 else PDF_WHITE
            story.append(_colored_box([Paragraph(f"{i}. {safe_text(b)}", st["small"])], bg_color=bg))
        story.append(Spacer(1, 8))

    if resources:
        story.append(Paragraph("Free Resources", st["h2"]))
        for b in section_bullets(resources)[:8]:
            story.append(Paragraph(f"\u2022 {safe_text(b)}", st["body"]))
        story.append(Spacer(1, 8))

    if checklist:
        story.append(Paragraph("Interview Day Checklist", st["h2"]))
        for b in section_bullets(checklist)[:10]:
            story.append(Paragraph(f"\u25a1 {safe_text(b)}", st["body"]))

    # Fallback: if no sections parsed, render raw text
    if not tech_q and not hr_q and not resources:
        _render_lines(text, st, story)

    doc.build(story, onFirstPage=_footer_handler, onLaterPages=_footer_handler)
    return filename


def make_pdf_resume(job_id: str, company: str, resume_data: dict) -> str:
    _, st = _pdf_styles()
    filename = f"{job_id}_resume.pdf"
    filepath = PDF_DIR / filename
    doc = SimpleDocTemplate(str(filepath), pagesize=A4, leftMargin=72, rightMargin=72, topMargin=60, bottomMargin=40)

    story = []
    content = strip_md(resume_data.get("data", ""))
    lines = content.split('\n')

    name_style = ParagraphStyle("RName", fontName="Helvetica-Bold", fontSize=20, textColor=PDF_DARK, spaceAfter=2, leading=24)
    jobtitle_style = ParagraphStyle("RJob", fontName="Helvetica", fontSize=12, textColor=PDF_PURPLE, spaceAfter=2, leading=15)
    contact_style = ParagraphStyle("RContact", fontName="Helvetica", fontSize=9, textColor=PDF_MUTED, spaceAfter=6, leading=12)
    section_style = ParagraphStyle("RSec", fontName="Helvetica-Bold", fontSize=12, textColor=PDF_PURPLE, spaceAfter=3, spaceBefore=8, leading=15)
    rbody_style = ParagraphStyle("RBody", fontName="Helvetica", fontSize=10.5, textColor=PDF_DARK, spaceAfter=3, leading=14)
    rbullet_style = ParagraphStyle("RBullet", fontName="Helvetica", fontSize=10.5, textColor=PDF_DARK, spaceAfter=2, leading=14, leftIndent=12)

    section_keywords = ["CONTACT", "SUMMARY", "PROFESSIONAL SUMMARY", "SKILLS", "TECHNICAL SKILLS",
                        "EXPERIENCE", "WORK EXPERIENCE", "PROJECTS", "EDUCATION", "CERTIFICATIONS",
                        "ACHIEVEMENTS", "AWARDS", "NAME", "JOB TITLE"]

    name_found = False
    for line in lines:
        ls = line.strip()
        if not ls:
            story.append(Spacer(1, 3))
            continue

        ls_clean = ls
        for prefix in ["NAME:", "JOB TITLE:", "CONTACT:"]:
            if ls.upper().startswith(prefix):
                ls_clean = ls[len(prefix):].strip()
                break

        if not name_found and (ls.upper().startswith("NAME:") or (not any(kw in ls.upper() for kw in section_keywords[3:]) and len(ls) < 60 and not ls.startswith(("\u2022", "-", "*")))):
            story.append(Paragraph(safe_text(ls_clean), name_style))
            name_found = True
            continue

        if ls.upper().startswith("JOB TITLE:"):
            story.append(Paragraph(safe_text(ls_clean), jobtitle_style))
            continue

        if ls.upper().startswith("CONTACT:") or ("|" in ls and "@" in ls):
            story.append(Paragraph(safe_text(ls_clean), contact_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=PDF_PURPLE, spaceAfter=6))
            continue

        is_section = any(ls.upper().startswith(kw) for kw in section_keywords[3:])
        is_bullet = ls.startswith(("\u2022", "-", "*"))

        if is_section:
            clean_label = ls
            for kw in section_keywords[3:]:
                if ls.upper().startswith(kw):
                    clean_label = ls[:len(kw)]
                    break
            story.append(Paragraph(safe_text(clean_label), section_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=PDF_GREY, spaceAfter=3))
            remainder = ls[len(clean_label):].strip(': ')
            if remainder:
                story.append(Paragraph(safe_text(remainder), rbody_style))
        elif is_bullet:
            story.append(Paragraph(safe_text(ls), rbullet_style))
        else:
            story.append(Paragraph(safe_text(ls), rbody_style))

    doc.build(story, onFirstPage=_no_footer, onLaterPages=_no_footer)
    return filename

# ── Pipeline runner ──────────────────────────────────────────────────────────

def run_pipeline(job_id: str, resume: str, jd: str, company: str):
    def update(stage: str, pct: int, msg: str):
        jobs[job_id].update({"stage": stage, "progress": pct, "message": msg})

    try:
        # ── Stage 1: Tavily web research (fast, no AI) ────────────────────────
        update("research", 5, "Searching the web for company intelligence...")
        research_result = agent_research(company, jd)
        research_snippets = research_result.get("data", "")

        # ── Stage 2: ARIA analysis (needs research snippets) ─────────────────
        update("analysis", 20, "ARIA is analyzing your resume against the JD...")
        analysis_result = agent_analysis(resume, jd, company, research=research_snippets)
        analysis_text = analysis_result.get("data", "")

        # Extract structured data from ARIA output for downstream agents
        match_score = _extract_match_score(analysis_text)
        gap_summary = _extract_gap_summary(analysis_text)

        update("analysis", 40, f"Match score: {match_score}%. Building your prep plan and company report...")

        # ── Stage 3: ATLAS + FORGE + NEXUS in parallel ───────────────────────
        loop = asyncio.new_event_loop()

        async def parallel_stage3():
            # ATLAS: prep plan (needs match_score from ARIA)
            p_task = loop.run_in_executor(
                executor, agent_plan, resume, jd, company, analysis_text, research_snippets, match_score
            )
            await asyncio.sleep(2)  # stagger to stay within Groq RPM
            # FORGE: resume rewrite (needs gap_summary and company values from ARIA)
            r_task = loop.run_in_executor(
                executor, agent_resume, resume, jd, company, research_snippets, gap_summary
            )
            await asyncio.sleep(2)
            # NEXUS: company report (needs research snippets)
            c_task = loop.run_in_executor(
                executor, agent_company_report, company, jd, research_snippets
            )
            return await asyncio.gather(p_task, r_task, c_task)

        plan_result, resume_result, company_report_result = loop.run_until_complete(parallel_stage3())
        loop.close()

        # ── Stage 4: Generate 4 PDFs ─────────────────────────────────────────
        update("pdf", 65, "Generating Company Report PDF...")
        company_file = make_pdf_company_report(job_id, company, company_report_result)

        update("pdf", 75, "Generating Reality Report PDF...")
        reality_file = make_pdf_reality(job_id, company, analysis_result, research_result)

        update("pdf", 85, "Generating Prep Plan PDF...")
        plan_file = make_pdf_plan(job_id, company, plan_result)

        update("pdf", 92, "Generating Revised Resume PDF...")
        resume_file = make_pdf_resume(job_id, company, resume_result)

        jobs[job_id].update({
            "stage": "done",
            "progress": 100,
            "message": "Your placement kit is ready! Download your 4 PDFs below.",
            "files": {
                "company": company_file,
                "reality": reality_file,
                "plan": plan_file,
                "resume": resume_file,
            }
        })

    except Exception as e:
        jobs[job_id].update({
            "stage": "error",
            "progress": 0,
            "message": f"Error: {str(e)[:200]}. Check your API keys and try again.",
        })

# ── Request/Response models ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    resume: str
    jd: str
    company: str

class AuthRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

# ── Routes ───────────────────────────────────────────────────────────────────

# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/ninelab/auth/register")
async def auth_register(req: AuthRequest):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(503, detail="Auth not configured. Set SUPABASE_URL and SUPABASE_KEY.")
    if not req.email or "@" not in req.email:
        raise HTTPException(400, detail="Please enter a valid email address.")
    if len(req.password) < 6:
        raise HTTPException(400, detail="Password must be at least 6 characters.")

    import httpx
    # Use admin endpoint with service key to auto-confirm email
    admin_headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "email": req.email,
        "password": req.password,
        "email_confirm": True,
        "user_metadata": {"full_name": req.full_name or ""}
    }
    try:
        r = httpx.post(f"{SUPABASE_URL}/auth/v1/admin/users",
                        headers=admin_headers, json=payload, timeout=10)
        data = r.json()
    except Exception as e:
        raise HTTPException(500, detail=f"Registration failed: {str(e)[:100]}")

    if r.status_code == 200 and data.get("id"):
        # Now log the user in to get a token
        login_payload = {"email": req.email, "password": req.password}
        anon_headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        lr = httpx.post(f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
                         headers=anon_headers, json=login_payload, timeout=10)
        ldata = lr.json()
        if lr.status_code == 200 and ldata.get("access_token"):
            user = ldata.get("user") or {}
            return JSONResponse({
                "success": True,
                "access_token": ldata["access_token"],
                "refresh_token": ldata.get("refresh_token", ""),
                "user": {
                    "id": user.get("id"),
                    "email": user.get("email"),
                    "full_name": (user.get("user_metadata") or {}).get("full_name", ""),
                },
            })
        raise HTTPException(500, detail="Account created but login failed. Please try logging in.")
    elif r.status_code == 422:
        msg = data.get("msg") or data.get("message") or ""
        if "already" in str(msg).lower() or "exists" in str(msg).lower():
            raise HTTPException(409, detail="An account with this email already exists. Please log in.")
        raise HTTPException(400, detail=msg or "Registration failed.")
    else:
        msg = data.get("msg") or data.get("message") or data.get("error", "") or "Registration failed."
        if "already" in str(msg).lower():
            raise HTTPException(409, detail="An account with this email already exists. Please log in.")
        raise HTTPException(r.status_code if r.status_code >= 400 else 500, detail=str(msg)[:200])


@app.post("/ninelab/auth/login")
async def auth_login(req: AuthRequest):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(503, detail="Auth not configured.")
    if not req.email or not req.password:
        raise HTTPException(400, detail="Email and password are required.")

    payload = {"email": req.email, "password": req.password}
    result = supabase_auth_request("POST", "/token?grant_type=password", payload=payload)
    data = result["data"]

    if result["status"] == 200 and data.get("access_token"):
        user = data.get("user") or {}
        return JSONResponse({
            "success": True,
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "user": {
                "id": user.get("id"),
                "email": user.get("email"),
                "full_name": (user.get("user_metadata") or {}).get("full_name", ""),
            },
        })
    elif result["status"] in (400, 401, 422):
        raise HTTPException(401, detail="Invalid email or password.")
    else:
        raise HTTPException(500, detail="Login failed. Please try again.")


@app.post("/ninelab/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    if token:
        supabase_auth_request("POST", "/logout", token=token)
    return JSONResponse({"success": True})


@app.get("/ninelab/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    if not token:
        raise HTTPException(401, detail="Not authenticated.")
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(401, detail="Session expired. Please log in again.")
    return JSONResponse({
        "id": user.get("id"),
        "email": user.get("email"),
        "full_name": (user.get("user_metadata") or {}).get("full_name", ""),
    })


@app.post("/ninelab/auth/refresh")
async def auth_refresh(request: Request):
    body = await request.json()
    refresh_token = body.get("refresh_token", "")
    if not refresh_token:
        raise HTTPException(400, detail="Refresh token required.")
    result = supabase_auth_request("POST", "/token?grant_type=refresh_token",
                                    payload={"refresh_token": refresh_token})
    data = result["data"]
    if result["status"] == 200 and data.get("access_token"):
        user = data.get("user") or {}
        return JSONResponse({
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", refresh_token),
            "user": {
                "id": user.get("id"),
                "email": user.get("email"),
                "full_name": (user.get("user_metadata") or {}).get("full_name", ""),
            },
        })
    raise HTTPException(401, detail="Session expired. Please log in again.")


class LinkedInImportRequest(BaseModel):
    url: str

@app.post("/ninelab/import-linkedin")
async def import_linkedin(req: LinkedInImportRequest):
    url = req.url.strip()
    if "linkedin.com" not in url.lower():
        raise HTTPException(400, detail="Please provide a valid LinkedIn profile URL.")
    if not TAVILY_API_KEY:
        raise HTTPException(400, detail="TAVILY_API_KEY not configured.")

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        raw_content = ""

        # Try direct extraction first
        try:
            extract_result = client.extract(urls=[url])
            results = extract_result.get("results", [])
            if results and results[0].get("raw_content"):
                raw_content = results[0]["raw_content"][:8000]
        except Exception:
            pass

        # Fallback: search for the profile
        if not raw_content or len(raw_content) < 100:
            search_results = client.search(
                f"site:linkedin.com {url}",
                max_results=3,
                include_raw_content=True
            )
            for r in search_results.get("results", []):
                content = r.get("raw_content") or r.get("content", "")
                if content and len(content) > len(raw_content):
                    raw_content = content[:8000]

        if not raw_content or len(raw_content) < 50:
            raise HTTPException(422, detail="Could not fetch LinkedIn profile. Make sure the profile is public and the URL is correct.")

        prompt = f"""You are a resume writer. Extract and format the following LinkedIn profile content into a clean, professional resume text.

LinkedIn Profile Content:
{raw_content}

Format the output as a proper resume with these sections (include only sections that have data):
- Name and Contact Info (if available)
- Professional Summary (2-3 lines)
- Work Experience (company, role, duration, key achievements)
- Education
- Skills
- Certifications / Projects (if available)

Be concise, professional, and use action verbs. Do NOT add any information that is not in the source content. Output only the resume text, no commentary."""

        resume_text = gemini_call(prompt)
        return JSONResponse({"text": resume_text.strip()})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to import LinkedIn profile: {str(e)}")


@app.post("/ninelab/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are accepted. Please upload a .pdf file.")
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(400, detail="File is too large. Maximum size is 10MB.")
    try:
        from pypdf import PdfReader
        import io
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(400, detail="File is too large. Maximum size is 10MB.")
        reader = PdfReader(io.BytesIO(contents))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise HTTPException(422, detail="Could not extract text from this PDF. Please ensure it contains selectable text (scanned/image-based PDFs are not supported).")
        word_count = len(full_text.split())
        return JSONResponse({"text": full_text, "pages": len(reader.pages), "words": word_count})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Error reading PDF: {str(e)}")


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
async def generate(req: GenerateRequest, request: Request,
                   authorization: Optional[str] = Header(None)):
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

    if not GEMINI_API_KEY and not GROQ_API_KEY:
        raise HTTPException(400, detail="No AI API key configured. Add GROQ_API_KEY or GEMINI_API_KEY.")
    if not TAVILY_API_KEY:
        raise HTTPException(400, detail="TAVILY_API_KEY not configured. Add it to your environment variables.")

    # Rate limiting: 3/day for logged-in users, 1/day for anonymous
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    auth_user_data = get_user_from_token(token) if token else None
    is_authenticated = bool(auth_user_data)

    if is_authenticated:
        user_id = auth_user_data["id"]
        today = date.today().isoformat()
        rec = user_daily_usage.get(user_id, {})
        if rec.get("date") == today and rec.get("count", 0) >= 3:
            raise HTTPException(429, detail="You've used all 3 of your daily generations. Come back tomorrow!")
    else:
        if not check_usage_limit(ip):
            raise HTTPException(429, detail=(
                "Daily limit reached. Please log in or create a free account for unlimited access."
            ))

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "stage": "queued",
        "progress": 0,
        "message": "⏳ Starting your placement pipeline...",
        "files": None,
        "created_at": time.time(),
    }

    if is_authenticated:
        today = date.today().isoformat()
        rec = user_daily_usage.get(user_id, {})
        if rec.get("date") == today:
            user_daily_usage[user_id]["count"] = rec.get("count", 0) + 1
        else:
            user_daily_usage[user_id] = {"date": today, "count": 1}
    else:
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
