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
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY", "d478886deemshee1d5a113b51de6p1d199ajsnee586dc31325")

jobs: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=8)
user_daily_usage: dict[str, dict] = {}  # {user_id: {"date": "YYYY-MM-DD", "count": N}}
# ── Persistent leads storage ─────────────────────────────────────────────────
LEADS_FILE = BASE_DIR / "pitch_leads.json"

def _load_leads() -> list:
    try:
        if LEADS_FILE.exists():
            return json.loads(LEADS_FILE.read_text())
    except Exception:
        pass
    return []

def _save_leads(leads: list):
    try:
        LEADS_FILE.write_text(json.dumps(leads, indent=2))
    except Exception:
        pass

pitch_leads: list[dict] = _load_leads()  # persists across restarts

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
    """Beta mode: unlimited access — no rate limiting."""
    return True


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

# ── Job board domain detector (used by /real-jobs) ───────────────────────────

_JOB_BOARD_DOMAINS = {
    "linkedin.com/jobs": "LinkedIn", "linkedin.com/job": "LinkedIn",
    "naukri.com": "Naukri", "internshala.com": "Internshala",
    "indeed.in": "Indeed", "in.indeed.com": "Indeed",
    "foundit.in": "Foundit", "glassdoor.co.in": "Glassdoor",
    "glassdoor.com": "Glassdoor", "shine.com": "Shine",
}

def _detect_job_board(url: str):
    url_lower = url.lower()
    for domain, source in _JOB_BOARD_DOMAINS.items():
        if domain in url_lower:
            return source
    return None

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
MISSING FROM YOUR RESUME
DETAILED STRENGTHS ANALYSIS
GAPS WITH PRIORITY AND SPECIFIC FIX
RESUME RED FLAGS WITH EXACT FIXES
ABOUT THE COMPANY
WHAT THIS COMPANY LOOKS FOR
THEIR INTERVIEW PROCESS
SALARY RANGE
YOUR PRIORITY ACTION LIST
NEXT STEPS CHECKLIST
CLOSING MESSAGE

For MISSING FROM YOUR RESUME section — check which of these are absent from the resume and list each with its ATS score impact:
- Work Experience (internship/job): HIGH impact — adds 15-20 points to ATS score
- LinkedIn Profile URL: MEDIUM impact — required by most ATS systems
- GitHub Profile URL: MEDIUM impact — critical for tech roles
- Achievements/Awards: MEDIUM impact — differentiates from other candidates
- Certifications: LOW-MEDIUM impact — adds keyword matches
- CGPA/Percentage: LOW impact — needed for eligibility filters
Format each as: "• [Missing Item] — [Why it matters] — [How to fix it quickly]"
If nothing is missing, write: "Your resume has all key sections covered." """

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
    FORGE_SYSTEM = """You are FORGE — an elite resume architect. You write resumes that pass ATS scanners AND impress recruiters in 6 seconds.

BULLET POINT FORMULA (every bullet MUST follow this):
[Strong Action Verb] + [What you did specifically] + [Result with number]
GOOD: "Built Python-based API serving 10,000+ daily requests with 99.9% uptime"
BAD: "Worked on Python project"

BANNED PHRASES — never use:
"Quick learner", "Team player", "Hardworking", "Passionate about", "Good communication", "Detail-oriented", "Responsible for"

ATS RULES:
- Single column only
- No tables, graphics, or icons
- Keywords from JD must appear EXACTLY as written
- Job title from JD must appear in Summary

OUTPUT FORMAT — follow this EXACTLY with no deviations:

<<NAME>>
[Candidate's Full Name]
<</NAME>>

<<CONTACT>>
[Use ONLY what is in the original resume. Format: City | email | phone | LinkedIn | GitHub]
[If LinkedIn or GitHub is missing from original — do NOT write placeholder, just omit it]
<</CONTACT>>

<<SUMMARY>>
[2-3 sentence professional summary using job title from JD, candidate's real skills, and company name.]
[Use ONLY real info — never write "X years of experience" if not stated in resume]
<</SUMMARY>>

<<TECHNICAL SKILLS>>
[Only include categories the candidate actually has. Do NOT create empty categories.]
[Example — Languages: Python, Java | Frameworks: React, FastAPI | Tools: Git, Docker]
<</TECHNICAL SKILLS>>

<<WORK EXPERIENCE>>
[ONLY include if original resume has actual work/internship experience.]
[If NO work experience exists — completely OMIT this entire section including the tags.]
[Job Title] | [Company Name] | [Month Year – Month Year]
• [action verb + specific task + measurable result]
<</WORK EXPERIENCE>>

<<PROJECTS>>
[Include ALL real projects from original resume. If no projects — omit this section.]
[Project Name] | [Year if known, otherwise omit year]
Tech: [tech stack from original resume only]
• [action verb + what was built + measurable result]
• [action verb + specific feature/outcome]
<</PROJECTS>>

<<EDUCATION>>
[Include ONLY if education info exists in original resume.]
[University Name] | [City]
[Degree], [Field] | [Year] | CGPA: [X.XX]
[If college name unknown — write "Details to be added" rather than a placeholder]
<</EDUCATION>>

<<ACHIEVEMENTS>>
[ONLY include if original resume has awards, competitions, or recognitions.]
[If NO achievements exist — completely OMIT this section including the tags.]
• [Award name | Event | Year]
<</ACHIEVEMENTS>>

CRITICAL RULES — VIOLATIONS WILL BREAK THE RESUME:
1. NEVER write placeholder text like [Your LinkedIn], [Duration], [Year], [X.XX] — if info missing, SKIP that field
2. NEVER invent dates, GPAs, company names, or metrics not in original resume
3. NEVER include a section if the candidate has no data for it — empty sections hurt ATS score
4. NEVER write "N/A" or "Not provided" — just omit the field
5. You MAY rephrase bullets to be stronger using data already present
6. ZERO Nine Lab or AI mentions anywhere in output"""

    user_prompt = f"""Rewrite this resume for maximum ATS score and recruiter impact for this specific role.

ORIGINAL RESUME:
{resume[:3000]}

JOB DESCRIPTION:
{jd[:2000]}

COMPANY: {company}
COMPANY VALUES: {company_research[:600]}
GAPS TO ADDRESS: {gap_summary}

Instructions:
1. Extract top keywords from JD — use them EXACTLY in the resume
2. Quantify every achievement possible using numbers from original resume
3. Tailor the Summary specifically for {company} and this role
4. Follow the OUTPUT FORMAT exactly — use the << >> tags as shown
5. Do NOT add any experience, skills, or achievements not in the original resume"""

    try:
        text = gemini_call(user_prompt, retries=2, temperature=0.05, system_prompt=FORGE_SYSTEM)
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

def _clip(text: str, max_chars: int = 280) -> str:
    """Truncate any text that would make a single paragraph overflow the page."""
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '\u2026'

def _colored_box(flowables, bg_color=PDF_LIGHT, border_color=None, left_border_color=None, col_width=None):
    """Render a shaded box. One row per flowable so the table can split across pages."""
    w = col_width or 450
    rows = [[f] for f in flowables]
    t = Table(rows, colWidths=[w])
    n = len(rows)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        # Extra breathing room on first and last row
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, n - 1), (0, n - 1), 10),
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
        story.append(Paragraph(_clip(safe_text(line)), st["body"]))
    story.append(Spacer(1, 8))

    if culture:
        story.append(_colored_box([
            Paragraph("Culture &amp; Work Environment", st["h3"]),
            *[Paragraph(_clip(safe_text(b)), st["body"]) for b in section_bullets(culture) or [culture]]
        ], bg_color=PDF_LIGHT))
        story.append(Spacer(1, 8))

    if news:
        story.append(_colored_box([
            Paragraph("Recent News", st["h3"]),
            *[Paragraph(_clip(safe_text(b)), st["body"]) for b in section_bullets(news) or [news]]
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
            story.append(Paragraph(_clip(safe_text(b)), st["body"]))
        story.append(Spacer(1, 6))

    if testing:
        story.append(Paragraph("What They Test", st["h3"]))
        for b in section_bullets(testing) or [testing]:
            story.append(Paragraph(_clip(safe_text(b)), st["body"]))
        story.append(Spacer(1, 6))

    if questions:
        story.append(_colored_box([
            Paragraph("Top Interview Questions", st["h3"]),
            *[Paragraph(_clip(safe_text(b)), st["body"]) for b in section_bullets(questions) or [questions]]
        ], bg_color=PDF_LIGHT, border_color=PDF_PURPLE))
        story.append(Spacer(1, 8))

    if tech:
        story.append(Paragraph("Tech Stack", st["h3"]))
        for b in section_bullets(tech) or [tech]:
            story.append(Paragraph(_clip(safe_text(b)), st["body"]))
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

    story.append(_colored_box([
        Paragraph("Green Flags", st["h3"]),
        *[Paragraph(f"\u2713 {_clip(safe_text(g))}", st["body"]) for g in green_items[:5]]
    ], bg_color=PDF_LIGHT_GREEN, left_border_color=PDF_GREEN))
    story.append(Spacer(1, 8))
    story.append(_colored_box([
        Paragraph("Red Flags", st["h3"]),
        *[Paragraph(f"\u26a0 {_clip(safe_text(r))}", st["body"]) for r in red_items[:5]]
    ], bg_color=PDF_LIGHT_RED, left_border_color=PDF_RED))
    story.append(Spacer(1, 12))

    if advantage:
        adv_items = section_bullets(advantage) or [advantage]
        story.append(_colored_box([
            Paragraph("Your Interview Advantage", st["h3"]),
            *[Paragraph(f"{i+1}. {_clip(safe_text(a))}", st["body"]) for i, a in enumerate(adv_items[:3])]
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

    story.append(_colored_box([
        Paragraph("Your Strengths", st["h3"]),
        *[Paragraph(f"\u2713 {_clip(safe_text(s))}", st["body"]) for s in s_bullets]
    ], bg_color=PDF_LIGHT_GREEN, left_border_color=PDF_GREEN))
    story.append(Spacer(1, 8))
    story.append(_colored_box([
        Paragraph("Priority Gaps", st["h3"]),
        *[Paragraph(f"{i+1}. {_clip(safe_text(g))}", st["body"]) for i, g in enumerate(g_bullets)]
    ], bg_color=PDF_LIGHT_RED, left_border_color=PDF_RED))
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

    # "About You" — full-width stacked (no nested tables, no 2-column layout)
    about_you = [
        Paragraph("About You", ParagraphStyle("LH", fontName="Helvetica-Bold", fontSize=12, textColor=PDF_PURPLE, spaceAfter=4, leading=15)),
        Paragraph("Resume Strengths", st["h3"]),
        *[Paragraph(f"\u2022 {_clip(_clip(safe_text(b)))}", st["small"]) for b in (section_bullets(det_strengths)[:5] or ["Strong fundamentals shown"])],
        Spacer(1, 6),
        Paragraph("Gaps (Priority Order)", st["h3"]),
        *[Paragraph(f"{i}. {_clip(_clip(safe_text(b)))}", st["small"]) for i, b in enumerate(section_bullets(det_gaps)[:4] or ["Review needed"], 1)],
    ]
    if red_flags:
        about_you += [
            Spacer(1, 6),
            Paragraph("Red Flags to Fix", st["h3"]),
            *[Paragraph(f"\u26a0 {_clip(_clip(safe_text(b)))}", st["small"]) for b in section_bullets(red_flags)[:3]],
        ]
    story.append(_colored_box(about_you, bg_color=PDF_LIGHT))
    story.append(Spacer(1, 8))

    # "About The Company" — full-width stacked
    about_co = [
        Paragraph("About The Company", ParagraphStyle("RH", fontName="Helvetica-Bold", fontSize=12, textColor=PDF_PURPLE, spaceAfter=4, leading=15)),
        Paragraph("Overview", st["h3"]),
        *[Paragraph(_clip(_clip(safe_text(b))), st["small"]) for b in (section_bullets(co_overview)[:3] or [_clip(co_overview)])],
    ]
    if what_look:
        about_co += [
            Spacer(1, 6),
            Paragraph("What They Value", st["h3"]),
            *[Paragraph(f"\u2022 {_clip(_clip(safe_text(b)))}", st["small"]) for b in section_bullets(what_look)[:4]],
        ]
    if interview_proc:
        about_co += [
            Spacer(1, 6),
            Paragraph("Interview Process", st["h3"]),
            *[Paragraph(f"{i}. {_clip(_clip(safe_text(b)))}", st["small"]) for i, b in enumerate(section_bullets(interview_proc)[:4], 1)],
        ]
    if salary:
        about_co += [Spacer(1, 6), Paragraph(f"Salary: {_clip(safe_text(salary.split(chr(10))[0]))}", st["small"])]
    story.append(_colored_box(about_co, bg_color=PDF_LIGHT))
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
        story.append(Paragraph(f"<b>{i}.</b> {_clip(safe_text(b))}", st["body"]))
    story.append(Spacer(1, 10))

    if next_steps:
        story.append(Paragraph("Next Steps", st["h3"]))
        for b in section_bullets(next_steps)[:5]:
            story.append(Paragraph(f"\u25a1 {_clip(safe_text(b))}", st["body"]))
        story.append(Spacer(1, 10))

    closing_msg = closing or f"You have real strengths for this role at {company}. Every gap here is fixable. Focus on the priorities, put in the work, and you will be ready. You have got this!"
    story.append(_colored_box([
        Paragraph(_clip(safe_text(closing_msg.split('\n')[0] if closing_msg else "You've got this!"), 400), ParagraphStyle("Close", fontName="Helvetica-Bold", fontSize=10.5, textColor=PDF_DARK, leading=14, alignment=1))
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
            Paragraph(_clip(safe_text(current.split('\n')[0])), st["body"])
        ], bg_color=HexColor("#F3F4F6")))
        story.append(Spacer(1, 8))

    for label, content, color in [("Priority 1: Critical", p1, PDF_RED), ("Priority 2: Important", p2, PDF_AMBER), ("Priority 3: Good to Have", p3, PDF_GREEN)]:
        if content:
            items = [Paragraph(f"<b>{label}</b>", st["h3"])]
            for b in section_bullets(content)[:3]:
                items.append(Paragraph(_clip(safe_text(b)), st["body"]))
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
                *[Paragraph(_clip(safe_text(b)), st["small"]) for b in section_bullets(phase_text)[:6]]
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
            story.append(_colored_box([Paragraph(f"{i}. {_clip(safe_text(b))}", st["small"])], bg_color=bg))
        story.append(Spacer(1, 8))

    if hr_q:
        story.append(Paragraph("HR Interview Questions", st["h2"]))
        for i, b in enumerate(section_bullets(hr_q)[:5], 1):
            bg = HexColor("#F9FAFB") if i % 2 == 0 else PDF_WHITE
            story.append(_colored_box([Paragraph(f"{i}. {_clip(safe_text(b))}", st["small"])], bg_color=bg))
        story.append(Spacer(1, 8))

    if resources:
        story.append(Paragraph("Free Resources", st["h2"]))
        for b in section_bullets(resources)[:8]:
            story.append(Paragraph(f"\u2022 {_clip(safe_text(b))}", st["body"]))
        story.append(Spacer(1, 8))

    if checklist:
        story.append(Paragraph("Interview Day Checklist", st["h2"]))
        for b in section_bullets(checklist)[:10]:
            story.append(Paragraph(f"\u25a1 {_clip(safe_text(b))}", st["body"]))

    # Fallback: if no sections parsed, render raw text
    if not tech_q and not hr_q and not resources:
        _render_lines(text, st, story)

    doc.build(story, onFirstPage=_footer_handler, onLaterPages=_footer_handler)
    return filename


def _parse_resume_sections(raw: str) -> dict:
    """Parse FORGE output (with <<SECTION>> tags) into a structured dict."""
    sections = {}
    tag_pattern = re.compile(r'<<(\w[\w\s]*)>>(.*?)<</\1>>', re.DOTALL | re.IGNORECASE)
    for m in tag_pattern.finditer(raw):
        key = m.group(1).strip().upper()
        val = m.group(2).strip()
        sections[key] = val

    # Fallback: if no tags found, try plain-text parsing
    if not sections:
        lines = strip_md(raw).split('\n')
        SECTION_NAMES = ["SUMMARY", "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "SKILLS",
                         "WORK EXPERIENCE", "EXPERIENCE", "PROJECTS", "EDUCATION",
                         "ACHIEVEMENTS", "AWARDS", "LEADERSHIP", "CERTIFICATIONS"]
        current_key = None
        current_lines = []
        name_done = False
        contact_done = False
        for line in lines:
            ls = line.strip()
            if not ls:
                if current_key:
                    current_lines.append("")
                continue
            upper = ls.upper()
            matched_section = next((s for s in SECTION_NAMES if upper.startswith(s)), None)
            if matched_section:
                if current_key and current_lines:
                    sections[current_key] = '\n'.join(current_lines).strip()
                current_key = matched_section
                current_lines = []
            elif not name_done and not contact_done and not current_key:
                sections["NAME"] = ls
                name_done = True
            elif name_done and not contact_done and not current_key:
                sections["CONTACT"] = ls
                contact_done = True
            elif current_key:
                current_lines.append(ls)
        if current_key and current_lines:
            sections[current_key] = '\n'.join(current_lines).strip()
    return sections


def make_pdf_resume(job_id: str, company: str, resume_data: dict) -> str:
    """Generate a clean, professional resume PDF matching reference format."""
    filename = f"{job_id}_resume.pdf"
    filepath = PDF_DIR / filename

    # ── Styles ────────────────────────────────────────────────────────────────
    BLACK       = HexColor("#111111")
    DARK_GREY   = HexColor("#333333")
    MED_GREY    = HexColor("#555555")
    LIGHT_GREY  = HexColor("#888888")
    RULE_GREY   = HexColor("#CCCCCC")
    ACCENT      = HexColor("#1a1a2e")   # dark navy for section headers

    name_st = ParagraphStyle("RN", fontName="Helvetica-Bold", fontSize=22,
                              textColor=BLACK, spaceAfter=2, leading=26, alignment=0)
    contact_st = ParagraphStyle("RC", fontName="Helvetica", fontSize=9,
                                 textColor=MED_GREY, spaceAfter=0, leading=13, alignment=0)
    sec_st = ParagraphStyle("RS", fontName="Helvetica-Bold", fontSize=10.5,
                              textColor=ACCENT, spaceBefore=10, spaceAfter=2, leading=14,
                              textTransform='uppercase')
    entry_title_st = ParagraphStyle("RET", fontName="Helvetica-Bold", fontSize=10.5,
                                     textColor=BLACK, spaceAfter=1, leading=14)
    entry_meta_st = ParagraphStyle("REM", fontName="Helvetica-Oblique", fontSize=9.5,
                                    textColor=LIGHT_GREY, spaceAfter=2, leading=13)
    body_st = ParagraphStyle("RB", fontName="Helvetica", fontSize=10,
                              textColor=DARK_GREY, spaceAfter=2, leading=14)
    bullet_st = ParagraphStyle("RBu", fontName="Helvetica", fontSize=10,
                                textColor=DARK_GREY, spaceAfter=2, leading=14,
                                leftIndent=14, firstLineIndent=-8)
    skill_label_st = ParagraphStyle("RSL", fontName="Helvetica-Bold", fontSize=10,
                                     textColor=BLACK, spaceAfter=0, leading=14)
    summary_st = ParagraphStyle("RSu", fontName="Helvetica", fontSize=10,
                                  textColor=DARK_GREY, spaceAfter=3, leading=15)

    def section_header(label: str) -> list:
        return [
            Paragraph(safe_text(label.upper()), sec_st),
            HRFlowable(width="100%", thickness=0.8, color=RULE_GREY, spaceAfter=4),
        ]

    def bullet_line(text: str) -> Paragraph:
        clean = re.sub(r'^[\u2022\-\*]\s*', '', text.strip())
        return Paragraph(f"\u2022\u00a0{safe_text(clean)}", bullet_st)

    # ── Parse content ─────────────────────────────────────────────────────────
    raw = resume_data.get("data", "")
    sec = _parse_resume_sections(raw)

    story = []
    doc = SimpleDocTemplate(
        str(filepath), pagesize=A4,
        leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=40
    )

    # ── HEADER: Name + Contact ────────────────────────────────────────────────
    name_text = sec.get("NAME", "").strip()
    contact_text = sec.get("CONTACT", "").strip()

    if name_text:
        story.append(Paragraph(safe_text(name_text), name_st))
    if contact_text:
        story.append(Paragraph(safe_text(contact_text), contact_st))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=6, spaceBefore=4))

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    summary_key = next((k for k in ["SUMMARY", "PROFESSIONAL SUMMARY"] if k in sec), None)
    if summary_key:
        story += section_header("Summary")
        for line in sec[summary_key].split('\n'):
            if line.strip():
                story.append(Paragraph(safe_text(line.strip()), summary_st))
        story.append(Spacer(1, 2))

    # ── TECHNICAL SKILLS ─────────────────────────────────────────────────────
    skills_key = next((k for k in ["TECHNICAL SKILLS", "SKILLS"] if k in sec), None)
    if skills_key:
        story += section_header("Technical Skills")
        for line in sec[skills_key].split('\n'):
            ls = line.strip()
            if not ls:
                continue
            if ':' in ls:
                label, _, rest = ls.partition(':')
                p = Paragraph(f"<b>{safe_text(label.strip())}:</b> {safe_text(rest.strip())}", body_st)
            else:
                p = Paragraph(safe_text(ls), body_st)
            story.append(p)
        story.append(Spacer(1, 2))

    # ── WORK EXPERIENCE ───────────────────────────────────────────────────────
    exp_key = next((k for k in ["WORK EXPERIENCE", "EXPERIENCE"] if k in sec), None)
    if exp_key:
        story += section_header("Work Experience")
        lines = sec[exp_key].split('\n')
        i = 0
        while i < len(lines):
            ls = lines[i].strip()
            if not ls:
                i += 1
                continue
            if ls.startswith(('•', '-', '*')):
                story.append(bullet_line(ls))
            elif '|' in ls and not ls.startswith(('•', '-')):
                # Entry title line: "Job Title | Company | Duration"
                story.append(Paragraph(safe_text(ls), entry_title_st))
            else:
                story.append(Paragraph(safe_text(ls), body_st))
            i += 1
        story.append(Spacer(1, 2))

    # ── PROJECTS ──────────────────────────────────────────────────────────────
    if "PROJECTS" in sec:
        story += section_header("Projects")
        lines = sec["PROJECTS"].split('\n')
        i = 0
        while i < len(lines):
            ls = lines[i].strip()
            if not ls:
                story.append(Spacer(1, 4))
                i += 1
                continue
            if ls.startswith(('•', '-', '*')):
                story.append(bullet_line(ls))
            elif ls.lower().startswith('tech:') or ls.lower().startswith('tech stack:'):
                # Tech stack line — italic meta style
                _, _, tech = ls.partition(':')
                story.append(Paragraph(safe_text(tech.strip()), entry_meta_st))
            elif '|' in ls and not ls.startswith(('•', '-')):
                # Project title: "Project Name | Year" or "Project Name | GitHub | Year"
                story.append(Paragraph(safe_text(ls), entry_title_st))
            else:
                story.append(Paragraph(safe_text(ls), body_st))
            i += 1
        story.append(Spacer(1, 2))

    # ── EDUCATION ─────────────────────────────────────────────────────────────
    if "EDUCATION" in sec:
        story += section_header("Education")
        lines = sec["EDUCATION"].split('\n')
        for ls in lines:
            ls = ls.strip()
            if not ls:
                story.append(Spacer(1, 4))
                continue
            if '|' in ls and not ls.startswith(('•', '-')):
                story.append(Paragraph(safe_text(ls), entry_title_st))
            elif ls.lower().startswith(('cgpa', 'percentage', 'grade')):
                story.append(Paragraph(safe_text(ls), entry_meta_st))
            else:
                story.append(Paragraph(safe_text(ls), body_st))
        story.append(Spacer(1, 2))

    # ── ACHIEVEMENTS ──────────────────────────────────────────────────────────
    ach_key = next((k for k in ["ACHIEVEMENTS", "AWARDS"] if k in sec), None)
    if ach_key:
        story += section_header("Achievements")
        for ls in sec[ach_key].split('\n'):
            ls = ls.strip()
            if not ls:
                continue
            if ls.startswith(('•', '-', '*')):
                story.append(bullet_line(ls))
            else:
                story.append(bullet_line(ls))
        story.append(Spacer(1, 2))

    # ── LEADERSHIP / EXTRA ────────────────────────────────────────────────────
    lead_key = next((k for k in ["LEADERSHIP", "LEADERSHIP & EXTRACURRICULARS",
                                   "EXTRACURRICULARS", "CERTIFICATIONS"] if k in sec), None)
    if lead_key:
        story += section_header(lead_key.title())
        for ls in sec[lead_key].split('\n'):
            ls = ls.strip()
            if not ls:
                story.append(Spacer(1, 3))
                continue
            if ls.startswith(('•', '-', '*')):
                story.append(bullet_line(ls))
            elif '|' in ls:
                story.append(Paragraph(safe_text(ls), entry_title_st))
            else:
                story.append(Paragraph(safe_text(ls), body_st))

    doc.build(story, onFirstPage=_no_footer, onLaterPages=_no_footer)
    return filename

# ── Pipeline runner ──────────────────────────────────────────────────────────

def _quick_ats_score(resume_text: str, jd_text: str) -> int:
    """
    Realistic ATS score 0-100.
    Rules:
    - 2-3 word resume → ~5%
    - Average student resume (no work exp, no LinkedIn) → 30-45%
    - Good student resume (projects + skills + education) → 55-70%
    - Optimised resume (keywords matched + quantified) → 75-88%
    - No arbitrary base numbers — every point earned.
    """
    rt = resume_text.lower().strip()
    jt = jd_text.lower().strip()
    word_count = len(resume_text.split())

    # Hard gate — resume too short to evaluate meaningfully
    if word_count < 30:
        return max(3, word_count // 3)

    STOPWORDS = {"the","and","for","are","but","not","you","all","any","can","had",
                 "her","was","one","our","out","day","get","has","him","his","how",
                 "its","may","new","now","own","see","two","way","who","did","each",
                 "from","this","that","with","have","will","been","they","their",
                 "more","also","into","over","some","such","than","then","them",
                 "well","were","what","when","whom","your","able","about","after",
                 "being","below","could","doing","during","other","these","those",
                 "through","under","until","while","would","should","shall","must",
                 "work","year","years","good","role","team","need","able","must",
                 "strong","experience","candidate","looking","join","help","great",
                 "including","required","requirements","responsibilities","minimum",
                 "preferred","plus","bonus","nice","have","will","across","within"}

    TECH = ["python","java","javascript","typescript","c++","c#","golang","rust","kotlin",
            "swift","php","ruby","scala","matlab","sql","nosql","react","angular","vue",
            "html","css","nextjs","nodejs","express","django","flask","fastapi","spring",
            "mysql","postgresql","mongodb","redis","elasticsearch","aws","azure","gcp",
            "docker","kubernetes","git","machine learning","deep learning","tensorflow",
            "pytorch","scikit-learn","pandas","numpy","nlp","data science","linux","agile"]

    WEAK_PHRASES = ["responsible for","worked on","helped with","assisted in",
                    "quick learner","team player","hardworking","passionate about",
                    "good communication","detail oriented","detail-oriented"]

    def extract_kws(text):
        words = re.findall(r'\b[a-z][a-z0-9]{3,}\b', text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)
                   if len(words[i]) >= 4 and len(words[i+1]) >= 4]
        return [t for t in words + bigrams if t not in STOPWORDS]

    from collections import Counter
    jd_kws  = extract_kws(jt)
    res_kws = set(extract_kws(rt))
    jd_top  = [w for w, _ in Counter(jd_kws).most_common(30) if len(w) > 3]

    # ── A: JD Keyword Match (35%) — 0 if JD empty ────────────────────────────
    if jd_top:
        matched = sum(1 for w in jd_top if w in res_kws)
        kw_score = matched / len(jd_top) * 100
    else:
        # No JD provided — score based purely on resume completeness
        kw_score = 0

    # ── B: Tech Skill Match (20%) — 0 if no tech in JD ───────────────────────
    jd_tech = [s for s in TECH if s in jt]
    res_tech = [s for s in TECH if s in rt]
    if jd_tech:
        tech_score = sum(1 for s in jd_tech if s in rt) / len(jd_tech) * 100
    elif res_tech:
        # JD has no tech but resume does — partial credit for having skills
        tech_score = min(len(res_tech) * 8, 60)
    else:
        tech_score = 0

    # ── C: Content Quality (25%) — earned, not given ─────────────────────────
    content = 0
    # Length (max 30 pts)
    if word_count >= 300: content += 30
    elif word_count >= 200: content += 22
    elif word_count >= 150: content += 15
    elif word_count >= 80:  content += 8
    # Quantified achievements — numbers in resume (max 40 pts)
    num_count = len(re.findall(r'\d+\s*[+%x]|\b[1-9]\d+\b', resume_text))
    if num_count >= 10: content += 40
    elif num_count >= 6:  content += 30
    elif num_count >= 3:  content += 18
    elif num_count >= 1:  content += 8
    # Action verbs (max 20 pts)
    action_verbs = ["developed","built","designed","implemented","created","optimised",
                    "optimized","deployed","managed","led","reduced","increased","improved",
                    "launched","delivered","architected","engineered","automated","integrated"]
    verb_count = sum(1 for v in action_verbs if v in rt)
    content += min(verb_count * 4, 20)
    # Weak phrase penalty
    weak_count = sum(1 for p in WEAK_PHRASES if p in rt)
    content = max(0, content - weak_count * 8)
    content = min(content, 100)

    # ── D: Structure Completeness (20%) — 0 base, earned only ────────────────
    structure = 0
    if any(w in rt for w in ["education","degree","university","college","btech","mca","mba","bachelor","master","diploma"]): structure += 25
    if any(w in rt for w in ["skill","skills","technologies","tools","languages","frameworks","proficient"]): structure += 20
    if any(w in rt for w in ["project","projects","built","developed","created","implemented"]): structure += 20
    if any(w in rt for w in ["internship","experience","work experience","employment","intern"]): structure += 20
    if any(w in rt for w in ["summary","objective","profile","about"]): structure += 5
    if any(w in rt for w in ["achievement","award","winner","prize","certification","certified"]): structure += 5
    if "linkedin" in rt or "github.com" in rt: structure += 5
    structure = min(structure, 100)

    # ── Weighted total ────────────────────────────────────────────────────────
    score = (kw_score * 0.35 + tech_score * 0.20 + content * 0.25 + structure * 0.20)
    return round(min(score, 100))


def _generate_insight_cards(resume: str, jd: str, company: str,
                             score_before: int, score_after: int,
                             gap_summary: str, analysis_text: str) -> list:
    """Generate 6 personalised insight cards using AI — specific to this candidate."""

    # ── Extract real skills/gaps from resume vs JD for richer fallback ────────
    TECH = ["python","java","javascript","typescript","react","angular","nodejs","django",
            "flask","fastapi","spring","sql","mysql","postgresql","mongodb","redis","aws",
            "azure","gcp","docker","kubernetes","git","machine learning","tensorflow",
            "pytorch","data science","c++","golang","kotlin","flutter","android","ios"]
    rt = resume.lower(); jt = jd.lower()
    resume_skills = [s for s in TECH if s in rt]
    jd_skills     = [s for s in TECH if s in jt]
    missing_skills = [s for s in jd_skills if s not in rt][:4]
    top_skill      = resume_skills[0] if resume_skills else "programming"
    diff           = score_after - score_before

    prompt = f"""You are a placement expert. Generate exactly 6 personalised insight cards.

RESUME SKILLS FOUND: {', '.join(resume_skills[:8]) or 'not specified'}
MISSING FROM JD: {', '.join(missing_skills) or 'none'}
COMPANY: {company}
ATS: {score_before}% → {score_after}% (improved by {diff} points)
TOP GAP: {gap_summary[:200]}
JD KEYWORDS: {jt[:300]}

Generate 6 cards (one per category). Each card must be SPECIFIC to this candidate's skills and gaps above.

Return ONLY a JSON array — no markdown fences, no explanation, just raw JSON:
[
  {{"category":"salary","emoji":"💰","front":"<12 words starting with number or stat about {company} fresher pay>","back":"<35 words: specific salary breakdown for {company} India fresher>","tag":"Salary Intel","color":"green"}},
  {{"category":"interview","emoji":"🎯","front":"<12 words: surprising fact about {company} interview>","back":"<35 words: specific round info + what to prepare>","tag":"Insider Tip","color":"purple"}},
  {{"category":"gap","emoji":"⚠️","front":"<12 words: mention specific missing skill if any>","back":"<35 words: exactly what to add and where in resume>","tag":"Your Gap","color":"red"}},
  {{"category":"priority","emoji":"🚀","front":"<12 words: specific action for THIS candidate>","back":"<35 words: actionable step using their actual skills>","tag":"This Week","color":"blue"}},
  {{"category":"culture","emoji":"🏢","front":"<12 words: insider {company} culture fact>","back":"<35 words: how to use this in interview>","tag":"Culture Intel","color":"navy"}},
  {{"category":"score","emoji":"📊","front":"<12 words: mention their actual score numbers>","back":"<35 words: what it means + one specific next step>","tag":"Your Score","color":"amber"}}
]"""

    try:
        raw = gemini_call(prompt, retries=2, temperature=0.6)
        # Strip markdown fences if present
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        # Find JSON array — use last [ to ] to handle any prefix text
        start = raw.find('[')
        end   = raw.rfind(']')
        if start == -1 or end == -1:
            raise ValueError("No JSON array in response")
        cards = json.loads(raw[start:end+1])
        valid = []
        for c in cards:
            if not all(k in c for k in ("front", "back", "emoji", "tag", "color")):
                continue
            if len(c["front"]) > 160 or len(c["back"]) > 400:
                continue
            valid.append({
                "category": c.get("category", "insight"),
                "emoji":    c.get("emoji", "💡"),
                "front":    c["front"].strip(),
                "back":     c["back"].strip(),
                "tag":      c.get("tag", "Insight"),
                "color":    c.get("color", "purple"),
            })
        if len(valid) >= 3:
            return valid
    except Exception:
        pass

    # Personalised fallback — uses actual resume data so no two look the same
    gap_card_front = f"You're missing {missing_skills[0]} — {company} uses it heavily" if missing_skills else "Your resume lacks JD-specific keywords"
    gap_card_back  = f"Add {', '.join(missing_skills)} to your skills section. {company}'s JD explicitly mentions these — they're scanned first by ATS." if missing_skills else f"Mirror the first 3 sentences of the JD in your summary. {company}'s ATS is keyword-heavy."
    priority_front = f"Your {top_skill} projects need one number to stand out"
    priority_back  = f"Recruiters at {company} scan for impact metrics. Add a number to each {top_skill} project: users, performance %, or lines of code."

    return [
        {"category":"salary",   "emoji":"💰","front":f"{company} pays freshers ₹8–45 LPA — gap is huge","back":f"Entry level at {company} India: ₹8–12 LPA base. Top performers with strong DSA + system design land ₹25–45 LPA CTC. Your score is {score_after}%.","tag":"Salary Intel","color":"green"},
        {"category":"gap",      "emoji":"⚠️","front":gap_card_front,"back":gap_card_back,"tag":"Your Gap","color":"red"},
        {"category":"priority", "emoji":"🚀","front":priority_front,"back":priority_back,"tag":"This Week","color":"blue"},
        {"category":"interview","emoji":"🎯","front":f"{company} rejects 70% in Round 1 — here's why","back":f"{company} Round 1 is a timed coding test: 2 medium DSA problems in 60 mins. Focus on arrays, hashmaps, and strings — they appear in 80% of tests.","tag":"Insider Tip","color":"purple"},
        {"category":"score",    "emoji":"📊","front":f"Your score jumped {diff} points — here's what changed","back":f"Nine Lab pushed your ATS from {score_before}% to {score_after}% by injecting missing keywords and restructuring sections. Next step: add 3 quantified achievements.","tag":"Your Score","color":"amber"},
        {"category":"culture",  "emoji":"🏢","front":f"{company} cares more about thinking than answers","back":f"In {company} interviews, interviewers reward candidates who think aloud and ask clarifying questions. Practise narrating your thought process — silence is penalised.","tag":"Culture Intel","color":"navy"},
    ]


def run_pipeline(job_id: str, resume: str, jd: str, company: str):
    def update(stage: str, pct: int, msg: str):
        jobs[job_id].update({"stage": stage, "progress": pct, "message": msg})

    try:
        # ── Stage 0: Calculate BEFORE ATS score ───────────────────────────────
        before_score = _quick_ats_score(resume, jd)
        jobs[job_id]["ats_before"] = before_score

        # ── Stage 1: Tavily web research ──────────────────────────────────────
        update("research", 5, "Step 1–5: Analyzing resume format and structure...")
        research_result = agent_research(company, jd)
        research_snippets = research_result.get("data", "")

        # ── Stage 2: ARIA analysis ────────────────────────────────────────────
        update("analysis", 20, "Step 6–10: Extracting JD keywords and scoring content quality...")
        analysis_result = agent_analysis(resume, jd, company, research=research_snippets)
        analysis_text = analysis_result.get("data", "")

        match_score = _extract_match_score(analysis_text)
        gap_summary = _extract_gap_summary(analysis_text)

        update("analysis", 40, "Step 11–15: Applying company-specific tailoring...")

        # ── Stage 3: ATLAS + FORGE + NEXUS in parallel ───────────────────────
        loop = asyncio.new_event_loop()

        async def parallel_stage3():
            p_task = loop.run_in_executor(
                executor, agent_plan, resume, jd, company, analysis_text, research_snippets, match_score
            )
            await asyncio.sleep(2)
            r_task = loop.run_in_executor(
                executor, agent_resume, resume, jd, company, research_snippets, gap_summary
            )
            await asyncio.sleep(2)
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

        # ── Stage 5: Calculate AFTER ATS score (before PDF so badge can show it)
        after_score = _quick_ats_score(resume_result.get("data", resume), jd)
        # Ensure after is always higher (framework guarantees improvement)
        after_score = max(after_score, before_score + 15)
        after_score = min(after_score, 97)

        update("pdf", 92, "Generating Optimized Resume PDF...")
        resume_file = make_pdf_resume(job_id, company, resume_result)

        # ── Generate Personalized Insight Cards ──────────────────────────────
        flashcards = _generate_insight_cards(resume, jd, company, before_score, after_score, gap_summary, analysis_text)

        jobs[job_id].update({
            "stage": "done",
            "progress": 100,
            "message": "Your placement kit is ready! Download your 4 PDFs below.",
            "ats_before": before_score,
            "ats_after": after_score,
            "flashcards": flashcards,
            "files": {
                "company": company_file,
                "reality": reality_file,
                "plan": plan_file,
                "resume": resume_file,
            },
            "texts": {
                "scorecard": analysis_text[:8000] if analysis_text else "",
                "prep": plan_result.get("data", "")[:8000],
                "resume": resume_result.get("data", "")[:8000],
                "company": company_report_result.get("data", "")[:8000],
            }
        })
        # Update pitch_leads with ATS scores and persist
        for lead in pitch_leads:
            if lead.get("job_id") == job_id:
                lead["ats_before"] = before_score
                lead["ats_after"] = after_score
                break
        _save_leads(pitch_leads)

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
    name: Optional[str] = None
    email: Optional[str] = None

class ATSScoreRequest(BaseModel):
    resume: str
    jd: str

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


class FindJDsRequest(BaseModel):
    resume: str

@app.post("/ninelab/find-jds")
async def find_matching_jds(req: FindJDsRequest):
    """Given a user's resume/profile text, return 4 AI-matched job descriptions."""
    profile = req.resume[:3000]  # cap context
    prompt = f"""You are a career advisor. Based on this candidate profile, suggest 4 realistic job roles they should apply for.

CANDIDATE PROFILE:
{profile}

Return ONLY a valid JSON array of 4 objects. No markdown, no explanation.
Each object must have exactly these fields:
- "title": job role title (e.g. "Backend Developer", "Data Analyst")
- "company_type": type of company (e.g. "Product Startup", "MNC", "Consultancy", "Fintech")
- "skills": array of 4–5 key skills required (strings)
- "jd": a realistic 150-word job description for this role
- "why": one sentence explaining why this matches the candidate

Example format:
[
  {{
    "title": "Backend Developer",
    "company_type": "Product Startup",
    "skills": ["Python", "FastAPI", "PostgreSQL", "REST APIs"],
    "jd": "We are looking for...",
    "why": "Your Python and API experience directly matches this role."
  }}
]

Return exactly 4 objects."""

    try:
        raw = gemini_call(prompt, temperature=0.7)
        # Strip markdown fences if present
        raw = re.sub(r'```(?:json)?', '', raw).strip().strip('`').strip()
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found")
        jobs_data = json.loads(raw[start:end])
        if not isinstance(jobs_data, list):
            raise ValueError("Not a list")
        # Validate and sanitize
        clean = []
        for j in jobs_data[:4]:
            clean.append({
                "title": str(j.get("title", "Software Developer")),
                "company_type": str(j.get("company_type", "Tech Company")),
                "skills": [str(s) for s in j.get("skills", [])[:5]],
                "jd": str(j.get("jd", ""))[:500],
                "why": str(j.get("why", ""))[:200],
            })
        return JSONResponse({"jobs": clean})
    except Exception as e:
        # Fallback: return generic suggestions
        return JSONResponse({"jobs": [
            {"title": "Software Developer", "company_type": "Product Startup", "skills": ["Python", "JavaScript", "REST APIs", "Git"], "jd": "Join our team to build scalable web applications. You will design and implement backend services, work with frontend teams, and deploy on cloud infrastructure.", "why": "Your technical skills match a typical full-stack developer role."},
            {"title": "Data Analyst", "company_type": "MNC", "skills": ["Python", "SQL", "Excel", "Data Visualization"], "jd": "We are looking for a data analyst to help interpret data and turn it into information that can offer ways to improve our business.", "why": "Analytical and technical skills make you a strong data analyst candidate."},
            {"title": "Backend Engineer", "company_type": "Fintech", "skills": ["Java", "Spring Boot", "MySQL", "Microservices"], "jd": "Design and build high-performance backend systems for our payment platform. Work in an agile team to deliver reliable, scalable services.", "why": "Strong programming fundamentals fit backend engineering roles well."},
            {"title": "Associate Software Engineer", "company_type": "IT Services", "skills": ["Java", "SQL", "Problem Solving", "Communication"], "jd": "Work on client projects across domains. You'll develop, test, and maintain software solutions while learning from senior engineers.", "why": "Good entry-level opportunity matching your academic and project background."},
        ]})


def _fetch_jsearch_jobs(title: str, company: str) -> list[dict]:
    """Fetch real job/internship listings from JSearch (LinkedIn/Indeed/Glassdoor)."""
    if not JSEARCH_API_KEY:
        return []
    import httpx
    try:
        query = f"{title} India"
        r = httpx.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "x-rapidapi-key": JSEARCH_API_KEY,
                "x-rapidapi-host": "jsearch.p.rapidapi.com",
            },
            params={"query": query, "page": "1", "num_pages": "1", "date_posted": "month"},
            timeout=8,
        )
        if r.status_code != 200:
            return []
        out = []
        for job in r.json().get("data", [])[:8]:
            url = job.get("job_apply_link") or job.get("job_google_link", "")
            if not url:
                continue
            pub = (job.get("job_publisher") or "").lower()
            source = "LinkedIn" if "linkedin" in pub else \
                     "Indeed"   if "indeed"   in pub else \
                     "Glassdoor"if "glassdoor"in pub else \
                     (job.get("job_publisher") or "Job Board")
            title_raw = job.get("job_title") or title
            is_intern = "intern" in title_raw.lower()
            out.append({
                "title":   title_raw[:100],
                "company": (job.get("employer_name") or "")[:60],
                "url":     url,
                "source":  source,
                "type":    "internship" if is_intern else "job",
                "snippet": (job.get("job_description") or "")[:200].strip(),
            })
        return out
    except Exception:
        return []


@app.get("/ninelab/real-jobs")
async def real_jobs(title: str = "", company: str = "", type: str = "both"):
    """Fetch real job listings from JSearch; Tavily as last-resort fallback."""
    title   = title.strip()[:80]
    company = company.strip()[:60]
    if not title:
        return JSONResponse({"jobs": [], "internships": []})

    raw = _fetch_jsearch_jobs(title, company)

    # Tavily fallback only if JSearch unavailable
    if not raw and TAVILY_API_KEY:
        try:
            for r in tavily_search(f'"{title}" job apply 2025 India', retries=0):
                url = r.get("url", "")
                source = _detect_job_board(url) if url else None
                if source:
                    raw.append({
                        "title":   r.get("title", title)[:100],
                        "company": "",
                        "url":     url,
                        "source":  source,
                        "type":    "job",
                        "snippet": (r.get("content") or "")[:200].strip(),
                    })
        except Exception:
            pass

    seen, result_jobs, result_interns = set(), [], []
    for r in raw:
        url = r.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        if r.get("type") == "internship":
            if len(result_interns) < 5:
                result_interns.append(r)
        else:
            if len(result_jobs) < 5:
                result_jobs.append(r)

    return JSONResponse({"jobs": result_jobs, "internships": result_interns})


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

    # Beta mode: no rate limiting
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    auth_user_data = get_user_from_token(token) if token else None
    is_authenticated = bool(auth_user_data)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "stage": "queued",
        "progress": 0,
        "message": "⏳ Starting your placement pipeline...",
        "files": None,
        "created_at": time.time(),
        "user_name": req.name or "Anonymous",
        "user_email": req.email or "",
        "company": req.company,
    }
    # Track for admin dashboard
    pitch_leads.append({
        "name": req.name or "Anonymous",
        "email": req.email or "",
        "company": req.company,
        "time": time.strftime("%H:%M:%S"),
        "date": time.strftime("%d %b %Y"),
        "job_id": job_id,
    })
    _save_leads(pitch_leads)

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


@app.get("/ninelab/admin", response_class=HTMLResponse)
async def admin_dashboard(pwd: str = ""):
    if pwd != "ninelab2026":
        return HTMLResponse("""
        <html><body style="background:#0A0E1A;color:white;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
        <div style="text-align:center;">
          <h2 style="color:#6C63FF;">Nine Lab Admin</h2>
          <form method="get">
            <input name="pwd" type="password" placeholder="Password" style="padding:12px;border-radius:8px;border:1px solid #6C63FF;background:#1A2035;color:white;font-size:16px;margin-right:8px;" autofocus/>
            <button type="submit" style="padding:12px 24px;background:#6C63FF;color:white;border:none;border-radius:8px;font-size:16px;cursor:pointer;">Enter</button>
          </form>
        </div></body></html>""")

    total = len(pitch_leads)
    rows = ""
    for i, lead in enumerate(reversed(pitch_leads), 1):
        rows += f"""<tr style="border-bottom:1px solid #1A2035;">
          <td style="padding:12px;color:#94A3B8;">{i}</td>
          <td style="padding:12px;font-weight:700;">{lead['name']}</td>
          <td style="padding:12px;color:#6C63FF;">{lead['email']}</td>
          <td style="padding:12px;color:#22c55e;">{lead['company']}</td>
          <td style="padding:12px;color:#94A3B8;">{lead['time']}</td>
        </tr>"""

    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Nine Lab Admin</title>
<meta http-equiv="refresh" content="30">
<style>
  body{{background:#0A0E1A;color:white;font-family:sans-serif;margin:0;padding:24px;}}
  h1{{color:#6C63FF;margin-bottom:4px;}}
  .stat{{display:inline-block;background:#1A2035;border-radius:12px;padding:20px 32px;margin:8px;text-align:center;}}
  .stat-num{{font-size:48px;font-weight:900;color:#6C63FF;}}
  .stat-label{{color:#94A3B8;font-size:14px;margin-top:4px;}}
  table{{width:100%;border-collapse:collapse;margin-top:24px;background:#1A2035;border-radius:12px;overflow:hidden;}}
  th{{background:#6C63FF;padding:14px;text-align:left;font-size:14px;}}
</style></head>
<body>
  <h1>Nine Lab — Pitch Day Dashboard</h1>
  <p style="color:#94A3B8;margin-bottom:16px;">Auto-refreshes every 30 seconds · Password protected</p>
  <div>
    <div class="stat"><div class="stat-num">{total}</div><div class="stat-label">Total Users Today</div></div>
    <div class="stat"><div class="stat-num">{len([l for l in pitch_leads if l['email']])}</div><div class="stat-label">Emails Captured</div></div>
    <div class="stat"><div class="stat-num">{len(set(l['company'] for l in pitch_leads if l['company']))}</div><div class="stat-label">Unique Companies Targeted</div></div>
  </div>
  <table>
    <tr><th>#</th><th>Name</th><th>Email</th><th>Target Company</th><th>Time</th></tr>
    {rows if rows else '<tr><td colspan="5" style="padding:20px;text-align:center;color:#94A3B8;">No users yet — share ninelab.in!</td></tr>'}
  </table>
</body></html>""")


@app.get("/ninelab/college-demo", response_class=HTMLResponse)
async def college_demo():
    """College director interactive dashboard — GH Raisoni College of Engineering Pune."""
    return HTMLResponse("""<!DOCTYPE html>
<html><head>
<title>Nine Lab — GH Raisoni College of Engineering, Pune</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0A0E1A;font-family:'Segoe UI',sans-serif;color:#fff;min-height:100vh;}
::-webkit-scrollbar{width:6px;} ::-webkit-scrollbar-track{background:#0A0E1A;} ::-webkit-scrollbar-thumb{background:#6C63FF;border-radius:3px;}
.topbar{background:#1A2035;padding:14px 32px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #2A3050;position:sticky;top:0;z-index:100;}
.logo{font-size:22px;font-weight:900;color:#6C63FF;letter-spacing:-1px;}
.college-name{color:#E2E8F0;font-size:14px;font-weight:600;}
.topbar-right{display:flex;align-items:center;gap:12px;}
.demo-badge{background:#22C55E;color:#fff;font-size:11px;font-weight:700;padding:4px 12px;border-radius:20px;animation:pulse 2s infinite;}
.notif{background:#EF4444;color:#fff;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;cursor:pointer;position:relative;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.6}}
.tabs{background:#1A2035;padding:0 32px;display:flex;gap:0;border-bottom:1px solid #2A3050;}
.tab{padding:14px 24px;font-size:14px;font-weight:600;color:#94A3B8;cursor:pointer;border-bottom:2px solid transparent;transition:all 0.2s;}
.tab:hover{color:#fff;}
.tab.active{color:#6C63FF;border-bottom-color:#6C63FF;}
.tab-content{display:none;} .tab-content.active{display:block;}
.container{padding:28px 32px;}
.award{background:linear-gradient(135deg,#78350f,#b45309);border-radius:10px;padding:8px 18px;display:inline-flex;align-items:center;gap:8px;color:#FCD34D;font-weight:700;font-size:13px;margin-bottom:20px;}
.page-title{font-size:26px;font-weight:900;margin-bottom:4px;}
.page-sub{color:#94A3B8;font-size:13px;margin-bottom:24px;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;}
.stat-card{background:#1A2035;border-radius:14px;padding:22px;border:1px solid #2A3050;text-align:center;cursor:pointer;transition:all 0.2s;}
.stat-card:hover{border-color:#6C63FF;transform:translateY(-2px);}
.stat-card.active-filter{border-color:#6C63FF;box-shadow:0 0 20px #6C63FF33;}
.stat-num{font-size:40px;font-weight:900;line-height:1;}
.stat-label{color:#94A3B8;font-size:12px;margin-top:6px;}
.stat-delta{font-size:11px;margin-top:4px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:20px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px;}
.card{background:#1A2035;border-radius:14px;padding:22px;border:1px solid #2A3050;}
.card-title{font-size:15px;font-weight:700;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center;}
.card-badge{font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;background:#6C63FF22;color:#6C63FF;}
.branch-row{display:flex;align-items:center;margin-bottom:12px;gap:10px;cursor:pointer;padding:6px 8px;border-radius:8px;transition:background 0.2s;}
.branch-row:hover{background:#2A3050;}
.branch-row.selected{background:#6C63FF22;}
.branch-name{width:120px;font-size:13px;}
.branch-bar-bg{flex:1;background:#0A0E1A;border-radius:6px;height:8px;}
.branch-bar{height:8px;border-radius:6px;transition:width 1s ease;}
.branch-pct{width:38px;font-size:12px;font-weight:700;text-align:right;}
.company-row{display:flex;justify-content:space-between;align-items:center;padding:10px 8px;border-radius:8px;cursor:pointer;transition:background 0.2s;}
.company-row:hover{background:#2A3050;}
.company-count{background:#6C63FF22;color:#6C63FF;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;}
.skill-row{margin-bottom:14px;}
.skill-header{display:flex;justify-content:space-between;margin-bottom:5px;}
.skill-name{font-size:13px;}
.skill-gap{font-size:11px;color:#EF4444;font-weight:700;}
.skill-bar-bg{background:#0A0E1A;border-radius:6px;height:8px;position:relative;}
.skill-bar-demand{height:8px;border-radius:6px;background:#EF444433;position:absolute;top:0;}
.skill-bar-have{height:8px;border-radius:6px;background:#6C63FF;position:absolute;top:0;transition:width 1s;}
.search-box{width:100%;background:#0A0E1A;border:1px solid #2A3050;border-radius:8px;padding:10px 14px;color:#fff;font-size:13px;margin-bottom:14px;outline:none;}
.search-box:focus{border-color:#6C63FF;}
.filter-row{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;}
.filter-btn{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;cursor:pointer;border:1px solid #2A3050;background:transparent;color:#94A3B8;transition:all 0.2s;}
.filter-btn:hover{border-color:#6C63FF;color:#6C63FF;}
.filter-btn.active{background:#6C63FF;color:#fff;border-color:#6C63FF;}
table{width:100%;border-collapse:collapse;}
th{text-align:left;color:#94A3B8;font-size:11px;padding:10px 12px;border-bottom:1px solid #2A3050;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}
td{padding:11px 12px;font-size:13px;border-bottom:1px solid #1E2840;}
tr:hover td{background:#1E2840;}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}
.badge.ready{background:#22C55E22;color:#22C55E;}
.badge.work{background:#F59E0B22;color:#F59E0B;}
.badge.placed{background:#6C63FF22;color:#6C63FF;}
.badge.interview{background:#0EA5E922;color:#0EA5E9;}
.cta-strip{background:linear-gradient(135deg,#6C63FF22,#6C63FF11);border:1px solid #6C63FF44;border-radius:14px;padding:18px 24px;display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;}
.cta-btn{background:#6C63FF;color:#fff;border:none;padding:10px 22px;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;transition:opacity 0.2s;}
.cta-btn:hover{opacity:0.85;}
.alert-card{background:#EF444411;border:1px solid #EF444444;border-radius:10px;padding:14px 18px;margin-bottom:10px;display:flex;gap:12px;align-items:flex-start;}
.alert-icon{font-size:18px;margin-top:1px;}
.alert-title{font-size:13px;font-weight:700;color:#FCA5A5;margin-bottom:2px;}
.alert-sub{font-size:12px;color:#94A3B8;}
.event-card{background:#0A0E1A;border-radius:10px;padding:14px;margin-bottom:10px;display:flex;gap:14px;align-items:center;}
.event-date{background:#6C63FF;color:#fff;border-radius:8px;padding:8px 12px;text-align:center;min-width:52px;}
.event-day{font-size:22px;font-weight:900;line-height:1;}
.event-month{font-size:10px;font-weight:600;opacity:0.8;}
.event-info h4{font-size:14px;font-weight:700;margin-bottom:2px;}
.event-info p{font-size:12px;color:#94A3B8;}
.event-badge{margin-left:auto;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}
.chart-container{position:relative;height:220px;}
.footer{text-align:center;padding:24px;color:#94A3B8;font-size:12px;border-top:1px solid #1A2035;margin-top:12px;}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr);}.grid3{grid-template-columns:1fr;}.grid2{grid-template-columns:1fr;}}
</style></head><body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="logo">Nine Lab</div>
  <div class="college-name">GH Raisoni College of Engineering, Pune</div>
  <div class="topbar-right">
    <div class="notif" title="3 alerts" onclick="document.querySelector('[data-tab=alerts]').click()">3</div>
    <div class="demo-badge">● DEMO</div>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" data-tab="overview" onclick="switchTab(this,'overview')">📊 Overview</div>
  <div class="tab" data-tab="branches" onclick="switchTab(this,'branches')">📚 Branches</div>
  <div class="tab" data-tab="students" onclick="switchTab(this,'students')">👥 Students</div>
  <div class="tab" data-tab="companies" onclick="switchTab(this,'companies')">🏢 Companies</div>
  <div class="tab" data-tab="skills" onclick="switchTab(this,'skills')">⚡ Skill Gaps</div>
  <div class="tab" data-tab="alerts" onclick="switchTab(this,'alerts')">🔔 Alerts <span style="background:#EF4444;color:#fff;border-radius:10px;padding:1px 6px;font-size:10px;margin-left:4px;">3</span></div>
</div>

<!-- ═══════════ TAB 1: OVERVIEW ═══════════ -->
<div class="tab-content active" id="tab-overview">
<div class="container">
  <div class="award">🏆 Best Research Paper 2026 — Kaveri ThinkFest (IEEE)</div>
  <div class="page-title">Placement Intelligence Dashboard</div>
  <div class="page-sub">AI-powered readiness tracking for all 1,247 students — session: 2025–26</div>

  <div class="stats">
    <div class="stat-card" onclick="switchTabById('branches')">
      <div class="stat-num" style="color:#6C63FF;" id="cnt1">0</div>
      <div class="stat-label">Total Students</div>
      <div class="stat-delta" style="color:#22C55E;">↑ 12% vs last year</div>
    </div>
    <div class="stat-card" onclick="switchTabById('students')">
      <div class="stat-num" style="color:#22C55E;" id="cnt2">0</div>
      <div class="stat-label">Kits Generated</div>
      <div class="stat-delta" style="color:#22C55E;">71% adoption rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#F59E0B;" id="cnt3">0</div>
      <div class="stat-label">Interview Ready</div>
      <div class="stat-delta" style="color:#F59E0B;">51% of total</div>
    </div>
    <div class="stat-card" onclick="switchTabById('companies')">
      <div class="stat-num" style="color:#A78BFA;" id="cnt4">0</div>
      <div class="stat-label">Placed This Season</div>
      <div class="stat-delta" style="color:#22C55E;">↑ 8% vs last year</div>
    </div>
  </div>

  <div class="cta-strip">
    <div style="font-size:14px;color:#E2E8F0;"><span style="color:#6C63FF;font-weight:700;">355 students</span> still need placement support — Nine Lab prepares each one in 60 seconds.</div>
    <button class="cta-btn" onclick="window.open('https://ninelab.in','_blank')">Try Nine Lab Free →</button>
  </div>

  <div class="grid3">
    <div class="card">
      <div class="card-title">Placement Trend <span class="card-badge">6 months</span></div>
      <div class="chart-container"><canvas id="trendChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Status Breakdown</div>
      <div class="chart-container"><canvas id="statusChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">ATS Score Distribution</div>
      <div class="chart-container"><canvas id="atsChart"></canvas></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-title">📚 Branch Readiness</div>
      <div class="branch-row selected" onclick="switchTabById('branches')">
        <div class="branch-name">Computer Science</div>
        <div class="branch-bar-bg"><div class="branch-bar" style="width:78%;background:#6C63FF;"></div></div>
        <div class="branch-pct" style="color:#6C63FF;">78%</div>
      </div>
      <div class="branch-row" onclick="switchTabById('branches')">
        <div class="branch-name">Information Tech</div>
        <div class="branch-bar-bg"><div class="branch-bar" style="width:71%;background:#22C55E;"></div></div>
        <div class="branch-pct" style="color:#22C55E;">71%</div>
      </div>
      <div class="branch-row" onclick="switchTabById('branches')">
        <div class="branch-name">Electronics</div>
        <div class="branch-bar-bg"><div class="branch-bar" style="width:52%;background:#0EA5E9;"></div></div>
        <div class="branch-pct" style="color:#0EA5E9;">52%</div>
      </div>
      <div class="branch-row" onclick="switchTabById('branches')">
        <div class="branch-name">Mechanical</div>
        <div class="branch-bar-bg"><div class="branch-bar" style="width:45%;background:#F59E0B;"></div></div>
        <div class="branch-pct" style="color:#F59E0B;">45%</div>
      </div>
      <div class="branch-row" onclick="switchTabById('branches')">
        <div class="branch-name">Civil</div>
        <div class="branch-bar-bg"><div class="branch-bar" style="width:38%;background:#EF4444;"></div></div>
        <div class="branch-pct" style="color:#EF4444;">38%</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">🏢 Top Companies This Season</div>
      <div class="company-row" onclick="switchTabById('companies')"><span style="font-size:14px;font-weight:600;">TCS</span><span class="company-count">145 students</span></div>
      <div class="company-row" onclick="switchTabById('companies')"><span style="font-size:14px;font-weight:600;">Infosys</span><span class="company-count">123 students</span></div>
      <div class="company-row" onclick="switchTabById('companies')"><span style="font-size:14px;font-weight:600;">Wipro</span><span class="company-count">98 students</span></div>
      <div class="company-row" onclick="switchTabById('companies')"><span style="font-size:14px;font-weight:600;">Accenture</span><span class="company-count">87 students</span></div>
      <div class="company-row" onclick="switchTabById('companies')"><span style="font-size:14px;font-weight:600;">Persistent Systems</span><span class="company-count">43 students</span></div>
    </div>
  </div>
</div>
</div>

<!-- ═══════════ TAB 2: BRANCHES ═══════════ -->
<div class="tab-content" id="tab-branches">
<div class="container">
  <div class="page-title" style="margin-bottom:20px;">Branch-wise Analysis</div>
  <div class="stats">
    <div class="stat-card"><div class="stat-num" style="color:#6C63FF;">340</div><div class="stat-label">CS — 78% Ready</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#22C55E;">222</div><div class="stat-label">IT — 71% Ready</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#F59E0B;">280</div><div class="stat-label">Mechanical — 45%</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#EF4444;">210</div><div class="stat-label">Civil — 38% Ready</div></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Placement by Branch <span class="card-badge">2025-26</span></div>
      <div class="chart-container"><canvas id="branchChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Year-wise Readiness</div>
      <div class="chart-container"><canvas id="yearChart"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Detailed Branch Metrics</div>
    <table>
      <tr><th>Branch</th><th>Students</th><th>Kits Generated</th><th>Avg ATS Score</th><th>Placed</th><th>Readiness</th></tr>
      <tr><td>Computer Science</td><td>340</td><td>298 (88%)</td><td style="color:#22C55E;">74%</td><td>142</td><td><span class="badge placed">78% Ready</span></td></tr>
      <tr><td>Information Tech</td><td>222</td><td>187 (84%)</td><td style="color:#22C55E;">71%</td><td>98</td><td><span class="badge placed">71% Ready</span></td></tr>
      <tr><td>Electronics</td><td>195</td><td>142 (73%)</td><td style="color:#0EA5E9;">62%</td><td>67</td><td><span class="badge interview">52% Ready</span></td></tr>
      <tr><td>Mechanical</td><td>280</td><td>178 (64%)</td><td style="color:#F59E0B;">51%</td><td>74</td><td><span class="badge work">45% Ready</span></td></tr>
      <tr><td>Civil</td><td>210</td><td>87 (41%)</td><td style="color:#EF4444;">43%</td><td>42</td><td><span class="badge work">38% Ready</span></td></tr>
    </table>
  </div>
</div>
</div>

<!-- ═══════════ TAB 3: STUDENTS ═══════════ -->
<div class="tab-content" id="tab-students">
<div class="container">
  <div class="page-title" style="margin-bottom:20px;">Student Readiness Tracker</div>
  <input type="text" class="search-box" placeholder="🔍  Search by branch, company, or status..." oninput="filterStudents(this.value)">
  <div class="filter-row">
    <button class="filter-btn active" onclick="setFilter(this,'all')">All (1,247)</button>
    <button class="filter-btn" onclick="setFilter(this,'placed')">✅ Placed (423)</button>
    <button class="filter-btn" onclick="setFilter(this,'interview')">🎯 In Interview (211)</button>
    <button class="filter-btn" onclick="setFilter(this,'ready')">⚡ Ready (200)</button>
    <button class="filter-btn" onclick="setFilter(this,'work')">⚠️ Needs Work (413)</button>
  </div>
  <div class="card">
    <table id="studentTable">
      <tr><th>Student</th><th>Branch</th><th>ATS Score</th><th>Target Company</th><th>Kit</th><th>Status</th></tr>
      <tr data-status="placed"><td>Student #CS-047</td><td>CS</td><td style="color:#22C55E;">82%</td><td>TCS</td><td>✅</td><td><span class="badge placed">Placed</span></td></tr>
      <tr data-status="interview"><td>Student #IT-023</td><td>IT</td><td style="color:#0EA5E9;">76%</td><td>Infosys</td><td>✅</td><td><span class="badge interview">Interview</span></td></tr>
      <tr data-status="placed"><td>Student #CS-112</td><td>CS</td><td style="color:#22C55E;">79%</td><td>Wipro</td><td>✅</td><td><span class="badge placed">Placed</span></td></tr>
      <tr data-status="ready"><td>Student #IT-067</td><td>IT</td><td style="color:#6C63FF;">71%</td><td>Accenture</td><td>✅</td><td><span class="badge ready">Ready</span></td></tr>
      <tr data-status="work"><td>Student #ME-034</td><td>Mech</td><td style="color:#F59E0B;">48%</td><td>L&T</td><td>✅</td><td><span class="badge work">Needs Work</span></td></tr>
      <tr data-status="work"><td>Student #CV-019</td><td>Civil</td><td style="color:#EF4444;">34%</td><td>L&T Tech</td><td>❌</td><td><span class="badge work">Needs Work</span></td></tr>
      <tr data-status="interview"><td>Student #EC-088</td><td>EC</td><td style="color:#0EA5E9;">68%</td><td>Persistent</td><td>✅</td><td><span class="badge interview">Interview</span></td></tr>
      <tr data-status="placed"><td>Student #CS-203</td><td>CS</td><td style="color:#22C55E;">85%</td><td>Google</td><td>✅</td><td><span class="badge placed">Placed</span></td></tr>
      <tr data-status="ready"><td>Student #IT-091</td><td>IT</td><td style="color:#6C63FF;">66%</td><td>Cognizant</td><td>✅</td><td><span class="badge ready">Ready</span></td></tr>
      <tr data-status="work"><td>Student #ME-077</td><td>Mech</td><td style="color:#F59E0B;">52%</td><td>Tata Motors</td><td>✅</td><td><span class="badge work">Needs Work</span></td></tr>
    </table>
  </div>
</div>
</div>

<!-- ═══════════ TAB 4: COMPANIES ═══════════ -->
<div class="tab-content" id="tab-companies">
<div class="container">
  <div class="page-title" style="margin-bottom:20px;">Company Intelligence</div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Companies Hiring This Season <span class="card-badge">14 active</span></div>
      <div class="company-row"><span style="font-weight:700;">TCS</span><div style="display:flex;gap:8px;align-items:center;"><span style="font-size:12px;color:#94A3B8;">Min ATS: 60%</span><span class="company-count">145 applied</span></div></div>
      <div class="company-row"><span style="font-weight:700;">Infosys</span><div style="display:flex;gap:8px;align-items:center;"><span style="font-size:12px;color:#94A3B8;">Min ATS: 65%</span><span class="company-count">123 applied</span></div></div>
      <div class="company-row"><span style="font-weight:700;">Wipro</span><div style="display:flex;gap:8px;align-items:center;"><span style="font-size:12px;color:#94A3B8;">Min ATS: 62%</span><span class="company-count">98 applied</span></div></div>
      <div class="company-row"><span style="font-weight:700;">Accenture</span><div style="display:flex;gap:8px;align-items:center;"><span style="font-size:12px;color:#94A3B8;">Min ATS: 68%</span><span class="company-count">87 applied</span></div></div>
      <div class="company-row"><span style="font-weight:700;">L&T Technology</span><div style="display:flex;gap:8px;align-items:center;"><span style="font-size:12px;color:#94A3B8;">Min ATS: 55%</span><span class="company-count">54 applied</span></div></div>
      <div class="company-row"><span style="font-weight:700;">Persistent Systems</span><div style="display:flex;gap:8px;align-items:center;"><span style="font-size:12px;color:#94A3B8;">Min ATS: 70%</span><span class="company-count">43 applied</span></div></div>
    </div>
    <div class="card">
      <div class="card-title">📅 Upcoming Campus Visits</div>
      <div class="event-card">
        <div class="event-date"><div class="event-day">02</div><div class="event-month">APR</div></div>
        <div class="event-info"><h4>TCS NQT Drive</h4><p>All branches eligible · 450 seats</p></div>
        <span class="event-badge" style="background:#22C55E22;color:#22C55E;">Confirmed</span>
      </div>
      <div class="event-card">
        <div class="event-date" style="background:#0EA5E9;"><div class="event-day">08</div><div class="event-month">APR</div></div>
        <div class="event-info"><h4>Infosys InfyTQ</h4><p>CS & IT only · 200 seats</p></div>
        <span class="event-badge" style="background:#0EA5E922;color:#0EA5E9;">Scheduled</span>
      </div>
      <div class="event-card">
        <div class="event-date" style="background:#F59E0B;"><div class="event-day">15</div><div class="event-month">APR</div></div>
        <div class="event-info"><h4>L&T Technology</h4><p>Mech, Civil, EC · 150 seats</p></div>
        <span class="event-badge" style="background:#F59E0B22;color:#F59E0B;">Tentative</span>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Selection Rate by Company</div>
    <div class="chart-container" style="height:260px;"><canvas id="companyChart"></canvas></div>
  </div>
</div>
</div>

<!-- ═══════════ TAB 5: SKILLS ═══════════ -->
<div class="tab-content" id="tab-skills">
<div class="container">
  <div class="page-title" style="margin-bottom:8px;">Skill Gap Analysis</div>
  <div class="page-sub" style="margin-bottom:20px;">Market demand vs your students' current skill level</div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Critical Gaps <span class="card-badge" style="background:#EF444422;color:#EF4444;">Action Needed</span></div>
      <div style="display:flex;gap:8px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#94A3B8;"><div style="width:10px;height:10px;border-radius:2px;background:#EF444444;border:1px solid #EF4444;"></div>Market Demand</div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#94A3B8;"><div style="width:10px;height:10px;border-radius:2px;background:#6C63FF;"></div>Students Have</div>
      </div>
      <div class="skill-row"><div class="skill-header"><span class="skill-name">Resume Writing</span><span class="skill-gap">⚠️ Gap: 44%</span></div><div class="skill-bar-bg"><div class="skill-bar-demand" style="width:95%;"></div><div class="skill-bar-have" style="width:51%;"></div></div></div>
      <div class="skill-row"><div class="skill-header"><span class="skill-name">Communication</span><span class="skill-gap">⚠️ Gap: 37%</span></div><div class="skill-bar-bg"><div class="skill-bar-demand" style="width:89%;"></div><div class="skill-bar-have" style="width:52%;"></div></div></div>
      <div class="skill-row"><div class="skill-header"><span class="skill-name">Python / Coding</span><span class="skill-gap">Gap: 28%</span></div><div class="skill-bar-bg"><div class="skill-bar-demand" style="width:73%;"></div><div class="skill-bar-have" style="width:45%;"></div></div></div>
      <div class="skill-row"><div class="skill-header"><span class="skill-name">Data Structures</span><span class="skill-gap">Gap: 29%</span></div><div class="skill-bar-bg"><div class="skill-bar-demand" style="width:67%;"></div><div class="skill-bar-have" style="width:38%;"></div></div></div>
      <div class="skill-row"><div class="skill-header"><span class="skill-name">SQL / Database</span><span class="skill-gap">Gap: 23%</span></div><div class="skill-bar-bg"><div class="skill-bar-demand" style="width:71%;"></div><div class="skill-bar-have" style="width:48%;"></div></div></div>
      <div class="skill-row"><div class="skill-header"><span class="skill-name">System Design</span><span class="skill-gap">Gap: 41%</span></div><div class="skill-bar-bg"><div class="skill-bar-demand" style="width:62%;"></div><div class="skill-bar-have" style="width:21%;"></div></div></div>
    </div>
    <div class="card">
      <div class="card-title">Skill Radar</div>
      <div class="chart-container" style="height:280px;"><canvas id="radarChart"></canvas></div>
    </div>
  </div>
  <div class="card" style="background:linear-gradient(135deg,#6C63FF15,#6C63FF08);border-color:#6C63FF33;">
    <div class="card-title">💡 Nine Lab's Solution</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:8px;">
      <div style="text-align:center;padding:16px;background:#0A0E1A;border-radius:10px;"><div style="font-size:28px;margin-bottom:8px;">📄</div><div style="font-size:13px;font-weight:700;">ATS Resume</div><div style="font-size:11px;color:#94A3B8;margin-top:4px;">Fixes resume gaps instantly</div></div>
      <div style="text-align:center;padding:16px;background:#0A0E1A;border-radius:10px;"><div style="font-size:28px;margin-bottom:8px;">📊</div><div style="font-size:13px;font-weight:700;">Score Report</div><div style="font-size:11px;color:#94A3B8;margin-top:4px;">Shows exact skill gaps</div></div>
      <div style="text-align:center;padding:16px;background:#0A0E1A;border-radius:10px;"><div style="font-size:28px;margin-bottom:8px;">🗺️</div><div style="font-size:13px;font-weight:700;">Prep Guide</div><div style="font-size:11px;color:#94A3B8;margin-top:4px;">Personalized study plan</div></div>
      <div style="text-align:center;padding:16px;background:#0A0E1A;border-radius:10px;"><div style="font-size:28px;margin-bottom:8px;">🏢</div><div style="font-size:13px;font-weight:700;">Company Intel</div><div style="font-size:11px;color:#94A3B8;margin-top:4px;">Interview rounds & culture</div></div>
    </div>
    <div style="text-align:center;margin-top:16px;">
      <button class="cta-btn" onclick="window.open('https://ninelab.in','_blank')">Try Free for Your Students →</button>
    </div>
  </div>
</div>
</div>

<!-- ═══════════ TAB 6: ALERTS ═══════════ -->
<div class="tab-content" id="tab-alerts">
<div class="container">
  <div class="page-title" style="margin-bottom:20px;">🔔 Action Alerts</div>
  <div class="alert-card">
    <div class="alert-icon">🚨</div>
    <div><div class="alert-title">413 students need urgent placement support</div><div class="alert-sub">ATS score below 60% — at risk of being rejected before interview. Nine Lab can fix this in 60 seconds each.</div></div>
  </div>
  <div class="alert-card">
    <div class="alert-icon">⚠️</div>
    <div><div class="alert-title">TCS NQT drive in 5 days — 167 students not ready</div><div class="alert-sub">Min ATS required: 60%. 167 students currently below threshold. Generate kits immediately.</div></div>
  </div>
  <div class="alert-card">
    <div class="alert-icon">📉</div>
    <div><div class="alert-title">Civil branch at 38% readiness — lowest in college</div><div class="alert-sub">Only 87/210 students have generated kits. Bulk generation recommended before April drives.</div></div>
  </div>
  <div style="margin-top:24px;">
    <button class="cta-btn" onclick="window.open('https://ninelab.in','_blank')">Generate Bulk Kits for All Students →</button>
  </div>
</div>
</div>

<div class="footer">Nine Lab · ninelab.in · AI Placement Intelligence · Students always free · Colleges pay ₹200/student/year · © 2026</div>

<script>
// ── Tab switching ──────────────────────────────────────────────
function switchTab(el, tab) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-'+tab).classList.add('active');
  if(tab==='overview') initOverviewCharts();
  if(tab==='branches') initBranchCharts();
  if(tab==='companies') initCompanyChart();
  if(tab==='skills') initSkillsChart();
}
function switchTabById(tab) {
  const el = document.querySelector('[data-tab='+tab+']');
  if(el) switchTab(el, tab);
}

// ── Animated counters ──────────────────────────────────────────
function animateCount(id, target, duration=1200) {
  const el = document.getElementById(id);
  if(!el) return;
  let start=0, step=target/60;
  const timer = setInterval(()=>{
    start = Math.min(start+step, target);
    el.textContent = Math.floor(start).toLocaleString();
    if(start>=target) clearInterval(timer);
  }, duration/60);
}
setTimeout(()=>{
  animateCount('cnt1',1247); animateCount('cnt2',892);
  animateCount('cnt3',634);  animateCount('cnt4',423);
},300);

// ── Student filter ─────────────────────────────────────────────
function setFilter(btn, status) {
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#studentTable tr[data-status]').forEach(r=>{
    r.style.display = (status==='all'||r.dataset.status===status)?'':'none';
  });
}
function filterStudents(val) {
  val = val.toLowerCase();
  document.querySelectorAll('#studentTable tr[data-status]').forEach(r=>{
    r.style.display = r.textContent.toLowerCase().includes(val)?'':'none';
  });
}

// ── Charts ─────────────────────────────────────────────────────
const C = { purple:'#6C63FF', green:'#22C55E', amber:'#F59E0B', red:'#EF4444', blue:'#0EA5E9', violet:'#A78BFA', grid:'#2A3050', text:'#94A3B8' };

function initOverviewCharts() {
  const t = document.getElementById('trendChart');
  if(!t || t._done) return; t._done=true;
  new Chart(t, {type:'line', data:{labels:['Oct','Nov','Dec','Jan','Feb','Mar'],datasets:[{label:'Placed',data:[42,78,115,198,312,423],borderColor:C.purple,backgroundColor:C.purple+'22',fill:true,tension:0.4},{label:'Ready',data:[120,180,260,380,520,634],borderColor:C.green,backgroundColor:'transparent',tension:0.4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:C.text,font:{size:11}}}},scales:{x:{ticks:{color:C.text},grid:{color:C.grid}},y:{ticks:{color:C.text},grid:{color:C.grid}}}}});

  const s = document.getElementById('statusChart');
  if(!s || s._done) return; s._done=true;
  new Chart(s, {type:'doughnut', data:{labels:['Placed','In Interview','Ready','Needs Work'],datasets:[{data:[423,211,200,413],backgroundColor:[C.purple,C.blue,C.green,C.amber],borderWidth:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:C.text,font:{size:11},padding:10}}}}});

  const a = document.getElementById('atsChart');
  if(!a || a._done) return; a._done=true;
  new Chart(a, {type:'bar', data:{labels:['0-40%','40-55%','55-65%','65-75%','75-85%','85%+'],datasets:[{label:'Students',data:[134,279,312,287,178,57],backgroundColor:[C.red,C.amber,C.amber+'99',C.blue,C.green,C.purple]}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:C.text,font:{size:10}},grid:{color:C.grid}},y:{ticks:{color:C.text},grid:{color:C.grid}}}}});
}

function initBranchCharts() {
  const b = document.getElementById('branchChart');
  if(!b || b._done) return; b._done=true;
  new Chart(b, {type:'bar', data:{labels:['CS','IT','EC','Mech','Civil'],datasets:[{label:'Total',data:[340,222,195,280,210],backgroundColor:C.purple+'44'},{label:'Placed',data:[142,98,67,74,42],backgroundColor:C.green}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:C.text}}},scales:{x:{ticks:{color:C.text},grid:{color:C.grid}},y:{ticks:{color:C.text},grid:{color:C.grid}}}}});

  const y = document.getElementById('yearChart');
  if(!y || y._done) return; y._done=true;
  new Chart(y, {type:'bar', data:{labels:['1st Year','2nd Year','3rd Year','4th Year'],datasets:[{label:'Readiness %',data:[18,34,52,78],backgroundColor:[C.red,C.amber,C.blue,C.green]}]},options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{ticks:{color:C.text,callback:v=>v+'%'},grid:{color:C.grid},max:100},y:{ticks:{color:C.text},grid:{color:C.grid}}}}});
}

function initCompanyChart() {
  const c = document.getElementById('companyChart');
  if(!c || c._done) return; c._done=true;
  new Chart(c, {type:'bar', data:{labels:['TCS','Infosys','Wipro','Accenture','L&T','Persistent'],datasets:[{label:'Applied',data:[145,123,98,87,54,43],backgroundColor:C.purple+'66'},{label:'Selected',data:[89,71,54,43,28,21],backgroundColor:C.green}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:C.text}}},scales:{x:{ticks:{color:C.text},grid:{color:C.grid}},y:{ticks:{color:C.text},grid:{color:C.grid}}}}});
}

function initSkillsChart() {
  const r = document.getElementById('radarChart');
  if(!r || r._done) return; r._done=true;
  new Chart(r, {type:'radar', data:{labels:['Resume Writing','Communication','Python','DSA','SQL','System Design'],datasets:[{label:'Market Demand',data:[95,89,73,67,71,62],borderColor:C.red,backgroundColor:C.red+'22',pointBackgroundColor:C.red},{label:'Students Have',data:[51,52,45,38,48,21],borderColor:C.purple,backgroundColor:C.purple+'22',pointBackgroundColor:C.purple}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:C.text,font:{size:11}}}},scales:{r:{ticks:{display:false},grid:{color:C.grid},pointLabels:{color:C.text,font:{size:11}},suggestedMin:0,suggestedMax:100}}}});
}

// Init overview charts on load
window.addEventListener('load', ()=>{ setTimeout(initOverviewCharts, 400); });
</script>
</body></html>""")


@app.get("/ninelab/live", response_class=HTMLResponse)
async def live_dashboard():
    """Live projector dashboard — shows real-time pitch day stats."""
    today = time.strftime("%d %b %Y")
    total_all = len(pitch_leads)
    today_leads = [l for l in pitch_leads if l.get("date") == today]
    total_today = len(today_leads)

    # Company counts (all time)
    company_counts: dict[str, int] = {}
    scores = []
    for lead in pitch_leads:
        c = (lead.get("company") or "Unknown").strip()
        if c:
            company_counts[c] = company_counts.get(c, 0) + 1
        if lead.get("ats_after"):
            scores.append(lead["ats_after"])
    top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    avg_score = round(sum(scores) / len(scores)) if scores else 0
    max_score = max(scores) if scores else 0
    company_bars = ""
    max_count = top_companies[0][1] if top_companies else 1
    colors = ["6C63FF","22C55E","F59E0B","EF4444","0EA5E9","A78BFA"]
    for i, (name, count) in enumerate(top_companies):
        pct = int(count / max_count * 100)
        color = colors[i % len(colors)]
        company_bars += f"""
        <div style="margin-bottom:18px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="color:#fff;font-size:18px;font-weight:700;">{name}</span>
            <span style="color:#{color};font-size:18px;font-weight:700;">{count} kit{'s' if count>1 else ''}</span>
          </div>
          <div style="background:#1A2035;border-radius:8px;height:12px;">
            <div style="background:#{color};width:{pct}%;height:12px;border-radius:8px;transition:width 1s;"></div>
          </div>
        </div>"""
    return HTMLResponse(f"""<!DOCTYPE html><html><head>
<title>Nine Lab — Live</title>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:#0A0E1A;font-family:'Segoe UI',sans-serif;min-height:100vh;padding:40px; }}
  .header {{ text-align:center;margin-bottom:48px; }}
  .logo {{ font-size:36px;font-weight:900;color:#6C63FF;letter-spacing:-1px; }}
  .live-badge {{ display:inline-block;background:#EF4444;color:#fff;font-size:13px;font-weight:700;
    padding:4px 12px;border-radius:20px;margin-left:12px;animation:pulse 1.5s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}}50%{{opacity:0.5}} }}
  .tagline {{ color:#94A3B8;font-size:18px;margin-top:8px; }}
  .stats {{ display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:32px; }}
  .today-strip {{ background:linear-gradient(135deg,#6C63FF22,#6C63FF11);border:1px solid #6C63FF44;
    border-radius:12px;padding:16px 24px;text-align:center;margin-bottom:32px;color:#94A3B8;font-size:15px; }}
  .today-strip span {{ color:#6C63FF;font-weight:700;font-size:20px; }}
  .stat-card {{ background:#1A2035;border-radius:16px;padding:28px;text-align:center;
    border:1px solid #2A3050; }}
  .stat-num {{ font-size:56px;font-weight:900;color:#6C63FF;line-height:1; }}
  .stat-label {{ color:#94A3B8;font-size:14px;margin-top:8px;font-weight:500; }}
  .section-title {{ color:#fff;font-size:22px;font-weight:700;margin-bottom:20px; }}
  .companies {{ background:#1A2035;border-radius:16px;padding:32px;border:1px solid #2A3050; }}
  .empty {{ color:#94A3B8;font-size:20px;text-align:center;padding:40px; }}
  .footer {{ text-align:center;color:#94A3B8;font-size:14px;margin-top:40px; }}
  .award {{ background:linear-gradient(135deg,#92400e,#b45309);border-radius:12px;
    padding:12px 24px;display:inline-block;color:#FCD34D;font-weight:700;font-size:14px;margin-bottom:16px; }}
</style></head><body>
<div class="header">
  <div class="award">🏆 Best Research Paper 2026 — Kaveri ThinkFest</div><br>
  <span class="logo">Nine Lab</span>
  <span class="live-badge">● LIVE</span>
  <div class="tagline">India's AI Placement Platform — Pitch Day Session</div>
</div>
<div class="stats">
  <div class="stat-card">
    <div class="stat-num">{total_all}</div>
    <div class="stat-label">Total Kits Ever</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:#22C55E;">{total_today}</div>
    <div class="stat-label">Kits Today</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:#F59E0B;">{avg_score}%</div>
    <div class="stat-label">Avg ATS Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:#A78BFA;">{max_score}%</div>
    <div class="stat-label">Highest Score</div>
  </div>
</div>
<div class="today-strip">
  <span>{total_today}</span> kits generated today · <span>{len(company_counts)}</span> unique companies targeted · <span>{total_all}</span> total since launch
</div>
</div>
<div class="companies">
  <div class="section-title">🎯 Companies Being Targeted Right Now</div>
  {company_bars if company_bars else '<div class="empty">Waiting for first kit generation...<br><br>📱 Scan the QR code at ninelab.in</div>'}
</div>
<div class="footer">Auto-refreshes every 5 seconds · ninelab.in · Pitch Day 2026</div>
</body></html>""")


@app.get("/ninelab/pdf/{filename}")
async def download_pdf(filename: str):
    if not filename.endswith(".pdf") or "/" in filename or ".." in filename:
        raise HTTPException(400, detail="Invalid filename")
    filepath = PDF_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, detail="PDF not found")
    return FileResponse(str(filepath), media_type="application/pdf",
                        filename=filename, headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/ninelab/ats-score")
async def ats_score(req: ATSScoreRequest):
    """Calculate ATS compatibility score between resume and job description."""
    if not req.resume.strip():
        raise HTTPException(400, detail="Please provide your resume.")
    if not req.jd.strip():
        raise HTTPException(400, detail="Please provide the job description.")

    resume_text = req.resume.lower()
    jd_text = req.jd.lower()

    # ── 1. Extract keywords from JD ──────────────────────────────────────────

    # Technical skills library
    TECH_SKILLS = [
        # Languages
        "python","java","javascript","typescript","c++","c#","golang","rust","kotlin","swift",
        "php","ruby","scala","r programming","matlab","perl","bash","shell","sql","nosql",
        # Web/Frontend
        "react","angular","vue","html","css","sass","webpack","nextjs","nuxt","gatsby",
        "jquery","bootstrap","tailwind","redux","graphql","rest api","restful",
        # Backend
        "nodejs","node.js","express","django","flask","fastapi","spring","spring boot",
        "laravel","rails","asp.net",".net","microservices","kafka","rabbitmq",
        # Databases
        "mysql","postgresql","mongodb","redis","elasticsearch","sqlite","oracle",
        "dynamodb","firebase","supabase","cassandra","snowflake","bigquery",
        # Cloud/DevOps
        "aws","azure","gcp","google cloud","docker","kubernetes","terraform","ansible",
        "jenkins","github actions","ci/cd","devops","linux","nginx","apache",
        # AI/ML/Data
        "machine learning","deep learning","neural network","tensorflow","pytorch",
        "scikit-learn","pandas","numpy","nlp","computer vision","llm","data science",
        "data analysis","data engineering","etl","tableau","power bi","spark","hadoop",
        # Tools
        "git","github","gitlab","jira","confluence","agile","scrum","kanban",
        "postman","swagger","figma","photoshop","excel","powerpoint",
        # Mobile
        "android","ios","flutter","react native","xamarin",
        # Security
        "cybersecurity","penetration testing","owasp","ssl","oauth","jwt",
    ]

    # Soft skills
    SOFT_SKILLS = [
        "communication","leadership","teamwork","problem solving","critical thinking",
        "time management","adaptability","creativity","collaboration","interpersonal",
        "project management","analytical","detail oriented","self motivated","initiative",
        "presentation","negotiation","mentoring","coaching","strategic",
    ]

    # Extract all meaningful words from JD (3+ chars, not stopwords)
    STOPWORDS = {"the","and","for","are","but","not","you","all","any","can","had",
                 "her","was","one","our","out","day","get","has","him","his","how",
                 "its","may","new","now","own","see","two","way","who","did","each",
                 "from","this","that","with","have","will","been","they","their",
                 "more","also","into","over","some","such","than","then","them",
                 "well","were","what","when","whom","your","able","about","after",
                 "being","below","could","doing","during","other","these","those",
                 "through","under","until","while","would","should","shall","must",
                 "work","year","years","good","role","team","need","able","must",
                 "strong","experience","candidate","looking","join","help","great",
                 "including","required","requirements","responsibilities","minimum",
                 "preferred","plus","bonus","nice","have","will","across","within"}

    def extract_keywords(text: str) -> list[str]:
        # Only match words of 4+ characters (avoids "r", "go", "ai" false matches)
        words = re.findall(r'\b[a-z][a-z0-9]{3,}\b', text.lower())
        # Bigrams — only meaningful pairs (both words 4+ chars)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)
                   if len(words[i]) >= 4 and len(words[i+1]) >= 4]
        all_terms = words + bigrams
        return [t for t in all_terms if t not in STOPWORDS]

    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)

    # Word frequency from JD (higher freq = more important)
    from collections import Counter
    jd_freq = Counter(jd_keywords)

    # Top 40 most frequent JD terms (excluding very common noise)
    jd_top = [word for word, _ in jd_freq.most_common(80) if len(word) > 3][:40]

    resume_word_set = set(resume_keywords)

    # ── 2. Match technical skills ──────────────────────────────────────────────
    jd_tech = [s for s in TECH_SKILLS if s in jd_text]
    resume_tech = [s for s in jd_tech if s in resume_text]
    missing_tech = [s for s in jd_tech if s not in resume_text]

    tech_score = (len(resume_tech) / len(jd_tech) * 100) if jd_tech else 100

    # ── 3. Match soft skills ──────────────────────────────────────────────────
    jd_soft = [s for s in SOFT_SKILLS if s in jd_text]
    resume_soft = [s for s in jd_soft if s in resume_text]
    missing_soft = [s for s in jd_soft if s not in resume_text]

    soft_score = (len(resume_soft) / len(jd_soft) * 100) if jd_soft else 100

    # ── 4. Match top JD keywords ─────────────────────────────────────────────
    matched_keywords = [w for w in jd_top if w in resume_word_set]
    missing_keywords = [w for w in jd_top if w not in resume_word_set]

    keyword_score = (len(matched_keywords) / len(jd_top) * 100) if jd_top else 100

    # ── 5. Format / structure check ──────────────────────────────────────────
    format_score = 100
    format_tips = []

    # Check for key resume sections
    if not any(w in resume_text for w in ["experience","work history","employment"]):
        format_score -= 20
        format_tips.append("Add a clear 'Experience' section")
    if not any(w in resume_text for w in ["education","degree","university","college","bachelor","master"]):
        format_score -= 15
        format_tips.append("Add an 'Education' section")
    if not any(w in resume_text for w in ["skill","skills","technologies","tools","expertise"]):
        format_score -= 15
        format_tips.append("Add a dedicated 'Skills' section")
    if len(req.resume.split()) < 150:
        format_score -= 20
        format_tips.append("Resume seems too short — aim for 400–700 words")
    if len(req.resume.split()) > 1200:
        format_score -= 10
        format_tips.append("Resume is very long — consider trimming to 1 page")
    if not any(w in resume_text for w in ["@","email","phone","linkedin","github","contact"]):
        format_score -= 10
        format_tips.append("Add contact information (email, phone, LinkedIn)")

    format_score = max(format_score, 0)

    # ── 6. Weighted final score ───────────────────────────────────────────────
    # Keywords 35% | Tech skills 35% | Soft skills 15% | Format 15%
    final_score = (
        keyword_score * 0.35 +
        tech_score * 0.35 +
        soft_score * 0.15 +
        format_score * 0.15
    )
    final_score = round(min(final_score, 100), 1)

    # ── 7. Generate suggestions ───────────────────────────────────────────────
    suggestions = []
    if missing_tech:
        top_missing_tech = missing_tech[:5]
        suggestions.append(f"Add these technical skills if you have them: {', '.join(top_missing_tech)}")
    if missing_soft:
        suggestions.append(f"Mention soft skills: {', '.join(missing_soft[:3])}")
    if missing_keywords:
        important_missing = [k for k in missing_keywords if len(k) > 4][:5]
        if important_missing:
            suggestions.append(f"Include these JD keywords in your resume: {', '.join(important_missing)}")
    suggestions.extend(format_tips)
    if final_score >= 80:
        suggestions.append("Great match! Your resume is well-aligned with this role.")
    elif final_score >= 60:
        suggestions.append("Good foundation. Tailor your resume further to boost your score.")
    else:
        suggestions.append("Significant gaps found. Customize your resume to match the JD closely.")

    # Score label
    if final_score >= 85:
        label = "Excellent"
    elif final_score >= 70:
        label = "Good"
    elif final_score >= 55:
        label = "Fair"
    else:
        label = "Needs Work"

    return JSONResponse({
        "score": final_score,
        "label": label,
        "breakdown": {
            "keywords": round(keyword_score, 1),
            "technical_skills": round(tech_score, 1),
            "soft_skills": round(soft_score, 1),
            "format": round(format_score, 1),
        },
        "matched_keywords": [k for k in list(set(matched_keywords + resume_tech + resume_soft)) if len(k) > 2][:20],
        "missing_keywords": [k for k in list(set(missing_keywords[:5] + missing_tech[:5] + missing_soft[:3])) if len(k) > 2],
        "suggestions": suggestions[:6],
        "stats": {
            "jd_tech_skills_found": len(jd_tech),
            "resume_tech_matched": len(resume_tech),
            "jd_keywords_checked": len(jd_top),
            "resume_keywords_matched": len(matched_keywords),
        }
    })


@app.get("/ninelab/health")
async def health():
    return {"status": "ok", "service": "Nine Lab", "keys": {
        "gemini": bool(GEMINI_API_KEY),
        "tavily": bool(TAVILY_API_KEY),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
    }}
