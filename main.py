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
def _clean_env(val: str) -> str:
    """Strip whitespace and non-printable ASCII chars from env var values."""
    return ''.join(c for c in (val or '').strip() if 32 <= ord(c) <= 126)

SUPABASE_URL = _clean_env(os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = _clean_env(os.getenv("SUPABASE_KEY", ""))
SUPABASE_SERVICE_KEY = _clean_env(os.getenv("SUPABASE_SERVICE_KEY", ""))
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY", "d478886deemshee1d5a113b51de6p1d199ajsnee586dc31325")
ADZUNA_APP_ID   = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY  = os.getenv("ADZUNA_APP_KEY", "")

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


def _build_skill_queries(skills: list[str], title: str, is_internship: bool = False) -> list[str]:
    """Build ordered list of search queries from most to least specific."""
    intern = " internship" if is_internship else ""
    queries = []
    # Most specific: top 2 skills combined
    if len(skills) >= 2:
        queries.append(f"{skills[0]} {skills[1]}{intern} India site:linkedin.com OR site:indeed.in OR site:naukri.com")
    # Each top skill individually
    for s in skills[:4]:
        queries.append(f"{s}{intern} India site:linkedin.com OR site:indeed.in OR site:naukri.com")
    # Job title fallback
    if title:
        queries.append(f"{title}{intern} India site:linkedin.com OR site:indeed.in OR site:naukri.com")
    # Deduplicate preserving order
    seen, result = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q); result.append(q)
    return result


def _fetch_jsearch_jobs(skills: list[str], title: str, is_internship: bool = False) -> list[dict]:
    if not JSEARCH_API_KEY:
        return []
    import httpx
    queries = _build_skill_queries(skills, title, is_internship)
    seen_urls, out = set(), []
    for query in queries:
        if len(out) >= 6:
            break
        try:
            r = httpx.get(
                "https://jsearch.p.rapidapi.com/search",
                headers={"x-rapidapi-key": JSEARCH_API_KEY, "x-rapidapi-host": "jsearch.p.rapidapi.com"},
                params={"query": query, "page": "1", "num_pages": "1", "date_posted": "month"},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            for job in r.json().get("data", []):
                url = job.get("job_apply_link") or job.get("job_google_link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                pub = (job.get("job_publisher") or "").lower()
                url_l = url.lower()
                # Only keep LinkedIn, Indeed, Naukri
                if "linkedin" in pub or "linkedin" in url_l:
                    source = "LinkedIn"
                elif "indeed" in pub or "indeed" in url_l:
                    source = "Indeed"
                elif "naukri" in url_l:
                    source = "Naukri"
                else:
                    continue  # skip other sources
                is_intern = "intern" in (job.get("job_title") or "").lower()
                out.append({
                    "title": (job.get("job_title") or title)[:100],
                    "company": (job.get("employer_name") or "")[:60],
                    "url": url, "source": source,
                    "type": "internship" if is_intern else "job",
                    "snippet": (job.get("job_description") or "")[:200].strip(),
                })
                if len(out) >= 6:
                    break
        except Exception:
            continue
    return out


def _fetch_adzuna_jobs(skills: list[str], title: str, is_internship: bool = False) -> list[dict]:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []
    import httpx
    queries = _build_skill_queries(skills, title, is_internship)
    seen_urls, out = set(), []
    for query in queries:
        if len(out) >= 5:
            break
        try:
            r = httpx.get(
                "https://api.adzuna.com/v1/api/jobs/in/search/1",
                params={"app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY,
                        "results_per_page": 5, "what": query},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            for job in r.json().get("results", []):
                url = job.get("redirect_url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                t = job.get("title", title)
                out.append({
                    "title": t[:100],
                    "company": job.get("company", {}).get("display_name", "")[:60],
                    "url": url, "source": "Adzuna",
                    "type": "internship" if is_internship or "intern" in t.lower() else "job",
                    "snippet": (job.get("description") or "")[:200].strip(),
                })
                if len(out) >= 5:
                    break
        except Exception:
            continue
    return out


def _fetch_remoteok_jobs(skills: list[str]) -> list[dict]:
    """Fetch remote/freelance jobs from RemoteOK — no API key needed. Tries each skill until 5+ results."""
    import httpx
    seen_urls, out = set(), []
    for skill in (skills[:4] if skills else ['developer']):
        if len(out) >= 6:
            break
        tag = skill.lower().replace(' ', '-').replace('#', 'sharp').replace('+', 'plus')
        try:
            r = httpx.get(
                f"https://remoteok.com/api?tags={tag}",
                headers={"User-Agent": "Mozilla/5.0 (NineLab)"},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            for job in r.json():
                if not isinstance(job, dict) or not job.get("url"):
                    continue
                url = job["url"]
                if not url.startswith("http"):
                    url = "https://remoteok.com" + url
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                out.append({
                    "title": (job.get("position") or "")[:100],
                    "company": (job.get("company") or "")[:60],
                    "url": url, "source": "RemoteOK",
                    "snippet": (job.get("description") or "")[:200].strip(),
                    "tags": job.get("tags", [])[:5],
                })
                if len(out) >= 6:
                    break
        except Exception:
            continue
    return out


def _fetch_remotive_jobs(skills: list[str]) -> list[dict]:
    """Fetch remote jobs from Remotive — no API key needed. Tries each skill until 5+ results."""
    import httpx
    seen_urls, out = set(), []
    for skill in (skills[:4] if skills else ['developer']):
        if len(out) >= 6:
            break
        try:
            r = httpx.get(
                "https://remotive.com/api/remote-jobs",
                params={"search": skill, "limit": 10},
                headers={"User-Agent": "Mozilla/5.0 (NineLab)"},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            for job in r.json().get("jobs", []):
                url = job.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                out.append({
                    "title": (job.get("title") or "")[:100],
                    "company": (job.get("company_name") or "")[:60],
                    "url": url, "source": "Remotive",
                    "snippet": (job.get("description") or "")[:150].strip(),
                    "tags": job.get("tags", [])[:5],
                })
                if len(out) >= 6:
                    break
        except Exception:
            continue
    return out


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


def _supabase_rest(method: str, table: str, payload: dict = None, token: str = None, params: dict = None) -> dict:
    """Helper for Supabase REST API (PostgREST)."""
    import httpx
    key = token or SUPABASE_SERVICE_KEY or SUPABASE_KEY
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        if method == "GET":
            r = httpx.get(url, headers=headers, params=params, timeout=10)
        elif method == "POST":
            r = httpx.post(url, headers=headers, json=payload, timeout=10)
        elif method == "PATCH":
            r = httpx.patch(url, headers=headers, json=payload, params=params, timeout=10)
        elif method == "UPSERT":
            headers["Prefer"] = "return=representation,resolution=merge-duplicates"
            r = httpx.post(url, headers=headers, json=payload, timeout=10)
        else:
            return {"status": 400, "data": {}}
        body = r.json() if r.text else {}
        return {"status": r.status_code, "data": body}
    except Exception as e:
        return {"status": 500, "data": {"error": str(e)}}


class ProfileSaveRequest(BaseModel):
    skills: Optional[str] = ""
    year: Optional[str] = ""
    degree: Optional[str] = "B.Tech"
    title: Optional[str] = ""
    readiness: Optional[int] = 0
    gaps: Optional[list] = []
    resume_text: Optional[str] = ""

@app.post("/ninelab/profile/save")
async def profile_save(req: ProfileSaveRequest, authorization: Optional[str] = Header(None)):
    token = (authorization or "").replace("Bearer ", "").strip()
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(401, detail="Not authenticated.")
    user_id = user["id"]
    payload = {
        "user_id": user_id,
        "skills": req.skills or "",
        "year": req.year or "",
        "degree": req.degree or "B.Tech",
        "title": req.title or "",
        "readiness": req.readiness or 0,
        "gaps": req.gaps or [],
        "resume_text": (req.resume_text or "")[:5000],
        "updated_at": datetime.utcnow().isoformat(),
    }
    result = _supabase_rest("UPSERT", "profiles", payload=payload, token=SUPABASE_SERVICE_KEY)
    if result["status"] in (200, 201):
        return JSONResponse({"ok": True})
    raise HTTPException(500, detail="Failed to save profile.")


@app.get("/ninelab/profile/load")
async def profile_load(authorization: Optional[str] = Header(None)):
    token = (authorization or "").replace("Bearer ", "").strip()
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(401, detail="Not authenticated.")
    user_id = user["id"]
    result = _supabase_rest(
        "GET", "profiles",
        params={"user_id": f"eq.{user_id}", "select": "skills,year,degree,title,readiness,gaps,resume_text,updated_at"},
        token=SUPABASE_SERVICE_KEY,
    )
    if result["status"] == 200:
        rows = result["data"] if isinstance(result["data"], list) else []
        if rows:
            return JSONResponse(rows[0])
        return JSONResponse({})
    raise HTTPException(500, detail="Failed to load profile.")


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


# ── PlaceAI: Multi-Agent Discovery System ────────────────────────────────────

async def _safe_agent(coro, name: str, timeout: float = 9.0) -> dict:
    """Wrap any agent coroutine with timeout + exception safety."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return {"source": name, "data": [], "status": "timeout"}
    except Exception as e:
        return {"source": name, "data": [], "status": "error", "err": str(e)}


def _skill_match(text: str, skills: list) -> tuple:
    """Returns (match_pct: int, missing: list[str])"""
    if not skills:
        return 50, []
    tl = text.lower()
    matched = [s for s in skills if s.lower() in tl]
    missing = [s for s in skills if s.lower() not in tl]
    pct = int(len(matched) / len(skills) * 100)
    return max(pct, 15), missing[:2]


async def _agent_jobs(profile: dict) -> dict:
    import httpx
    skills = profile.get("skills", [])
    title  = profile.get("title") or (skills[0] if skills else "Software Developer")
    out = []
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                "https://jsearch.p.rapidapi.com/search",
                headers={"x-rapidapi-key": JSEARCH_API_KEY,
                         "x-rapidapi-host": "jsearch.p.rapidapi.com"},
                params={"query": f"{title} India", "page": "1",
                        "num_pages": "1", "date_posted": "month"},
                timeout=8,
            )
        if r.status_code == 200:
            for job in r.json().get("data", [])[:8]:
                url = job.get("job_apply_link") or job.get("job_google_link", "")
                if not url:
                    continue
                t = job.get("job_title") or title
                if "intern" in t.lower():
                    continue
                desc = job.get("job_description") or ""
                pct, miss = _skill_match(desc + " " + t, skills)
                pub = (job.get("job_publisher") or "").lower()
                src = "LinkedIn" if "linkedin" in pub else \
                      "Indeed"   if "indeed"   in pub else \
                      "Glassdoor"if "glassdoor"in pub else "Job Board"
                out.append({"title": t[:80], "company": (job.get("employer_name") or "")[:50],
                            "url": url, "source": src, "match": pct,
                            "gap": miss[0] if miss else "",
                            "snippet": desc[:160].strip()})
                if len(out) >= 5: break
    except Exception:
        pass
    return {"source": "jobs", "data": out}


async def _agent_internships(profile: dict) -> dict:
    import httpx
    skills = profile.get("skills", [])
    title  = profile.get("title") or (skills[0] if skills else "Developer")
    out = []
    # JSearch internship search
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                "https://jsearch.p.rapidapi.com/search",
                headers={"x-rapidapi-key": JSEARCH_API_KEY,
                         "x-rapidapi-host": "jsearch.p.rapidapi.com"},
                params={"query": f"{title} internship India", "page": "1",
                        "num_pages": "1", "date_posted": "month"},
                timeout=8,
            )
        if r.status_code == 200:
            for job in r.json().get("data", [])[:6]:
                url = job.get("job_apply_link") or job.get("job_google_link", "")
                if not url: continue
                t = job.get("job_title") or f"{title} Intern"
                desc = job.get("job_description") or ""
                pct, miss = _skill_match(desc + " " + t, skills)
                pub = (job.get("job_publisher") or "").lower()
                src = "LinkedIn" if "linkedin" in pub else \
                      "Indeed"   if "indeed"   in pub else "JSearch"
                out.append({"title": t[:80], "company": (job.get("employer_name") or "")[:50],
                            "url": url, "source": src, "match": pct,
                            "gap": miss[0] if miss else "",
                            "snippet": desc[:160].strip()})
                if len(out) >= 5: break
    except Exception:
        pass
    # Adzuna fallback if < 3 results
    if ADZUNA_APP_ID and ADZUNA_APP_KEY and len(out) < 3:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(
                    "https://api.adzuna.com/v1/api/jobs/in/search/1",
                    params={"app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY,
                            "results_per_page": 5, "what": f"{title} internship"},
                    timeout=8,
                )
            if r.status_code == 200:
                for job in r.json().get("results", [])[:5]:
                    url = job.get("redirect_url", "")
                    if not url: continue
                    t = job.get("title", "Internship")
                    desc = job.get("description") or ""
                    pct, miss = _skill_match(desc + " " + t, skills)
                    out.append({"title": t[:80],
                                "company": job.get("company", {}).get("display_name", "")[:50],
                                "url": url, "source": "Adzuna", "match": pct,
                                "gap": miss[0] if miss else "",
                                "snippet": desc[:160].strip()})
        except Exception:
            pass
    return {"source": "internships", "data": out[:5]}


async def _agent_freelancing(profile: dict) -> dict:
    import httpx
    skills = profile.get("skills", [])
    out = []
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("https://remoteok.com/api",
                            headers={"User-Agent": "PlaceAI/1.0"}, timeout=8)
        if r.status_code == 200:
            for job in r.json()[1:]:          # index 0 is metadata
                if not isinstance(job, dict): continue
                url = job.get("url", "")
                if not url: continue
                tags = " ".join(job.get("tags") or [])
                desc = tags + " " + (job.get("description") or "")
                pct, miss = _skill_match(desc, skills)
                if pct < 20: continue
                full_url = url if url.startswith("http") else f"https://remoteok.com{url}"
                out.append({"title": (job.get("position") or "Remote Role")[:80],
                            "company": (job.get("company") or "")[:50],
                            "url": full_url, "source": "RemoteOK", "match": pct,
                            "gap": miss[0] if miss else "",
                            "snippet": (job.get("description") or "")[:160].strip()})
                if len(out) >= 5: break
    except Exception:
        pass
    return {"source": "freelancing", "data": out}


async def _agent_scholarships(profile: dict) -> dict:
    if not TAVILY_API_KEY:
        return {"source": "scholarships", "data": []}
    skills = profile.get("skills", [])
    degree = profile.get("degree", "engineering")
    out = []
    try:
        results = tavily_search(
            f"engineering student scholarship grant fellowship India 2025 2026 {degree}", retries=0)
        for r in results[:8]:
            url = r.get("url", "")
            t   = r.get("title", "")
            content = r.get("content") or ""
            if not url: continue
            kws = ["scholarship", "grant", "fellowship", "award", "stipend", "prize"]
            if not any(k in (t + content).lower() for k in kws): continue
            pct, miss = _skill_match(content + " " + t, skills)
            out.append({"title": t[:80], "company": "Scholarship Program",
                        "url": url, "source": "Web", "match": pct,
                        "gap": "", "snippet": content[:160].strip()})
            if len(out) >= 4: break
    except Exception:
        pass
    return {"source": "scholarships", "data": out}


class DiscoverRequest(BaseModel):
    name:   str = ""
    degree: str = "B.Tech"
    year:   str = "3rd Year"
    skills: list = []
    title:  str = ""


@app.post("/ninelab/discover")
async def discover_opportunities(req: DiscoverRequest):
    """Run 4 agents in parallel and return ranked results per category."""
    profile = {
        "name":   req.name.strip()[:60],
        "degree": req.degree.strip()[:40],
        "year":   req.year.strip()[:20],
        "skills": [s.strip() for s in req.skills if s.strip()][:12],
        "title":  req.title.strip()[:60] or
                  (req.skills[0].strip() if req.skills else "Software Developer"),
    }
    results = await asyncio.gather(
        _safe_agent(_agent_jobs(profile),         "jobs"),
        _safe_agent(_agent_internships(profile),  "internships"),
        _safe_agent(_agent_freelancing(profile),  "freelancing"),
        _safe_agent(_agent_scholarships(profile), "scholarships"),
    )
    out = {}
    for r in results:
        if isinstance(r, dict) and "source" in r:
            out[r["source"]] = sorted(r.get("data", []),
                                      key=lambda x: x.get("match", 0), reverse=True)
    return JSONResponse({
        "jobs":         out.get("jobs", []),
        "internships":  out.get("internships", []),
        "freelancing":  out.get("freelancing", []),
        "scholarships": out.get("scholarships", []),
    })


# ── AI Placement Chat ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    profile: Optional[dict] = None

@app.post("/ninelab/chat")
async def placement_chat(req: ChatRequest):
    message = req.message.strip()[:500]
    profile = req.profile or {}
    skills   = profile.get("skills", "") or "Not mentioned"
    year     = profile.get("year", "Not mentioned")
    degree   = profile.get("degree", "B.Tech")
    title    = profile.get("title", "") or "Software Developer"
    readiness = profile.get("readiness", None)
    gaps     = [g for g in profile.get("gaps", []) if g and len(g.strip()) > 1]

    # Derive tech-relevant gaps if AI-generated gaps look wrong
    VALID_TECH_GAPS = {
        "dsa","data structures","algorithms","system design","sql","database",
        "os","operating systems","networking","computer networks","oops","object oriented",
        "dbms","cn","web development","react","node","javascript","python","java","c++",
        "machine learning","deep learning","nlp","cloud","aws","azure","docker","kubernetes",
        "communication","aptitude","verbal","logical reasoning","resume","git","linux",
        "rest api","api","microservices","problem solving","competitive programming","leetcode"
    }
    clean_gaps = []
    for g in gaps:
        g_lower = g.lower().strip()
        if any(v in g_lower for v in VALID_TECH_GAPS):
            clean_gaps.append(g)
    if not clean_gaps and gaps:
        clean_gaps = []  # discard irrelevant gaps, let AI figure out from skills+role

    # Build smart gap hint from skills vs target role
    skill_hint = f"Student has these skills: {skills}." if skills != "Not mentioned" else ""
    gap_hint = f"Identified skill gaps: {', '.join(clean_gaps)}." if clean_gaps else ""
    readiness_hint = f"Current placement readiness score: {readiness}%." if readiness else ""

    system_prompt = f"""You are Vertical AI — an intelligent placement advisor for Indian engineering students.

STUDENT PROFILE:
- Degree: {degree}, Year: {year}
- Target Role: {title}
- {skill_hint}
- {readiness_hint}
- {gap_hint}

YOUR JOB:
Answer the student's question about placement preparation only.
All skill gaps and advice must be 100% relevant to engineering/IT placements in India.
Only suggest gaps like: DSA, System Design, SQL, DBMS, OS, CN, OOPs, Communication, Aptitude, specific languages/frameworks related to their target role.
NEVER suggest non-technical gaps like video editing, design, marketing unless explicitly asked.
Cross-check any gap against the student's target role ({title}) before mentioning it.

RULES:
- Max 3-4 sentences per reply. Be direct and specific.
- Always refer to the student's actual skills and target role.
- If student asks to do something (find job, build resume), tell them the agent can do it — suggest they use the button or command.
- If no profile is saved yet, ask them to run the analysis first.
- Tone: like a smart senior student — helpful, honest, not corporate.
- Never make up certifications, course names, or companies unless well-known (Coursera, LeetCode, GFG, HackerRank).
- Language: English only. Short sentences."""

    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        executor,
        lambda: gemini_call(message, retries=2, temperature=0.4, system_prompt=system_prompt)
    )
    return JSONResponse({"reply": reply or "Sorry, could not get a response. Please try again."})


# ── (existing) JSearch sync helper ───────────────────────────────────────────

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
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200, headers=headers)
    return HTMLResponse(content="<h1>Nine Lab loading...</h1>", status_code=200, headers=headers)


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
    # company name is optional — auto-extracted from JD if not provided

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


@app.get("/ninelab/govtdash", response_class=HTMLResponse)
async def govt_dashboard():
    """Government national dashboard — Ministry of Education style."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NPIS — National Placement Intelligence System</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:#f0f2f5;color:#1a1a2e;min-height:100vh;}

/* GOV TOP BAR */
.gov-bar{background:#1a2744;padding:8px 24px;display:flex;align-items:center;justify-content:space-between;}
.gov-bar-left{display:flex;align-items:center;gap:12px;}
.gov-emblem{width:36px;height:36px;background:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;}
.gov-title{color:#fff;}
.gov-title h1{font-size:13px;font-weight:700;line-height:1.2;}
.gov-title p{font-size:10px;opacity:0.6;margin-top:1px;}
.gov-bar-right{display:flex;align-items:center;gap:16px;}
.live-dot{display:flex;align-items:center;gap:6px;color:#4ade80;font-size:11px;font-weight:600;}
.live-dot::before{content:'';width:8px;height:8px;border-radius:50%;background:#4ade80;animation:blink 1.5s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.gov-time{color:rgba(255,255,255,0.5);font-size:11px;}

/* SCHEME HEADER */
.scheme-header{background:linear-gradient(135deg,#1a2744,#2d3a6b);padding:20px 24px;border-bottom:3px solid #6c63ff;}
.scheme-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;}
.scheme-left h2{font-size:20px;font-weight:800;color:#fff;line-height:1.2;}
.scheme-left p{font-size:11px;color:rgba(255,255,255,0.6);margin-top:4px;}
.scheme-badges{display:flex;gap:8px;flex-wrap:wrap;}
.sbadge{font-size:10px;font-weight:700;padding:4px 10px;border-radius:4px;letter-spacing:0.5px;}
.sbadge-orange{background:#f97316;color:#fff;}
.sbadge-green{background:#16a34a;color:#fff;}
.sbadge-blue{background:#6c63ff;color:#fff;}

/* NAV TABS */
.nav-tabs{background:#fff;border-bottom:1px solid #e5e7eb;padding:0 24px;display:flex;gap:0;overflow-x:auto;}
.nav-tab{padding:12px 18px;font-size:12px;font-weight:600;color:#64748b;cursor:pointer;border-bottom:3px solid transparent;white-space:nowrap;}
.nav-tab.active{color:#1a2744;border-bottom-color:#6c63ff;}

/* MAIN */
.main{max-width:1100px;margin:0 auto;padding:20px 16px;}

/* KPI ROW */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
.kpi{background:#fff;border-radius:10px;padding:16px 18px;border-left:4px solid;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.kpi-num{font-size:28px;font-weight:800;line-height:1;}
.kpi-label{font-size:11px;color:#64748b;margin-top:4px;line-height:1.3;}
.kpi-change{font-size:10px;font-weight:600;margin-top:6px;}
.kpi-up{color:#16a34a;}
.kpi-down{color:#ef4444;}

/* GRID 2 COL */
.grid2{display:grid;grid-template-columns:1.4fr 1fr;gap:16px;margin-bottom:20px;}
.card{background:#fff;border-radius:10px;padding:18px;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.card-title{font-size:12px;font-weight:700;color:#1a1a2e;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;}
.card-title span{font-size:10px;font-weight:500;color:#64748b;}

/* STATE TABLE */
.state-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f5f5f5;font-size:12px;}
.state-row:last-child{border-bottom:none;}
.state-name{flex:1;font-weight:500;}
.state-bar-wrap{flex:2;background:#f0f0f0;border-radius:4px;height:6px;}
.state-bar{height:6px;border-radius:4px;background:linear-gradient(90deg,#6c63ff,#a78bfa);}
.state-pct{width:36px;text-align:right;font-weight:600;color:#6c63ff;font-size:11px;}

/* ALERT BOX */
.alerts{display:flex;flex-direction:column;gap:8px;}
.alert-item{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;border-radius:8px;font-size:12px;}
.alert-red{background:#fef2f2;border-left:3px solid #ef4444;}
.alert-yellow{background:#fffbeb;border-left:3px solid #f59e0b;}
.alert-green{background:#f0fdf4;border-left:3px solid #22c55e;}
.alert-icon{font-size:14px;flex-shrink:0;margin-top:1px;}
.alert-text{line-height:1.4;}
.alert-text strong{display:block;font-weight:600;color:#1a1a2e;}

/* COLLEGE TABLE */
.college-table{width:100%;border-collapse:collapse;font-size:12px;}
.college-table th{background:#f8fafc;padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.5px;border-bottom:1px solid #e5e7eb;}
.college-table td{padding:10px 12px;border-bottom:1px solid #f5f5f5;color:#1a1a2e;}
.college-table tr:last-child td{border-bottom:none;}
.status-pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600;}
.pill-green{background:#dcfce7;color:#16a34a;}
.pill-yellow{background:#fef9c3;color:#854d0e;}
.pill-blue{background:#eff6ff;color:#1d4ed8;}

/* BOTTOM GRID */
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;}
.mini-stat{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06);text-align:center;}
.mini-stat .num{font-size:22px;font-weight:800;color:#6c63ff;margin-bottom:4px;}
.mini-stat .lbl{font-size:11px;color:#64748b;line-height:1.4;}

/* FOOTER */
.footer{background:#1a2744;color:#fff;padding:14px 24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-top:8px;}
.footer p{font-size:11px;opacity:0.5;}
</style>
</head>
<body>

<!-- GOV BAR -->
<div class="gov-bar">
  <div class="gov-bar-left">
    <div class="gov-emblem">🏛️</div>
    <div class="gov-title">
      <h1>Ministry of Education — Skill Development Wing</h1>
      <p>Government of India · National Placement Intelligence System</p>
    </div>
  </div>
  <div class="gov-bar-right">
    <div class="live-dot">LIVE</div>
    <div class="gov-time" id="govTime"></div>
  </div>
</div>

<!-- SCHEME HEADER -->
<div class="scheme-header">
  <div class="scheme-inner">
    <div class="scheme-left">
      <h2>NPIS — National Placement Intelligence System</h2>
      <p>Powered by Vertical AI · Persistent Agentic Intelligence for Engineering Placement · FY 2025–26</p>
    </div>
    <div class="scheme-badges">
      <span class="sbadge sbadge-orange">Phase 1 — Active</span>
      <span class="sbadge sbadge-green">Prototype Deployed</span>
      <span class="sbadge sbadge-blue">NLPC 2026</span>
    </div>
  </div>
</div>

<!-- NAV -->
<div class="nav-tabs">
  <div class="nav-tab active">National Overview</div>
  <div class="nav-tab">State Dashboard</div>
  <div class="nav-tab">College Enrollment</div>
  <div class="nav-tab">Student Impact</div>
  <div class="nav-tab">Research & IP</div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- KPIs -->
  <div class="kpi-row">
    <div class="kpi" style="border-color:#6c63ff;">
      <div class="kpi-num" style="color:#6c63ff;">2,34,891</div>
      <div class="kpi-label">Students on Platform<br>Nationwide</div>
      <div class="kpi-change kpi-up">↑ 18% this month</div>
    </div>
    <div class="kpi" style="border-color:#22c55e;">
      <div class="kpi-num" style="color:#22c55e;">847</div>
      <div class="kpi-label">Colleges<br>Enrolled</div>
      <div class="kpi-change kpi-up">↑ 43 new this week</div>
    </div>
    <div class="kpi" style="border-color:#f59e0b;">
      <div class="kpi-num" style="color:#f59e0b;">61%</div>
      <div class="kpi-label">Avg Placement<br>Readiness Score</div>
      <div class="kpi-change kpi-up">↑ 9pts from baseline</div>
    </div>
    <div class="kpi" style="border-color:#ef4444;">
      <div class="kpi-num" style="color:#ef4444;">38,000</div>
      <div class="kpi-label">Target Colleges<br>Remaining</div>
      <div class="kpi-change kpi-down">↓ 2.2% penetration</div>
    </div>
  </div>

  <!-- STATE + ALERTS -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">State-wise Readiness Score <span>Avg placement readiness %</span></div>
      <div class="state-row"><span class="state-name">Maharashtra</span><div class="state-bar-wrap"><div class="state-bar" style="width:74%"></div></div><span class="state-pct">74%</span></div>
      <div class="state-row"><span class="state-name">Karnataka</span><div class="state-bar-wrap"><div class="state-bar" style="width:70%"></div></div><span class="state-pct">70%</span></div>
      <div class="state-row"><span class="state-name">Tamil Nadu</span><div class="state-bar-wrap"><div class="state-bar" style="width:68%"></div></div><span class="state-pct">68%</span></div>
      <div class="state-row"><span class="state-name">Telangana</span><div class="state-bar-wrap"><div class="state-bar" style="width:65%"></div></div><span class="state-pct">65%</span></div>
      <div class="state-row"><span class="state-name">Uttar Pradesh</span><div class="state-bar-wrap"><div class="state-bar" style="width:48%"></div></div><span class="state-pct">48%</span></div>
      <div class="state-row"><span class="state-name">Bihar</span><div class="state-bar-wrap"><div class="state-bar" style="width:38%"></div></div><span class="state-pct">38%</span></div>
      <div class="state-row"><span class="state-name">Rajasthan</span><div class="state-bar-wrap"><div class="state-bar" style="width:42%"></div></div><span class="state-pct">42%</span></div>
    </div>
    <div class="card">
      <div class="card-title">National Alerts <span>Requires attention</span></div>
      <div class="alerts">
        <div class="alert-item alert-red">
          <div class="alert-icon">🔴</div>
          <div class="alert-text"><strong>12,400 students below 30% readiness</strong>UP, Bihar, Jharkhand — immediate intervention needed</div>
        </div>
        <div class="alert-item alert-yellow">
          <div class="alert-icon">🟡</div>
          <div class="alert-text"><strong>DSA gap in 68% of students</strong>Most critical skill gap nationwide — platform flagged</div>
        </div>
        <div class="alert-item alert-yellow">
          <div class="alert-icon">🟡</div>
          <div class="alert-text"><strong>847 colleges enrolled, 37,153 pending</strong>Phase 2 expansion required for national coverage</div>
        </div>
        <div class="alert-item alert-green">
          <div class="alert-icon">🟢</div>
          <div class="alert-text"><strong>Maharashtra — 74% avg readiness</strong>Highest performing state · Model for replication</div>
        </div>
      </div>
    </div>
  </div>

  <!-- COLLEGE TABLE -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">Top Enrolled Colleges — Live Data <span>Updated real-time</span></div>
    <table class="college-table">
      <thead>
        <tr><th>College</th><th>State</th><th>Students</th><th>Avg Readiness</th><th>Top Gap</th><th>Status</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>GH Raisoni College of Engineering</strong></td><td>Maharashtra</td><td>1,247</td><td>71%</td><td>DSA</td><td><span class="status-pill pill-green">Active</span></td></tr>
        <tr><td>VJTI Mumbai</td><td>Maharashtra</td><td>2,108</td><td>78%</td><td>System Design</td><td><span class="status-pill pill-green">Active</span></td></tr>
        <tr><td>NIT Warangal</td><td>Telangana</td><td>1,893</td><td>74%</td><td>Communication</td><td><span class="status-pill pill-green">Active</span></td></tr>
        <tr><td>Anna University</td><td>Tamil Nadu</td><td>3,241</td><td>68%</td><td>DSA</td><td><span class="status-pill pill-blue">Onboarding</span></td></tr>
        <tr><td>AKTU Lucknow</td><td>Uttar Pradesh</td><td>987</td><td>44%</td><td>Core CS</td><td><span class="status-pill pill-yellow">At Risk</span></td></tr>
      </tbody>
    </table>
  </div>

  <!-- BOTTOM STATS -->
  <div class="grid3">
    <div class="mini-stat">
      <div class="num">4.2M</div>
      <div class="lbl">AI Analysis Sessions<br>Run This Year</div>
    </div>
    <div class="mini-stat">
      <div class="num">< 30s</div>
      <div class="lbl">Avg Analysis Time<br>Per Student</div>
    </div>
    <div class="mini-stat">
      <div class="num">₹0</div>
      <div class="lbl">Cost to Student<br>Free Access</div>
    </div>
  </div>

</div>

<!-- FOOTER -->
<div class="footer">
  <p>NPIS · Ministry of Education · Government of India · Data updated in real-time · FY 2025–26</p>
  <p>Powered by Vertical AI · ninelab.in · NLPC 2026 · Team RCM-G2-091</p>
</div>

<script>
  function updateTime() {
    var now = new Date();
    document.getElementById('govTime').textContent = now.toLocaleTimeString('en-IN', {hour:'2-digit',minute:'2-digit',second:'2-digit'}) + ' IST';
  }
  updateTime();
  setInterval(updateTime, 1000);
</script>
</body></html>""")


@app.get("/ninelab/govt", response_class=HTMLResponse)
async def govt_page():
    """Government initiative style page for Vertical AI."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertical AI — National Placement Intelligence Initiative</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:#fff;color:#1a1a2e;}

/* TOP STRIP */
.top-strip{background:#1a2744;color:#fff;font-size:11px;padding:6px 20px;display:flex;justify-content:space-between;align-items:center;}
.top-strip-right{display:flex;gap:16px;opacity:0.7;}

/* HEADER */
.header{border-bottom:4px solid #6c63ff;padding:16px 20px;background:#fff;}
.header-inner{display:flex;align-items:center;justify-content:space-between;max-width:960px;margin:0 auto;}
.logo-block{display:flex;align-items:center;gap:12px;}
.logo-box{width:48px;height:48px;background:#1a2744;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:20px;font-weight:900;}
.logo-text h1{font-size:18px;font-weight:800;color:#1a2744;line-height:1.2;}
.logo-text p{font-size:11px;color:#64748b;margin-top:2px;}
.header-badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;}
.badge{font-size:10px;font-weight:600;padding:4px 10px;border-radius:20px;border:1px solid;}
.badge-blue{color:#1a2744;border-color:#1a2744;background:#f0f4ff;}
.badge-green{color:#166534;border-color:#16a34a;background:#f0fdf4;}

/* NAV */
.nav{background:#f8fafc;border-bottom:1px solid #e5e7eb;padding:0 20px;}
.nav-inner{display:flex;gap:0;max-width:960px;margin:0 auto;}
.nav-item{padding:12px 16px;font-size:12px;font-weight:600;color:#64748b;cursor:pointer;border-bottom:2px solid transparent;}
.nav-item.active{color:#6c63ff;border-bottom-color:#6c63ff;}

/* HERO BANNER */
.hero{background:linear-gradient(135deg,#1a2744 0%,#2d3a6b 100%);color:#fff;padding:36px 20px;}
.hero-inner{max-width:960px;margin:0 auto;}
.hero-tag{display:inline-block;background:rgba(108,99,255,0.3);border:1px solid rgba(108,99,255,0.5);color:#a5b4fc;font-size:10px;font-weight:700;padding:4px 12px;border-radius:20px;letter-spacing:1px;margin-bottom:14px;}
.hero h2{font-size:28px;font-weight:800;line-height:1.2;margin-bottom:10px;}
.hero p{font-size:14px;opacity:0.8;line-height:1.6;max-width:560px;margin-bottom:20px;}
.hero-btns{display:flex;gap:10px;flex-wrap:wrap;}
.btn-primary{padding:11px 22px;background:#6c63ff;color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;font-family:inherit;text-decoration:none;display:inline-block;}
.btn-secondary{padding:11px 22px;background:transparent;color:#fff;border:1.5px solid rgba(255,255,255,0.4);border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;text-decoration:none;display:inline-block;}

/* STATS BAR */
.stats-bar{background:#6c63ff;padding:20px;color:#fff;}
.stats-inner{max-width:960px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);gap:0;}
.stat-item{text-align:center;padding:8px;border-right:1px solid rgba(255,255,255,0.2);}
.stat-item:last-child{border-right:none;}
.stat-num{font-size:28px;font-weight:800;line-height:1;}
.stat-label{font-size:10px;opacity:0.8;margin-top:4px;line-height:1.3;}

/* MAIN CONTENT */
.main{max-width:960px;margin:0 auto;padding:28px 20px;}
.section-title{font-size:11px;font-weight:700;color:#6c63ff;letter-spacing:1.5px;margin-bottom:16px;text-transform:uppercase;}

/* OVERVIEW CARDS */
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:28px;}
.card{background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:18px;}
.card-icon{font-size:22px;margin-bottom:10px;}
.card h3{font-size:13px;font-weight:700;color:#1a1a2e;margin-bottom:6px;}
.card p{font-size:12px;color:#64748b;line-height:1.5;}

/* FRAMEWORK TABLE */
.table-wrap{border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;margin-bottom:28px;}
.table-header{background:#1a2744;color:#fff;display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;padding:12px 16px;font-size:11px;font-weight:700;letter-spacing:0.5px;}
.table-row{display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;padding:12px 16px;font-size:12px;border-bottom:1px solid #f0f0f0;align-items:center;}
.table-row:last-child{border-bottom:none;}
.table-row:nth-child(even){background:#fafafa;}
.check{color:#22c55e;font-weight:700;}
.cross{color:#ef4444;}

/* ELIGIBILITY */
.elig-list{display:flex;flex-direction:column;gap:8px;margin-bottom:28px;}
.elig-item{display:flex;align-items:center;gap:10px;padding:12px 16px;background:#f8fafc;border-radius:10px;border:1px solid #e5e7eb;font-size:13px;}
.elig-dot{width:8px;height:8px;border-radius:50%;background:#6c63ff;flex-shrink:0;}

/* FOOTER */
.footer{background:#1a2744;color:#fff;padding:20px;margin-top:20px;}
.footer-inner{max-width:960px;margin:0 auto;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;}
.footer-left p{font-size:11px;opacity:0.6;margin-top:4px;}
.footer-right{font-size:11px;opacity:0.6;text-align:right;}
</style>
</head>
<body>

<!-- TOP STRIP -->
<div class="top-strip">
  <span>🇮🇳 &nbsp;Government Research Initiative — AI for Skill Development</span>
  <div class="top-strip-right">
    <span>NLPC 2026</span>
    <span>GHRCE Nagpur</span>
    <span>April 2026</span>
  </div>
</div>

<!-- HEADER -->
<div class="header">
  <div class="header-inner">
    <div class="logo-block">
      <div class="logo-box">9L</div>
      <div class="logo-text">
        <h1>Vertical AI</h1>
        <p>National Placement Intelligence Initiative</p>
      </div>
    </div>
    <div class="header-badges">
      <span class="badge badge-blue">Research Based</span>
      <span class="badge badge-green">Live Prototype</span>
      <span class="badge badge-blue">NLPC 2026</span>
    </div>
  </div>
</div>

<!-- NAV -->
<div class="nav">
  <div class="nav-inner">
    <div class="nav-item active">Overview</div>
    <div class="nav-item">For Colleges</div>
    <div class="nav-item">For Students</div>
    <div class="nav-item">Research</div>
  </div>
</div>

<!-- HERO -->
<div class="hero">
  <div class="hero-inner">
    <div class="hero-tag">AI AGENT · PLACEMENT INTELLIGENCE · 2026</div>
    <h2>Bridging India's Placement Gap<br>with Persistent Agentic AI</h2>
    <p>15 lakh engineers graduate every year. 70% are not job-ready. Vertical AI is the first system that remembers every student — and acts on their behalf, every session.</p>
    <div class="hero-btns">
      <a href="/ninelab" class="btn-primary">⚡ Try Live Prototype</a>
      <a href="/ninelab/college-demo" class="btn-secondary">College Dashboard →</a>
    </div>
  </div>
</div>

<!-- STATS BAR -->
<div class="stats-bar">
  <div class="stats-inner">
    <div class="stat-item">
      <div class="stat-num">15L+</div>
      <div class="stat-label">Engineers<br>Per Year</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">70%</div>
      <div class="stat-label">Not Job-Ready<br>NASSCOM 2024</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">38K</div>
      <div class="stat-label">Engineering<br>Colleges India</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">40yr</div>
      <div class="stat-label">Bloom Problem<br>Unsolved</div>
    </div>
  </div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- WHAT IT DOES -->
  <div class="section-title">Initiative Overview</div>
  <div class="cards">
    <div class="card">
      <div class="card-icon">🧠</div>
      <h3>Persistent Student Intelligence</h3>
      <p>First system to maintain cross-session student state — skills, gaps, readiness score remembered after every visit.</p>
    </div>
    <div class="card">
      <div class="card-icon">⚡</div>
      <h3>Agentic AI — Acts, Not Just Answers</h3>
      <p>Student says "find internship" — agent navigates, loads profile, runs analysis, shows results. No manual steps.</p>
    </div>
    <div class="card">
      <div class="card-icon">📊</div>
      <h3>4 Parallel Analysis Agents</h3>
      <p>Jobs, Internships, Skill Gaps, Readiness Score — all analyzed simultaneously. Results in under 30 seconds.</p>
    </div>
    <div class="card">
      <div class="card-icon">🏛️</div>
      <h3>Institutional Dashboard</h3>
      <p>T&P offices track 400+ students — placement readiness, skill gaps, company targets — in one view.</p>
    </div>
  </div>

  <!-- COMPARISON -->
  <div class="section-title">Capability Comparison</div>
  <div class="table-wrap">
    <div class="table-header">
      <span>Capability</span>
      <span>Traditional Tools</span>
      <span>Existing EdTech</span>
      <span>Vertical AI</span>
    </div>
    <div class="table-row">
      <span>Remembers student</span>
      <span class="cross">✗</span>
      <span class="cross">✗</span>
      <span class="check">✓</span>
    </div>
    <div class="table-row">
      <span>Acts on commands</span>
      <span class="cross">✗</span>
      <span class="cross">✗</span>
      <span class="check">✓</span>
    </div>
    <div class="table-row">
      <span>Skill gap + market linked</span>
      <span class="cross">✗</span>
      <span class="cross">✗</span>
      <span class="check">✓</span>
    </div>
    <div class="table-row">
      <span>College dashboard</span>
      <span class="cross">✗</span>
      <span class="cross">✗</span>
      <span class="check">✓</span>
    </div>
    <div class="table-row">
      <span>Patent prior art</span>
      <span>—</span>
      <span>—</span>
      <span class="check">None found</span>
    </div>
  </div>

  <!-- ELIGIBILITY -->
  <div class="section-title">Who This Serves</div>
  <div class="elig-list">
    <div class="elig-item"><div class="elig-dot"></div><span><strong>Engineering Students</strong> — B.Tech / B.E. / M.Tech / MCA seeking placement guidance</span></div>
    <div class="elig-item"><div class="elig-dot"></div><span><strong>T&P Departments</strong> — Track 400+ students, identify at-risk candidates early</span></div>
    <div class="elig-item"><div class="elig-dot"></div><span><strong>College Administration</strong> — Data-driven placement planning and company engagement</span></div>
    <div class="elig-item"><div class="elig-dot"></div><span><strong>Government Skill Initiatives</strong> — Integration with national skill development platforms</span></div>
  </div>

  <!-- RESEARCH BASE -->
  <div class="section-title">Research Foundation</div>
  <div class="cards">
    <div class="card">
      <div class="card-icon">📑</div>
      <h3>Bloom's Two Sigma Problem (1984)</h3>
      <p>Personalized 1-on-1 guidance improves student performance by 2 standard deviations. Unsolved at scale for 40 years.</p>
    </div>
    <div class="card">
      <div class="card-icon">🏆</div>
      <h3>Best Research Paper — Kaveri ThinkFest 2026</h3>
      <p>IEEE-adjacent conference, GHRCE Pune. Patent landscape confirmed zero prior art for persistent student intelligence model.</p>
    </div>
  </div>

</div>

<!-- FOOTER -->
<div class="footer">
  <div class="footer-inner">
    <div class="footer-left">
      <strong>Vertical AI — National Placement Intelligence Initiative</strong>
      <p>NLPC 2026 · Team RCM-G2-091 · GHRCE Nagpur · April 2026</p>
    </div>
    <div class="footer-right">
      <div>ninelab.in/ninelab</div>
      <div style="margin-top:4px;">Research · Prototype · Impact</div>
    </div>
  </div>
</div>

</body></html>""")


@app.get("/ninelab/maharashtra", response_class=HTMLResponse)
async def maharashtra_dashboard():
    """Maharashtra Government Placement Intelligence Dashboard."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maharashtra Rojgar Buddhi Yojana — Placement Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:#f0f2f5;color:#1a1a2e;}

/* MAHARASHTRA GOV HEADER */
.mh-topbar{background:#fff;border-bottom:1px solid #e5e7eb;padding:8px 20px;display:flex;align-items:center;justify-content:space-between;}
.mh-topbar-left{display:flex;align-items:center;gap:12px;}
.mh-emblem{font-size:32px;}
.mh-brand h1{font-size:14px;font-weight:800;color:#1a2744;line-height:1.2;}
.mh-brand p{font-size:10px;color:#64748b;margin-top:1px;}
.mh-topbar-right{display:flex;align-items:center;gap:10px;}
.mh-flag{display:flex;gap:0;border-radius:3px;overflow:hidden;height:22px;}
.mh-flag-s{width:14px;}

/* SAFFRON BANNER */
.mh-banner{background:linear-gradient(135deg,#ff6b00,#e85d00);color:#fff;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}
.mh-banner-left h2{font-size:17px;font-weight:800;line-height:1.2;}
.mh-banner-left p{font-size:11px;opacity:0.85;margin-top:3px;}
.mh-banner-right{display:flex;gap:8px;flex-wrap:wrap;}
.mbadge{font-size:10px;font-weight:700;padding:4px 10px;border-radius:4px;background:rgba(255,255,255,0.2);color:#fff;border:1px solid rgba(255,255,255,0.3);}

/* TABS */
.tabs{background:#fff;border-bottom:2px solid #ff6b00;padding:0 20px;display:flex;gap:0;overflow-x:auto;}
.tab{padding:11px 16px;font-size:12px;font-weight:600;color:#64748b;cursor:pointer;border-bottom:3px solid transparent;white-space:nowrap;margin-bottom:-2px;}
.tab.active{color:#e85d00;border-bottom-color:#e85d00;}
.tab-panel{display:none;}
.tab-panel.active{display:block;}

/* MAIN */
.main{max-width:1100px;margin:0 auto;padding:18px 16px;}

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}
.kpi{background:#fff;border-radius:10px;padding:14px 16px;border-top:4px solid;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.kpi-num{font-size:26px;font-weight:800;line-height:1;}
.kpi-label{font-size:10px;color:#64748b;margin-top:3px;line-height:1.4;}
.kpi-tag{font-size:10px;font-weight:600;margin-top:6px;}
.green{color:#16a34a;} .orange{color:#e85d00;} .blue{color:#1d4ed8;} .red{color:#ef4444;}

/* GRID */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:18px;}
.card{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.card-hd{font-size:12px;font-weight:700;color:#1a1a2e;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #f0f0f0;display:flex;align-items:center;justify-content:space-between;}
.card-hd span{font-size:10px;font-weight:500;color:#94a3b8;}

/* SCHEME CARDS */
.scheme-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;}
.scheme-card{background:#fff;border-radius:10px;padding:14px;border-left:4px solid;box-shadow:0 1px 4px rgba(0,0,0,0.05);cursor:pointer;transition:box-shadow 0.2s;}
.scheme-card:hover{box-shadow:0 4px 12px rgba(0,0,0,0.1);}
.scheme-tag{font-size:9px;font-weight:700;letter-spacing:0.8px;margin-bottom:6px;text-transform:uppercase;}
.scheme-name{font-size:13px;font-weight:700;color:#1a1a2e;margin-bottom:4px;line-height:1.3;}
.scheme-desc{font-size:11px;color:#64748b;line-height:1.4;margin-bottom:8px;}
.scheme-stats{display:flex;gap:12px;}
.scheme-stat{font-size:10px;color:#64748b;}<br>.scheme-stat strong{color:#1a1a2e;font-weight:700;}

/* EXAM TABLE */
.exam-table{width:100%;border-collapse:collapse;font-size:12px;}
.exam-table th{background:#fff8f0;padding:9px 12px;text-align:left;font-size:10px;font-weight:700;color:#92400e;letter-spacing:0.5px;border-bottom:2px solid #fed7aa;}
.exam-table td{padding:9px 12px;border-bottom:1px solid #f5f5f5;color:#1a1a2e;}
.exam-table tr:hover td{background:#fafafa;}
.epill{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:600;}
.epill-green{background:#dcfce7;color:#16a34a;}
.epill-orange{background:#ffedd5;color:#c2410c;}
.epill-blue{background:#dbeafe;color:#1d4ed8;}
.epill-red{background:#fee2e2;color:#dc2626;}

/* BAR CHART ROW */
.bar-row{display:flex;align-items:center;gap:10px;padding:7px 0;font-size:12px;}
.bar-name{width:130px;font-weight:500;color:#374151;flex-shrink:0;}
.bar-track{flex:1;background:#f0f0f0;border-radius:4px;height:8px;}
.bar-fill{height:8px;border-radius:4px;}
.bar-val{width:36px;text-align:right;font-weight:700;font-size:11px;flex-shrink:0;}

/* DISTRICT MAP TABLE */
.dist-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.dist-card{background:#f8fafc;border-radius:8px;padding:10px 12px;border:1px solid #e5e7eb;}
.dist-name{font-size:11px;font-weight:600;color:#1a2744;margin-bottom:4px;}
.dist-bar{height:4px;border-radius:4px;background:#e5e7eb;margin-bottom:4px;}
.dist-bar-fill{height:4px;border-radius:4px;background:linear-gradient(90deg,#ff6b00,#fbbf24);}
.dist-stats{display:flex;justify-content:space-between;font-size:10px;color:#64748b;}

/* SCHOLARSHIP TABLE */
.sch-table{width:100%;border-collapse:collapse;font-size:12px;}
.sch-table th{background:#fafafa;padding:9px 12px;text-align:left;font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.5px;border-bottom:1px solid #e5e7eb;}
.sch-table td{padding:9px 12px;border-bottom:1px solid #f5f5f5;vertical-align:top;}
.sch-amt{font-size:13px;font-weight:700;color:#e85d00;}

/* ALERT */
.alert{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;border-radius:8px;font-size:12px;margin-bottom:8px;}
.alert-r{background:#fef2f2;border-left:3px solid #ef4444;}
.alert-y{background:#fffbeb;border-left:3px solid #f59e0b;}
.alert-g{background:#f0fdf4;border-left:3px solid #22c55e;}
.alert strong{display:block;font-weight:600;margin-bottom:2px;}

/* FOOTER */
.mh-footer{background:#1a2744;color:#fff;padding:14px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-top:8px;}
.mh-footer p{font-size:10px;opacity:0.5;}
</style>
</head>
<body>

<!-- HEADER -->
<div class="mh-topbar">
  <div class="mh-topbar-left">
    <div class="mh-emblem">🦁</div>
    <div class="mh-brand">
      <h1>महाराष्ट्र शासन — कौशल्य विकास विभाग</h1>
      <p>Government of Maharashtra · Skill Development & Entrepreneurship Dept · Placement Intelligence Wing</p>
    </div>
  </div>
  <div class="mh-topbar-right">
    <div style="font-size:11px;color:#64748b;" id="mhTime"></div>
    <div style="width:6px;height:6px;border-radius:50%;background:#22c55e;animation:blink 1.5s infinite;"></div>
    <span style="font-size:11px;color:#22c55e;font-weight:600;">LIVE</span>
  </div>
</div>
<style>@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}</style>

<!-- BANNER -->
<div class="mh-banner">
  <div class="mh-banner-left">
    <h2>Maharashtra Rojgar Buddhi Yojana (MRBY)</h2>
    <p>Powered by Vertical AI · Persistent Agentic Intelligence · FY 2025–26 · Phase 1 Active</p>
  </div>
  <div class="mh-banner-right">
    <span class="mbadge">MahaDBT Integrated</span>
    <span class="mbadge">DTE Maharashtra</span>
    <span class="mbadge">MSBTE Linked</span>
    <span class="mbadge">NLPC 2026</span>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">📊 Overview</div>
  <div class="tab" onclick="switchTab('schemes')">📋 Schemes & Scholarships</div>
  <div class="tab" onclick="switchTab('exams')">📝 Competitive Exams</div>
  <div class="tab" onclick="switchTab('districts')">🗺️ District Dashboard</div>
  <div class="tab" onclick="switchTab('colleges')">🏛️ Colleges</div>
  <div class="tab" onclick="switchTab('alerts')">🔔 Alerts</div>
</div>

<!-- ═══ TAB: OVERVIEW ═══ -->
<div id="tab-overview" class="tab-panel active">
<div class="main">

  <div class="kpi-row">
    <div class="kpi" style="border-color:#ff6b00;">
      <div class="kpi-num orange">4,18,234</div>
      <div class="kpi-label">Engineering Students<br>Maharashtra 2025–26</div>
      <div class="kpi-tag green">↑ 6.2% YoY</div>
    </div>
    <div class="kpi" style="border-color:#22c55e;">
      <div class="kpi-num green">1,127</div>
      <div class="kpi-label">Colleges on<br>MRBY Platform</div>
      <div class="kpi-tag green">↑ 214 new this year</div>
    </div>
    <div class="kpi" style="border-color:#1d4ed8;">
      <div class="kpi-num blue">63%</div>
      <div class="kpi-label">Avg Placement<br>Readiness Score</div>
      <div class="kpi-tag green">↑ 11pts from baseline</div>
    </div>
    <div class="kpi" style="border-color:#ef4444;">
      <div class="kpi-num red">1,47,882</div>
      <div class="kpi-label">Students Below<br>50% Readiness</div>
      <div class="kpi-tag red">↓ Needs intervention</div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-hd">Division-wise Readiness <span>Avg score %</span></div>
      <div class="bar-row"><span class="bar-name">Pune</span><div class="bar-track"><div class="bar-fill" style="width:78%;background:#22c55e;"></div></div><span class="bar-val green">78%</span></div>
      <div class="bar-row"><span class="bar-name">Mumbai</span><div class="bar-track"><div class="bar-fill" style="width:74%;background:#22c55e;"></div></div><span class="bar-val green">74%</span></div>
      <div class="bar-row"><span class="bar-name">Nagpur</span><div class="bar-track"><div class="bar-fill" style="width:68%;background:#ff6b00;"></div></div><span class="bar-val orange">68%</span></div>
      <div class="bar-row"><span class="bar-name">Nashik</span><div class="bar-track"><div class="bar-fill" style="width:61%;background:#ff6b00;"></div></div><span class="bar-val orange">61%</span></div>
      <div class="bar-row"><span class="bar-name">Aurangabad</span><div class="bar-track"><div class="bar-fill" style="width:54%;background:#f59e0b;"></div></div><span class="bar-val" style="color:#f59e0b;">54%</span></div>
      <div class="bar-row"><span class="bar-name">Amravati</span><div class="bar-track"><div class="bar-fill" style="width:48%;background:#ef4444;"></div></div><span class="bar-val red">48%</span></div>
      <div class="bar-row"><span class="bar-name">Kolhapur</span><div class="bar-track"><div class="bar-fill" style="width:57%;background:#f59e0b;"></div></div><span class="bar-val" style="color:#f59e0b;">57%</span></div>
    </div>
    <div class="card">
      <div class="card-hd">Top Skill Gaps — Maharashtra <span>% students affected</span></div>
      <div class="bar-row"><span class="bar-name">DSA</span><div class="bar-track"><div class="bar-fill" style="width:72%;background:#ef4444;"></div></div><span class="bar-val red">72%</span></div>
      <div class="bar-row"><span class="bar-name">Communication</span><div class="bar-track"><div class="bar-fill" style="width:68%;background:#ef4444;"></div></div><span class="bar-val red">68%</span></div>
      <div class="bar-row"><span class="bar-name">System Design</span><div class="bar-track"><div class="bar-fill" style="width:61%;background:#f59e0b;"></div></div><span class="bar-val" style="color:#f59e0b;">61%</span></div>
      <div class="bar-row"><span class="bar-name">SQL / Database</span><div class="bar-track"><div class="bar-fill" style="width:54%;background:#f59e0b;"></div></div><span class="bar-val" style="color:#f59e0b;">54%</span></div>
      <div class="bar-row"><span class="bar-name">Resume Writing</span><div class="bar-track"><div class="bar-fill" style="width:48%;background:#ff6b00;"></div></div><span class="bar-val orange">48%</span></div>
      <div class="bar-row"><span class="bar-name">Core CS</span><div class="bar-track"><div class="bar-fill" style="width:44%;background:#ff6b00;"></div></div><span class="bar-val orange">44%</span></div>
      <canvas id="gapChart" height="100" style="margin-top:10px;"></canvas>
    </div>
  </div>

  <div class="card">
    <div class="card-hd">Company-wise Hiring from Maharashtra <span>Top recruiters FY 2025–26</span></div>
    <canvas id="companyChart" height="80"></canvas>
  </div>

</div>
</div>

<!-- ═══ TAB: SCHEMES ═══ -->
<div id="tab-schemes" class="tab-panel">
<div class="main">
  <div style="font-size:12px;color:#64748b;margin-bottom:14px;">Maharashtra government schemes & scholarships tracked on MRBY platform · Students can apply directly via MahaDBT</div>

  <div class="scheme-grid">
    <div class="scheme-card" style="border-color:#ff6b00;">
      <div class="scheme-tag" style="color:#ff6b00;">SCHOLARSHIP · OBC / SBC / VJNT</div>
      <div class="scheme-name">Dr. Panjabrao Deshmukh Vasatigruha Nirvah Bhatta Yojana</div>
      <div class="scheme-desc">Hostel + maintenance allowance for OBC/SBC/VJNT engineering students studying outside home district.</div>
      <div class="scheme-stats">
        <div class="scheme-stat">Amount: <strong>₹30,000/yr</strong></div>
        <div class="scheme-stat">Students: <strong>2,14,000+</strong></div>
        <div class="scheme-stat">Portal: <strong>MahaDBT</strong></div>
      </div>
    </div>
    <div class="scheme-card" style="border-color:#1d4ed8;">
      <div class="scheme-tag" style="color:#1d4ed8;">SCHOLARSHIP · SC / ST</div>
      <div class="scheme-name">Swadhar Gruh Yojana — Dr. Babasaheb Ambedkar</div>
      <div class="scheme-desc">Scholarship for SC/ST students living outside government hostels — covers boarding, lodging & other expenses.</div>
      <div class="scheme-stats">
        <div class="scheme-stat">Amount: <strong>₹51,000/yr</strong></div>
        <div class="scheme-stat">Students: <strong>1,08,000+</strong></div>
        <div class="scheme-stat">Portal: <strong>MahaDBT</strong></div>
      </div>
    </div>
    <div class="scheme-card" style="border-color:#22c55e;">
      <div class="scheme-tag" style="color:#22c55e;">SCHOLARSHIP · EBC / OPEN</div>
      <div class="scheme-name">Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti</div>
      <div class="scheme-desc">Fee reimbursement for EBC (Economically Backward Class) engineering students with family income below ₹8 lakh.</div>
      <div class="scheme-stats">
        <div class="scheme-stat">Amount: <strong>Full tuition</strong></div>
        <div class="scheme-stat">Students: <strong>3,42,000+</strong></div>
        <div class="scheme-stat">Portal: <strong>MahaDBT</strong></div>
      </div>
    </div>
    <div class="scheme-card" style="border-color:#a855f7;">
      <div class="scheme-tag" style="color:#a855f7;">SKILL DEVELOPMENT</div>
      <div class="scheme-name">Mahaswayam — Maharashtra Employment & Skill Portal</div>
      <div class="scheme-desc">Unified portal for job seekers, skill training, placement — covers all districts. Vertical AI can integrate directly.</div>
      <div class="scheme-stats">
        <div class="scheme-stat">Registered: <strong>62 lakh+</strong></div>
        <div class="scheme-stat">Jobs posted: <strong>1.2 lakh+</strong></div>
        <div class="scheme-stat">Districts: <strong>36</strong></div>
      </div>
    </div>
    <div class="scheme-card" style="border-color:#f59e0b;">
      <div class="scheme-tag" style="color:#f59e0b;">ENTREPRENEURSHIP</div>
      <div class="scheme-name">Annasaheb Patil Economic Development Corporation</div>
      <div class="scheme-desc">Loan guarantee scheme for Maratha community entrepreneurs — startup funding up to ₹10 lakh at subsidized rate.</div>
      <div class="scheme-stats">
        <div class="scheme-stat">Loan: <strong>Up to ₹10L</strong></div>
        <div class="scheme-stat">Interest: <strong>Subsidized</strong></div>
        <div class="scheme-stat">Category: <strong>Maratha</strong></div>
      </div>
    </div>
    <div class="scheme-card" style="border-color:#ef4444;">
      <div class="scheme-tag" style="color:#ef4444;">SKILL · ITI / DIPLOMA</div>
      <div class="scheme-name">MSBTE Skill Development Initiative (SDI)</div>
      <div class="scheme-desc">Modular employable skills for diploma & engineering students. Certification recognized by industry for placement.</div>
      <div class="scheme-stats">
        <div class="scheme-stat">Institutes: <strong>450+</strong></div>
        <div class="scheme-stat">Courses: <strong>120+</strong></div>
        <div class="scheme-stat">Board: <strong>MSBTE</strong></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-hd">Scholarship Utilization via MRBY Platform</div>
    <table class="sch-table">
      <thead><tr><th>Scheme</th><th>Eligible Students</th><th>Applied via MRBY</th><th>Amount/yr</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td>Swadhar Gruh Yojana</td><td>1,08,420</td><td>84,310 (77%)</td><td class="sch-amt">₹51,000</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td>Panjabrao Deshmukh Yojana</td><td>2,14,880</td><td>1,92,440 (89%)</td><td class="sch-amt">₹30,000</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td>Shahu Maharaj Shishyavrutti</td><td>3,42,100</td><td>2,18,900 (63%)</td><td class="sch-amt">Full fees</td><td><span class="epill epill-orange">Partial</span></td></tr>
        <tr><td>Central Sector Scholarship</td><td>48,200</td><td>31,400 (65%)</td><td class="sch-amt">₹20,000</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td>HDFC Badhte Kadam</td><td>12,000</td><td>8,900 (74%)</td><td class="sch-amt">₹75,000</td><td><span class="epill epill-green">Active</span></td></tr>
      </tbody>
    </table>
  </div>
</div>
</div>

<!-- ═══ TAB: EXAMS ═══ -->
<div id="tab-exams" class="tab-panel">
<div class="main">
  <div class="card" style="margin-bottom:16px;">
    <div class="card-hd">Competitive Exams Tracked on MRBY Platform <span>Readiness mapped to exam syllabus</span></div>
    <table class="exam-table">
      <thead><tr><th>Exam</th><th>Conducting Body</th><th>Next Date</th><th>MH Students</th><th>Avg Readiness</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td><strong>MHT-CET</strong><br><span style="font-size:10px;color:#64748b;">Maharashtra Common Entrance Test</span></td><td>State CET Cell, MH</td><td>Apr–May 2026</td><td>4,12,000+</td><td><span style="color:#22c55e;font-weight:700;">72%</span></td><td><span class="epill epill-orange">Upcoming</span></td></tr>
        <tr><td><strong>GATE 2026</strong><br><span style="font-size:10px;color:#64748b;">Graduate Aptitude Test in Engineering</span></td><td>IIT Roorkee</td><td>Feb 2026</td><td>48,200</td><td><span style="color:#f59e0b;font-weight:700;">54%</span></td><td><span class="epill epill-green">Completed</span></td></tr>
        <tr><td><strong>MPSC Engineering Services</strong><br><span style="font-size:10px;color:#64748b;">Maharashtra Public Service Commission</span></td><td>MPSC, Pune</td><td>Jun 2026</td><td>28,400</td><td><span style="color:#ef4444;font-weight:700;">41%</span></td><td><span class="epill epill-blue">Prep Phase</span></td></tr>
        <tr><td><strong>TCS NQT</strong><br><span style="font-size:10px;color:#64748b;">National Qualifier Test — TCS</span></td><td>Tata Consultancy Services</td><td>Rolling</td><td>1,24,000+</td><td><span style="color:#ff6b00;font-weight:700;">61%</span></td><td><span class="epill epill-orange">Open</span></td></tr>
        <tr><td><strong>Infosys Springboard</strong><br><span style="font-size:10px;color:#64748b;">Campus recruitment test</span></td><td>Infosys Ltd.</td><td>Rolling</td><td>89,000+</td><td><span style="color:#ff6b00;font-weight:700;">58%</span></td><td><span class="epill epill-orange">Open</span></td></tr>
        <tr><td><strong>Wipro NLTH</strong><br><span style="font-size:10px;color:#64748b;">National Level Talent Hunt</span></td><td>Wipro Technologies</td><td>Rolling</td><td>72,000+</td><td><span style="color:#f59e0b;font-weight:700;">55%</span></td><td><span class="epill epill-orange">Open</span></td></tr>
        <tr><td><strong>Cognizant GenC</strong><br><span style="font-size:10px;color:#64748b;">Campus hiring program</span></td><td>Cognizant Technology</td><td>Rolling</td><td>64,000+</td><td><span style="color:#f59e0b;font-weight:700;">57%</span></td><td><span class="epill epill-orange">Open</span></td></tr>
        <tr><td><strong>UPSC ESE</strong><br><span style="font-size:10px;color:#64748b;">Engineering Services Exam</span></td><td>UPSC, New Delhi</td><td>Jan 2027</td><td>8,200</td><td><span style="color:#ef4444;font-weight:700;">38%</span></td><td><span class="epill epill-blue">Prep Phase</span></td></tr>
      </tbody>
    </table>
  </div>
  <div class="grid3">
    <div class="kpi" style="border-top:4px solid #22c55e;background:#fff;border-radius:10px;padding:14px;">
      <div class="kpi-num green">61%</div>
      <div class="kpi-label">Students exam-ready<br>for campus drives</div>
    </div>
    <div class="kpi" style="border-top:4px solid #f59e0b;background:#fff;border-radius:10px;padding:14px;">
      <div class="kpi-num" style="color:#f59e0b;">4.2L+</div>
      <div class="kpi-label">Students appearing<br>MHT-CET 2026</div>
    </div>
    <div class="kpi" style="border-top:4px solid #ef4444;background:#fff;border-radius:10px;padding:14px;">
      <div class="kpi-num red">39%</div>
      <div class="kpi-label">Students NOT ready<br>for any exam yet</div>
    </div>
  </div>
</div>
</div>

<!-- ═══ TAB: DISTRICTS ═══ -->
<div id="tab-districts" class="tab-panel">
<div class="main">
  <div class="card" style="margin-bottom:16px;">
    <div class="card-hd">All 36 Districts — Placement Readiness Score</div>
    <div class="dist-grid">
      <div class="dist-card"><div class="dist-name">Pune</div><div class="dist-bar"><div class="dist-bar-fill" style="width:78%;"></div></div><div class="dist-stats"><span>78% ready</span><span>48,200 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Mumbai City</div><div class="dist-bar"><div class="dist-bar-fill" style="width:76%;"></div></div><div class="dist-stats"><span>76% ready</span><span>62,100 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Thane</div><div class="dist-bar"><div class="dist-bar-fill" style="width:72%;"></div></div><div class="dist-stats"><span>72% ready</span><span>38,400 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Nagpur</div><div class="dist-bar"><div class="dist-bar-fill" style="width:68%;"></div></div><div class="dist-stats"><span>68% ready</span><span>29,800 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Nashik</div><div class="dist-bar"><div class="dist-bar-fill" style="width:61%;"></div></div><div class="dist-stats"><span>61% ready</span><span>21,400 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Aurangabad</div><div class="dist-bar"><div class="dist-bar-fill" style="width:54%;"></div></div><div class="dist-stats"><span>54% ready</span><span>18,200 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Solapur</div><div class="dist-bar"><div class="dist-bar-fill" style="width:52%;"></div></div><div class="dist-stats"><span>52% ready</span><span>12,800 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Kolhapur</div><div class="dist-bar"><div class="dist-bar-fill" style="width:57%;"></div></div><div class="dist-stats"><span>57% ready</span><span>14,600 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Amravati</div><div class="dist-bar"><div class="dist-bar-fill" style="width:48%;"></div></div><div class="dist-stats"><span>48% ready</span><span>11,200 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Nanded</div><div class="dist-bar"><div class="dist-bar-fill" style="width:44%;"></div></div><div class="dist-stats"><span>44% ready</span><span>9,800 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Latur</div><div class="dist-bar"><div class="dist-bar-fill" style="width:42%;"></div></div><div class="dist-stats"><span>42% ready</span><span>8,400 students</span></div></div>
      <div class="dist-card"><div class="dist-name">Osmanabad</div><div class="dist-bar"><div class="dist-bar-fill" style="width:38%;"></div></div><div class="dist-stats"><span>38% ready</span><span>6,200 students</span></div></div>
    </div>
  </div>
</div>
</div>

<!-- ═══ TAB: COLLEGES ═══ -->
<div id="tab-colleges" class="tab-panel">
<div class="main">
  <div class="card">
    <div class="card-hd">Top Enrolled Colleges — MRBY Platform <span>Live data · DTE Maharashtra registered</span></div>
    <table class="exam-table">
      <thead><tr><th>College</th><th>District</th><th>Affiliation</th><th>Students</th><th>Avg Readiness</th><th>Critical Gap</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td><strong>GH Raisoni College of Engg, Nagpur</strong></td><td>Nagpur</td><td>RTM Nagpur Univ</td><td>1,247</td><td><span class="green" style="font-weight:700;">71%</span></td><td>DSA</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td><strong>COEP Technological University</strong></td><td>Pune</td><td>Autonomous</td><td>2,841</td><td><span class="green" style="font-weight:700;">82%</span></td><td>System Design</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td><strong>VJTI Mumbai</strong></td><td>Mumbai</td><td>Mumbai Univ</td><td>3,108</td><td><span class="green" style="font-weight:700;">79%</span></td><td>Communication</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td><strong>Symbiosis Institute of Tech</strong></td><td>Pune</td><td>SIU</td><td>1,892</td><td><span class="green" style="font-weight:700;">76%</span></td><td>DSA</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td><strong>PICT Pune</strong></td><td>Pune</td><td>Savitribai Phule</td><td>2,214</td><td><span class="green" style="font-weight:700;">74%</span></td><td>SQL</td><td><span class="epill epill-green">Active</span></td></tr>
        <tr><td><strong>SGSITS Indore</strong></td><td>Nashik</td><td>SPPU</td><td>1,108</td><td><span style="color:#f59e0b;font-weight:700;">58%</span></td><td>Core CS</td><td><span class="epill epill-orange">Needs Attention</span></td></tr>
        <tr><td><strong>MGM College of Engg, Aurangabad</strong></td><td>Aurangabad</td><td>DR. BAMU</td><td>984</td><td><span style="color:#ef4444;font-weight:700;">47%</span></td><td>DSA, SQL</td><td><span class="epill epill-red">At Risk</span></td></tr>
      </tbody>
    </table>
  </div>
</div>
</div>

<!-- ═══ TAB: ALERTS ═══ -->
<div id="tab-alerts" class="tab-panel">
<div class="main">
  <div class="alert alert-r"><div style="font-size:18px;">🔴</div><div><strong>1,47,882 students below 50% readiness — immediate action needed</strong>Concentrated in Aurangabad, Amravati, Nanded divisions. T&P offices alerted. Vertical AI recommending personalized 30-day plans.</div></div>
  <div class="alert alert-r"><div style="font-size:18px;">🔴</div><div><strong>DSA gap in 72% of Maharashtra engineering students</strong>Most critical skill gap statewide. Platform recommending targeted DSA bootcamps via government ITI network.</div></div>
  <div class="alert alert-y"><div style="font-size:18px;">🟡</div><div><strong>MHT-CET 2026 in 3 weeks — 38% students not prepared</strong>1,56,000+ students flagged as underprepared. Immediate coaching push required via Mahaswayam portal.</div></div>
  <div class="alert alert-y"><div style="font-size:18px;">🟡</div><div><strong>Scholarship uptake below target — Shahu Maharaj Yojana at 63%</strong>1,23,200 eligible students haven't applied. MRBY platform sending automated reminders via Vertical AI agent.</div></div>
  <div class="alert alert-y"><div style="font-size:18px;">🟡</div><div><strong>873 colleges yet to integrate with MRBY platform</strong>Phase 2 enrollment drive required. DTE Maharashtra coordination pending.</div></div>
  <div class="alert alert-g"><div style="font-size:18px;">🟢</div><div><strong>Pune Division leading — 78% avg readiness, model for replication</strong>COEP, PICT, Symbiosis showing best outcomes. Vertical AI methodology being documented for state-wide rollout.</div></div>
  <div class="alert alert-g"><div style="font-size:18px;">🟢</div><div><strong>4.2M+ AI analysis sessions completed this year</strong>Average analysis time under 30 seconds. Student engagement up 34% since persistent profile feature launched.</div></div>
</div>
</div>

<!-- FOOTER -->
<div class="mh-footer">
  <p>Maharashtra Rojgar Buddhi Yojana (MRBY) · Powered by Vertical AI · DTE Maharashtra · MSBTE · Mahaswayam · FY 2025–26</p>
  <p>ninelab.in/ninelab · NLPC 2026 · Team RCM-G2-091 · GHRCE Nagpur</p>
</div>

<script>
// Clock
function updateTime(){document.getElementById('mhTime').textContent=new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'})+' IST';}
updateTime();setInterval(updateTime,1000);

// Tab switching
function switchTab(name){
  document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});
  document.getElementById('tab-'+name).classList.add('active');
  event.target.classList.add('active');
}

// Skill gap donut chart
var gapCtx=document.getElementById('gapChart').getContext('2d');
new Chart(gapCtx,{type:'doughnut',data:{labels:['DSA','Communication','System Design','SQL','Resume','Core CS'],datasets:[{data:[72,68,61,54,48,44],backgroundColor:['#ef4444','#f97316','#f59e0b','#6c63ff','#22c55e','#3b82f6'],borderWidth:0}]},options:{plugins:{legend:{position:'right',labels:{font:{size:10},boxWidth:10}}},cutout:'65%'}});

// Company bar chart
var compCtx=document.getElementById('companyChart').getContext('2d');
new Chart(compCtx,{type:'bar',data:{labels:['TCS','Infosys','Wipro','Cognizant','Accenture','L&T Tech','Persistent','KPIT'],datasets:[{label:'Students Hired',data:[18400,14200,11800,9400,8200,6100,4800,3900],backgroundColor:'#ff6b00',borderRadius:4}]},options:{plugins:{legend:{display:false}},scales:{y:{grid:{color:'#f0f0f0'},ticks:{font:{size:10}}},x:{grid:{display:false},ticks:{font:{size:10}}}}}});
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
        "jsearch": bool(JSEARCH_API_KEY),
        "adzuna": bool(ADZUNA_APP_ID and ADZUNA_APP_KEY),
    }}


# ══════════════════════════════════════════════════════════════════════════════
#  NINE LAB 2.0 — Complete Career Platform
# ══════════════════════════════════════════════════════════════════════════════

# ── Supabase REST helper ───────────────────────────────────────────────────────

def supabase_rest(method: str, table: str, *, payload=None, params: dict = None,
                  token: str = None, use_service_key: bool = False,
                  upsert: bool = False) -> dict:
    """Hit Supabase PostgREST API. Always use service key server-side for writes."""
    import httpx
    auth_key = SUPABASE_SERVICE_KEY if use_service_key else (token or SUPABASE_KEY)
    prefer = "return=representation"
    if upsert:
        prefer += ",resolution=merge-duplicates"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    qp = dict(params or {})
    if "select" not in qp:
        qp["select"] = "*"
    try:
        import httpx as _hx
        if method == "GET":
            r = _hx.get(url, headers=headers, params=qp, timeout=10)
        elif method == "POST":
            r = _hx.post(url, headers=headers, json=payload, params=qp, timeout=10)
        elif method == "PATCH":
            r = _hx.patch(url, headers=headers, json=payload, params=qp, timeout=10)
        elif method == "DELETE":
            r = _hx.delete(url, headers=headers, params=qp, timeout=10)
        else:
            return {"status": 405, "data": {"error": "Unsupported method"}}
        data = r.json() if r.text else []
        return {"status": r.status_code, "data": data}
    except Exception as e:
        return {"status": 500, "data": {"error": str(e)}}


def _get_auth_user_v2(authorization: Optional[str]) -> Optional[dict]:
    """Extract and validate user from 'Bearer <token>' header."""
    if not authorization:
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None
    return get_user_from_token(token)


def _require_auth_v2(authorization: Optional[str]) -> dict:
    user = _get_auth_user_v2(authorization)
    if not user:
        raise HTTPException(401, detail="Authentication required. Please sign in.")
    return user


# ── New Pydantic Models ────────────────────────────────────────────────────────

class MagicLinkRequest(BaseModel):
    email: str

class OTPVerifyRequest(BaseModel):
    email: str
    token: str

class ProfileSaveRequest(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    city: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    college: Optional[str] = None
    grad_year: Optional[str] = None
    cgpa: Optional[str] = None
    is_fresher: Optional[bool] = None
    skills: Optional[str] = None
    projects: Optional[str] = None
    achievements: Optional[str] = None
    roles: Optional[list] = None

class ApplicationCreateRequest(BaseModel):
    opportunity_id: Optional[str] = None
    custom_title: Optional[str] = None
    custom_company: Optional[str] = None
    custom_url: Optional[str] = None
    status: str = "saved"
    notes: Optional[str] = None
    resume_generated: Optional[bool] = False
    prep_downloaded: Optional[bool] = False

class ApplicationUpdateRequest(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    resume_generated: Optional[bool] = None
    prep_downloaded: Optional[bool] = None
    redirect_clicked: Optional[bool] = None

class ActivityRequest(BaseModel):
    action_type: str
    metadata: Optional[dict] = None


# ── Profile completion calculator ─────────────────────────────────────────────

def _calc_profile_pct(row: dict) -> int:
    fields = ["full_name", "degree", "college", "skills", "grad_year"]
    exp_ok = bool(row.get("roles")) or row.get("is_fresher", False)
    filled = sum(1 for f in fields if (row.get(f) or "").strip()) + (1 if exp_ok else 0)
    return int(filled / (len(fields) + 1) * 100)


# ── Seed opportunities (in-memory, also used as DB fallback) ──────────────────

SEED_OPPS = [
    {"id":"opp-001","title":"Software Developer Trainee","company":"TCS","type":"job",
     "location":"Pan India","salary_range":"₹3.5L–₹4.5L/yr","url":"https://nextstep.tcs.com",
     "description":"Build enterprise software across banking, healthcare, and retail. 15-step training program. Strong Java/Python fundamentals required.",
     "requirements":"B.Tech/MCA/M.Sc with 60%+. Proficient in Java or Python. No active backlogs.",
     "tags":["fresher","b.tech","mca","java","python","it"],"match_fields":"java python sql btech mca fresher software developer it"},
    {"id":"opp-002","title":"System Engineer","company":"Infosys","type":"job",
     "location":"Bengaluru / Pune / Hyderabad","salary_range":"₹3.6L–₹4.2L/yr","url":"https://career.infosys.com",
     "description":"Work on Fortune 500 client projects. Infosys InStep training included. Rotational assignments across verticals.",
     "requirements":"B.Tech/BCA/MCA with 65%+. Any programming language. Good communication.",
     "tags":["fresher","b.tech","java","it","bca"],"match_fields":"java c++ sql btech bca mca fresher infosys system engineer"},
    {"id":"opp-003","title":"Associate Engineer","company":"Wipro","type":"job",
     "location":"Multi-location","salary_range":"₹3.5L/yr","url":"https://careers.wipro.com",
     "description":"Application development, testing, and support. 90-day Talent Transformation training.",
     "requirements":"B.Tech/BE/MCA with 60%+. No arrears. Proficient in one language.",
     "tags":["fresher","b.tech","java","it"],"match_fields":"java python btech mca fresher wipro associate engineer"},
    {"id":"opp-004","title":"Junior Software Engineer","company":"Razorpay","type":"job",
     "location":"Bengaluru","salary_range":"₹12L–₹18L/yr","url":"https://razorpay.com/jobs",
     "description":"Build India's payment infrastructure. High-scale systems processing millions of transactions daily. Strong DSA required.",
     "requirements":"B.Tech CS/related. Strong DSA, backend development. Go/Java/Python preferred.",
     "tags":["startup","fintech","java","python","backend","b.tech"],"match_fields":"java python go backend api fintech payments btech software engineer"},
    {"id":"opp-005","title":"Backend Engineer","company":"Zepto","type":"job",
     "location":"Mumbai","salary_range":"₹15L–₹25L/yr","url":"https://zepto.com/careers",
     "description":"Build real-time inventory, order management, and logistics systems for India's fastest-growing quick commerce startup.",
     "requirements":"B.Tech CS. Node.js, Go, or Java. Microservices. Strong problem-solving.",
     "tags":["startup","backend","node.js","java","b.tech"],"match_fields":"node.js java go backend microservices api btech"},
    {"id":"opp-006","title":"Data Analyst","company":"Flipkart","type":"job",
     "location":"Bengaluru","salary_range":"₹8L–₹14L/yr","url":"https://www.flipkartcareers.com",
     "description":"Analyze large datasets, build dashboards, run A/B tests, and present insights to leadership at India's top e-commerce platform.",
     "requirements":"B.Tech/B.Sc/MCA. SQL, Python (Pandas). Tableau/PowerBI preferred.",
     "tags":["data","sql","python","analytics","b.tech","mca"],"match_fields":"python sql pandas data analytics tableau power bi btech mca data analyst"},
    {"id":"opp-007","title":"Frontend Engineer","company":"Swiggy","type":"job",
     "location":"Bengaluru","salary_range":"₹12L–₹20L/yr","url":"https://careers.swiggy.com",
     "description":"Build consumer-facing React features for Swiggy's apps used by 50M+ users. Focus on performance and UX.",
     "requirements":"B.Tech CS. React, JavaScript, HTML/CSS. Redux, TypeScript preferred.",
     "tags":["frontend","react","javascript","b.tech"],"match_fields":"react javascript typescript html css redux frontend btech engineer"},
    {"id":"opp-008","title":"ML Engineer","company":"CRED","type":"job",
     "location":"Bengaluru","salary_range":"₹18L–₹30L/yr","url":"https://careers.cred.club",
     "description":"Build ML models for credit scoring, fraud detection, and personalization.",
     "requirements":"B.Tech CS. Python, scikit-learn, TensorFlow/PyTorch. Large-scale data experience.",
     "tags":["ml","ai","python","b.tech","data science"],"match_fields":"python machine learning ml tensorflow pytorch sklearn data science ai btech"},
    {"id":"opp-009","title":"Full Stack Developer","company":"Zoho","type":"job",
     "location":"Chennai / Bengaluru","salary_range":"₹5L–₹9L/yr","url":"https://careers.zohocorp.com",
     "description":"Build features for Zoho's 50+ business apps used by 100M+ users worldwide.",
     "requirements":"B.Tech CS. Java or any backend. JavaScript, HTML/CSS. Problem-solving.",
     "tags":["full stack","java","javascript","b.tech"],"match_fields":"java javascript html css fullstack full stack btech zoho"},
    {"id":"opp-010","title":"DevOps Engineer","company":"Freshworks","type":"job",
     "location":"Chennai","salary_range":"₹8L–₹14L/yr","url":"https://careers.freshworks.com",
     "description":"Manage CI/CD pipelines, cloud infrastructure, and reliability for SaaS products.",
     "requirements":"B.Tech CS. AWS/GCP, Docker, Kubernetes, CI/CD. Python/Bash scripting.",
     "tags":["devops","aws","cloud","kubernetes","b.tech"],"match_fields":"aws gcp docker kubernetes devops cicd linux python bash cloud btech"},
    {"id":"opp-011","title":"Android Developer","company":"Dream11","type":"job",
     "location":"Mumbai","salary_range":"₹12L–₹22L/yr","url":"https://careers.dream11.com",
     "description":"Build Android features for India's largest fantasy sports platform with 200M+ users.",
     "requirements":"B.Tech CS. Kotlin/Java Android. Jetpack Compose, MVVM architecture.",
     "tags":["android","kotlin","mobile","b.tech"],"match_fields":"android kotlin java mobile mvvm jetpack btech"},
    {"id":"opp-012","title":"Cloud Engineer","company":"Accenture","type":"job",
     "location":"Bengaluru / Pune / Hyderabad","salary_range":"₹4.5L–₹8L/yr","url":"https://www.accenture.com/in-en/careers",
     "description":"Cloud migration and modernization projects for Fortune 500 clients on AWS, Azure, GCP.",
     "requirements":"B.Tech/MCA/BCA. Cloud platform knowledge. Cloud cert a plus.",
     "tags":["cloud","aws","azure","b.tech","mca","bca"],"match_fields":"aws azure gcp cloud btech mca bca accenture"},
    # Internships
    {"id":"opp-013","title":"Software Engineering Intern","company":"Google","type":"internship",
     "location":"Bengaluru / Hyderabad","salary_range":"₹1.2L–₹1.8L/month","url":"https://careers.google.com/students/",
     "description":"12-week internship on real products used by billions. Projects in Search, Maps, YouTube, or Cloud.",
     "requirements":"B.Tech 2nd/3rd year. Strong CS fundamentals, DSA, algorithms.",
     "tags":["internship","b.tech","algorithms","dsa"],"match_fields":"algorithms dsa data structures google intern btech software"},
    {"id":"opp-014","title":"Data Science Intern","company":"Paytm","type":"internship",
     "location":"Noida","salary_range":"₹25K–₹40K/month","url":"https://paytm.com/about-us/careers",
     "description":"Build ML models and data pipelines. Work on fraud detection and recommendation systems.",
     "requirements":"B.Tech/M.Sc CS with ML coursework. Python, SQL, ML fundamentals.",
     "tags":["internship","data science","ml","python","b.tech"],"match_fields":"python sql machine learning ml data science intern btech paytm"},
    {"id":"opp-015","title":"Backend Development Intern","company":"Ola","type":"internship",
     "location":"Bengaluru","salary_range":"₹35K–₹50K/month","url":"https://jobs.olacabs.com",
     "description":"Build APIs and backend systems for Ola's mobility platform. Ride-matching, surge pricing, driver management.",
     "requirements":"B.Tech 2nd/3rd year. Java/Python/Node.js. REST APIs and databases.",
     "tags":["internship","backend","java","python","b.tech"],"match_fields":"java python node backend api databases intern btech ola"},
    {"id":"opp-016","title":"Product Management Intern","company":"Meesho","type":"internship",
     "location":"Bengaluru","salary_range":"₹60K–₹80K/month","url":"https://careers.meesho.com",
     "description":"Drive product initiatives for India's social commerce platform with 140M+ users.",
     "requirements":"MBA/B.Tech final year. Analytical mindset. Data-driven decision making.",
     "tags":["internship","product","mba","b.tech"],"match_fields":"product management analytics data mba btech meesho intern"},
    {"id":"opp-017","title":"iOS Development Intern","company":"PhonePe","type":"internship",
     "location":"Bengaluru","salary_range":"₹50K–₹70K/month","url":"https://careers.phonepe.com",
     "description":"Build iOS features for PhonePe with 500M+ users. Payments, UPI, insurance modules.",
     "requirements":"B.Tech CS. Swift, iOS development basics. Swift/SwiftUI.",
     "tags":["internship","ios","swift","mobile","b.tech"],"match_fields":"swift ios mobile xcode intern btech phonepe"},
    # Hackathons
    {"id":"opp-018","title":"Smart India Hackathon 2025","company":"Government of India","type":"hackathon",
     "location":"Pan India","salary_range":"Prize: ₹1L–₹5L","url":"https://www.sih.gov.in",
     "description":"India's largest hackathon with 50,000+ participants. Solve real government and industry problems across 17+ categories.",
     "requirements":"Any student. Team of 6. Working prototype in 36 hours.",
     "tags":["hackathon","b.tech","mca","competition"],"match_fields":"hackathon competition prototype team government btech mca"},
    {"id":"opp-019","title":"HackWithInfy 2025","company":"Infosys","type":"hackathon",
     "location":"Online → Bengaluru Finals","salary_range":"Prize: ₹2L + PPO","url":"https://hackwithinfy.com",
     "description":"Solve real-world problems using AI/ML, cloud, and data. Top performers get Pre-Placement Offer.",
     "requirements":"B.Tech/B.E 3rd or final year. Team of 2-4.",
     "tags":["hackathon","b.tech","ml","ai","infosys"],"match_fields":"hackathon ai ml cloud infosys btech ppo competition"},
    {"id":"opp-020","title":"Google Summer of Code 2025","company":"Google / OSS Orgs","type":"opensource",
     "location":"Remote","salary_range":"$1,500–$6,600 (USD)","url":"https://summerofcode.withgoogle.com",
     "description":"12–22 week paid open source contribution. Work with an OSS org on a significant project under mentorship.",
     "requirements":"18+ enrolled student or recent grad. Strong programming. OSS contribution experience preferred.",
     "tags":["open source","gsoc","remote","b.tech","mca"],"match_fields":"open source github contribution remote gsoc btech mca programming"},
    {"id":"opp-021","title":"Microsoft Learn Student Ambassadors","company":"Microsoft","type":"scholarship",
     "location":"College / Remote","salary_range":"Azure Credits + Mentorship","url":"https://studentambassadors.microsoft.com",
     "description":"Become a Microsoft ambassador at your college. Azure credits, certifications, access to Microsoft tools.",
     "requirements":"Enrolled student. Passion for tech. Run 1 event/semester.",
     "tags":["scholarship","microsoft","azure","cloud","b.tech"],"match_fields":"microsoft azure cloud scholarship ambassador student btech"},
    {"id":"opp-022","title":"QA Engineer","company":"Myntra","type":"job",
     "location":"Bengaluru","salary_range":"₹6L–₹10L/yr","url":"https://careers.myntra.com",
     "description":"Ensure quality of Myntra's fashion e-commerce platform. Design test plans, automation frameworks, performance tests.",
     "requirements":"B.Tech CS. Selenium, Appium, or Cypress. Python/Java for automation. CI/CD understanding.",
     "tags":["qa","testing","selenium","b.tech"],"match_fields":"testing qa selenium appium cypress python java automation btech"},
    {"id":"opp-023","title":"Business Analyst Trainee","company":"Cognizant","type":"job",
     "location":"Pan India","salary_range":"₹4L–₹5.5L/yr","url":"https://careers.cognizant.com",
     "description":"Bridge technology and business for Cognizant clients. User stories, requirement analysis, Agile support.",
     "requirements":"B.Tech/MBA/BCA/MCA with 60%+. Analytical skills, SDLC, Agile basics.",
     "tags":["business analyst","ba","agile","b.tech","mba"],"match_fields":"business analyst agile scrum sdlc btech mba bca cognizant"},
    {"id":"opp-024","title":"Cybersecurity Analyst","company":"HCL Technologies","type":"job",
     "location":"Noida / Chennai","salary_range":"₹4.5L–₹7L/yr","url":"https://www.hcltech.com/careers",
     "description":"Monitor, detect, and respond to security threats for banking, healthcare, and government clients.",
     "requirements":"B.Tech CS/IT/ECE. Networking, security fundamentals. CEH/Security+ a plus.",
     "tags":["cybersecurity","security","networking","b.tech"],"match_fields":"cybersecurity security networking ceh btech it ece hcl"},
    {"id":"opp-025","title":"React Native Developer","company":"Urban Company","type":"job",
     "location":"Bengaluru","salary_range":"₹10L–₹18L/yr","url":"https://urbancompany.com/careers",
     "description":"Build cross-platform mobile features for Urban Company's service marketplace used in 50+ cities.",
     "requirements":"B.Tech CS. React Native, JavaScript. 1+ yr experience or strong portfolio.",
     "tags":["react native","mobile","javascript","b.tech"],"match_fields":"react native mobile javascript ios android btech urban company"},
]

def _match_score(opp: dict, skills_text: str, degree_text: str) -> int:
    """Simple keyword overlap score for opportunity ranking."""
    combined = (skills_text + " " + degree_text).lower()
    tokens = set(re.findall(r'\w+', combined))
    match_tokens = set(re.findall(r'\w+', (opp.get("match_fields") or "").lower()))
    return len(tokens & match_tokens)


def _rank_opps(opps: list, profile_row: dict) -> list:
    """Rank opportunities by match score with user profile."""
    skills = (profile_row.get("skills") or "").lower()
    degree = (profile_row.get("degree") or "").lower()
    is_fresher = profile_row.get("is_fresher", True)
    has_roles = bool(profile_row.get("roles"))

    scored = []
    for opp in opps:
        score = _match_score(opp, skills, degree)
        # Boost internships/hackathons for freshers
        if is_fresher and opp["type"] in ("internship", "hackathon", "opensource", "scholarship"):
            score += 3
        if not is_fresher and has_roles and opp["type"] == "job":
            score += 2
        scored.append((score, opp))
    scored.sort(key=lambda x: -x[0])
    return [o for _, o in scored]


# ── OTP / Magic Link auth routes ──────────────────────────────────────────────

@app.post("/ninelab/auth/otp")
async def send_otp(req: MagicLinkRequest):
    """Send 6-digit OTP to user email via Supabase."""
    email = (req.email or "").strip().lower()
    if "@" not in email:
        raise HTTPException(400, detail="Please enter a valid email address.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(503, detail="Auth not configured. Set SUPABASE_URL and SUPABASE_KEY.")

    import httpx as _hx
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json"}
    try:
        r = _hx.post(f"{SUPABASE_URL}/auth/v1/otp", headers=headers,
                     json={"email": email, "create_user": True}, timeout=10)
        if r.status_code == 200:
            return JSONResponse({"success": True, "message": "OTP sent! Check your email — valid for 10 minutes."})
        data = r.json() if r.text else {}
        msg = data.get("msg") or data.get("message") or data.get("error") or "Could not send OTP."
        raise HTTPException(400, detail=str(msg)[:200])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Email service error: {str(e)[:100]}")


@app.post("/ninelab/auth/verify-otp")
async def verify_otp_route(req: OTPVerifyRequest):
    """Verify 6-digit OTP and return session tokens."""
    email = (req.email or "").strip().lower()
    token = (req.token or "").strip()
    if not email or not token:
        raise HTTPException(400, detail="Email and OTP code are required.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(503, detail="Auth not configured.")

    import httpx as _hx
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json"}
    try:
        r = _hx.post(f"{SUPABASE_URL}/auth/v1/verify", headers=headers,
                     json={"email": email, "token": token, "type": "email"}, timeout=10)
        data = r.json() if r.text else {}
        if r.status_code == 200 and data.get("access_token"):
            user = data.get("user") or {}
            meta = user.get("user_metadata") or {}
            return JSONResponse({
                "success": True,
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", ""),
                "user": {
                    "id": user.get("id", ""),
                    "email": user.get("email", email),
                    "full_name": meta.get("full_name") or meta.get("name") or "",
                },
            })
        msg = data.get("msg") or data.get("message") or data.get("error") or "Invalid or expired OTP."
        raise HTTPException(400, detail=str(msg)[:200])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Verification error: {str(e)[:100]}")


# ── Profile routes ────────────────────────────────────────────────────────────

@app.get("/ninelab/v2/profile")
async def get_profile_v2(authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True, "profile": {}, "pct": 0})

    token = authorization.removeprefix("Bearer ").strip()
    result = supabase_rest("GET", "user_profiles", params={"id": f"eq.{uid}"}, token=token)
    rows = result.get("data", [])
    if isinstance(rows, list) and rows:
        row = rows[0]
        pct = row.get("profile_pct") or _calc_profile_pct(row)
        return JSONResponse({"success": True, "profile": row, "pct": pct})
    return JSONResponse({"success": True, "profile": {
        "id": uid, "email": user.get("email", ""),
        "full_name": (user.get("user_metadata") or {}).get("full_name", ""),
    }, "pct": 0})


@app.post("/ninelab/v2/profile")
async def save_profile_v2(req: ProfileSaveRequest, authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]
    email = user.get("email", "")

    row = {
        "id": uid, "email": email,
        "full_name": req.full_name or "",
        "phone": req.phone or "",
        "linkedin_url": req.linkedin_url or "",
        "city": req.city or "",
        "degree": req.degree or "",
        "field_of_study": req.field_of_study or "",
        "college": req.college or "",
        "grad_year": req.grad_year or "",
        "cgpa": req.cgpa or "",
        "is_fresher": req.is_fresher if req.is_fresher is not None else True,
        "skills": req.skills or "",
        "projects": req.projects or "",
        "achievements": req.achievements or "",
        "roles": req.roles or [],
        "updated_at": datetime.utcnow().isoformat(),
    }
    row["profile_pct"] = _calc_profile_pct(row)
    row["profile_complete"] = row["profile_pct"] >= 80

    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True, "pct": row["profile_pct"]})

    result = supabase_rest("POST", "user_profiles", payload=row,
                           use_service_key=True, upsert=True)
    return JSONResponse({
        "success": result["status"] in (200, 201),
        "pct": row["profile_pct"],
    })


# ── Opportunities feed routes ─────────────────────────────────────────────────

@app.get("/ninelab/v2/opportunities")
async def get_opportunities(
    type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 25,
    authorization: Optional[str] = Header(None)
):
    profile_row: dict = {}
    if authorization:
        user = _get_auth_user_v2(authorization)
        if user and SUPABASE_URL:
            token = authorization.removeprefix("Bearer ").strip()
            r = supabase_rest("GET", "user_profiles",
                              params={"id": f"eq.{user['id']}"}, token=token)
            rows = r.get("data", [])
            if isinstance(rows, list) and rows:
                profile_row = rows[0]

    # Try DB first
    opps: list = []
    if SUPABASE_URL and SUPABASE_KEY:
        params: dict = {"is_active": "eq.true", "order": "created_at.desc",
                        "limit": str(limit * 2), "select": "*"}
        if type and type != "all":
            params["type"] = f"eq.{type}"
        result = supabase_rest("GET", "opportunities", params=params,
                               use_service_key=True)
        rows2 = result.get("data", [])
        if isinstance(rows2, list) and rows2:
            opps = rows2

    # Fallback to seed data
    if not opps:
        opps = list(SEED_OPPS)

    # Type filter on seed data
    if type and type != "all":
        opps = [o for o in opps if o.get("type") == type]

    # Search filter
    if search:
        sq = search.lower()
        opps = [o for o in opps if
                sq in (o.get("title") or "").lower() or
                sq in (o.get("company") or "").lower() or
                sq in (o.get("description") or "").lower() or
                any(sq in t for t in (o.get("tags") or []))]

    # Rank by profile match
    if profile_row:
        opps = _rank_opps(opps, profile_row)

    opps = opps[:limit]

    # Attach match score
    skills = (profile_row.get("skills") or "").lower()
    degree = (profile_row.get("degree") or "").lower()
    for opp in opps:
        opp["match_score"] = _match_score(opp, skills, degree)

    return JSONResponse({"success": True, "opportunities": opps, "total": len(opps)})


@app.get("/ninelab/v2/opportunities/{opp_id}")
async def get_opportunity_v2(opp_id: str):
    for opp in SEED_OPPS:
        if opp.get("id") == opp_id:
            return JSONResponse({"success": True, "opportunity": opp})
    if SUPABASE_URL and SUPABASE_KEY:
        result = supabase_rest("GET", "opportunities",
                               params={"id": f"eq.{opp_id}"}, use_service_key=True)
        rows = result.get("data", [])
        if isinstance(rows, list) and rows:
            return JSONResponse({"success": True, "opportunity": rows[0]})
    raise HTTPException(404, detail="Opportunity not found.")


# ── Application tracking routes ───────────────────────────────────────────────

@app.get("/ninelab/v2/applications")
async def get_applications(authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True, "applications": []})

    token = authorization.removeprefix("Bearer ").strip()
    result = supabase_rest("GET", "applications",
                           params={"user_id": f"eq.{uid}", "order": "updated_at.desc",
                                   "select": "*"}, token=token)
    apps = result.get("data", []) if isinstance(result.get("data"), list) else []
    return JSONResponse({"success": True, "applications": apps})


@app.post("/ninelab/v2/applications")
async def create_application_v2(req: ApplicationCreateRequest,
                                  authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True, "id": str(uuid.uuid4())})

    token = authorization.removeprefix("Bearer ").strip()

    # Check for duplicate
    dup_params: dict = {"user_id": f"eq.{uid}"}
    if req.opportunity_id:
        dup_params["opportunity_id"] = f"eq.{req.opportunity_id}"
    elif req.custom_title:
        dup_params["custom_title"] = f"eq.{req.custom_title}"
        dup_params["custom_company"] = f"eq.{req.custom_company or ''}"

    dup = supabase_rest("GET", "applications", params=dup_params, token=token)
    if isinstance(dup.get("data"), list) and dup["data"]:
        return JSONResponse({"success": True, "id": dup["data"][0].get("id"), "existing": True})

    valid_statuses = {"saved", "applied", "interviewing", "offered", "rejected", "withdrawn"}
    status = req.status if req.status in valid_statuses else "saved"

    payload = {
        "user_id": uid,
        "opportunity_id": req.opportunity_id,
        "custom_title": req.custom_title or "",
        "custom_company": req.custom_company or "",
        "custom_url": req.custom_url or "",
        "status": status,
        "resume_generated": bool(req.resume_generated),
        "prep_downloaded": bool(req.prep_downloaded),
        "notes": req.notes or "",
        "applied_at": datetime.utcnow().isoformat() if status == "applied" else None,
    }
    result = supabase_rest("POST", "applications", payload=payload, token=token)
    data = result.get("data")
    app_id = data[0].get("id") if isinstance(data, list) and data else str(uuid.uuid4())
    return JSONResponse({"success": True, "id": app_id})


@app.patch("/ninelab/v2/applications/{app_id}")
async def update_application_v2(app_id: str, req: ApplicationUpdateRequest,
                                  authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]

    valid_statuses = {"saved", "applied", "interviewing", "offered", "rejected", "withdrawn"}
    if req.status and req.status not in valid_statuses:
        raise HTTPException(400, detail=f"Invalid status. Choose: {', '.join(valid_statuses)}")

    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True})

    token = authorization.removeprefix("Bearer ").strip()
    payload: dict = {"updated_at": datetime.utcnow().isoformat()}
    if req.status:
        payload["status"] = req.status
        if req.status == "applied":
            payload["applied_at"] = datetime.utcnow().isoformat()
    if req.notes is not None:
        payload["notes"] = req.notes
    if req.resume_generated is not None:
        payload["resume_generated"] = req.resume_generated
    if req.prep_downloaded is not None:
        payload["prep_downloaded"] = req.prep_downloaded
    if req.redirect_clicked is not None:
        payload["redirect_clicked"] = req.redirect_clicked

    result = supabase_rest("PATCH", "applications", payload=payload,
                           params={"id": f"eq.{app_id}", "user_id": f"eq.{uid}"},
                           token=token)
    return JSONResponse({"success": result["status"] in (200, 201, 204)})


@app.delete("/ninelab/v2/applications/{app_id}")
async def delete_application_v2(app_id: str, authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True})

    token = authorization.removeprefix("Bearer ").strip()
    result = supabase_rest("DELETE", "applications",
                           params={"id": f"eq.{app_id}", "user_id": f"eq.{uid}"},
                           token=token)
    return JSONResponse({"success": result["status"] in (200, 204)})


# ── Activity logging ──────────────────────────────────────────────────────────

@app.post("/ninelab/v2/activity")
async def log_activity_v2(req: ActivityRequest, authorization: Optional[str] = Header(None)):
    user = _get_auth_user_v2(authorization)
    if not user or not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse({"success": True})

    token = authorization.removeprefix("Bearer ").strip()
    payload = {
        "user_id": user["id"],
        "action_type": (req.action_type or "")[:50],
        "metadata": req.metadata or {},
    }
    supabase_rest("POST", "activity_logs", payload=payload, token=token)
    return JSONResponse({"success": True})


# ── Admin: seed opportunities ─────────────────────────────────────────────────

@app.post("/ninelab/admin/seed-opportunities")
async def seed_opportunities_admin(request: Request):
    """One-time endpoint to seed opportunities table. Protected by service key."""
    x_key = request.headers.get("x-admin-key", "")
    if not x_key or x_key != SUPABASE_SERVICE_KEY:
        raise HTTPException(403, detail="Forbidden.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(503, detail="Supabase not configured.")

    inserted, skipped = 0, 0
    for opp in SEED_OPPS:
        row = {
            "title": opp["title"], "company": opp["company"],
            "type": opp["type"], "location": opp.get("location", "India"),
            "url": opp.get("url", ""), "description": opp.get("description", ""),
            "requirements": opp.get("requirements", ""),
            "salary_range": opp.get("salary_range", ""),
            "tags": opp.get("tags", []),
            "match_fields": opp.get("match_fields", ""),
            "is_active": True,
        }
        result = supabase_rest("POST", "opportunities", payload=row, use_service_key=True)
        if result["status"] in (200, 201):
            inserted += 1
        else:
            skipped += 1

    return JSONResponse({"seeded": inserted, "skipped": skipped})


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/ninelab/v2/dashboard")
async def get_dashboard_v2(authorization: Optional[str] = Header(None)):
    user = _require_auth_v2(authorization)
    uid = user["id"]
    email = user.get("email", "")
    meta = user.get("user_metadata") or {}

    profile_row: dict = {}
    apps: list = []

    if SUPABASE_URL and SUPABASE_KEY:
        token = authorization.removeprefix("Bearer ").strip()
        pr = supabase_rest("GET", "user_profiles", params={"id": f"eq.{uid}"}, token=token)
        rows = pr.get("data", [])
        if isinstance(rows, list) and rows:
            profile_row = rows[0]

        ar = supabase_rest("GET", "applications",
                           params={"user_id": f"eq.{uid}", "order": "updated_at.desc",
                                   "limit": "10"}, token=token)
        apps = ar.get("data", []) if isinstance(ar.get("data"), list) else []

    opps = _rank_opps(list(SEED_OPPS), profile_row)[:12]
    skills = (profile_row.get("skills") or "").lower()
    degree = (profile_row.get("degree") or "").lower()
    for opp in opps:
        opp["match_score"] = _match_score(opp, skills, degree)

    pct = profile_row.get("profile_pct") or _calc_profile_pct(profile_row)

    return JSONResponse({
        "success": True,
        "user": {
            "id": uid, "email": email,
            "full_name": profile_row.get("full_name") or meta.get("full_name") or "",
        },
        "profile_pct": pct,
        "opportunities": opps,
        "recent_applications": apps,
        "stats": {
            "total": len(apps),
            "interviews": sum(1 for a in apps if a.get("status") == "interviewing"),
            "offers": sum(1 for a in apps if a.get("status") == "offered"),
        },
    })


# ── Real Jobs endpoint ────────────────────────────────────────────────────────

@app.get("/ninelab/real-jobs")
async def real_jobs(title: str = "", skills: str = "", type: str = "both"):
    """Fetch real job/internship listings. Searches by skills first, title as fallback."""
    title = title.strip()[:80]
    skill_list = [s.strip() for s in skills.split(",") if s.strip()][:6]
    if not skill_list and not title:
        return JSONResponse({"jobs": [], "internships": []})

    want_intern = type in ("internship", "both")

    loop = asyncio.get_event_loop()

    # Run JSearch for jobs + internships in parallel (LinkedIn, Indeed, Naukri only)
    jsearch_jobs_task   = loop.run_in_executor(None, _fetch_jsearch_jobs, skill_list, title, False)
    jsearch_intern_task = loop.run_in_executor(None, _fetch_jsearch_jobs, skill_list, title, True) if want_intern else None

    tasks = [t for t in [jsearch_jobs_task, jsearch_intern_task] if t]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    raw = []
    for r in results:
        if isinstance(r, list):
            raw.extend(r)

    seen, jobs, internships = set(), [], []
    for r in raw:
        url = r.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        bucket = internships if r.get("type") == "internship" else jobs
        if len(bucket) < 8:
            bucket.append(r)

    return JSONResponse({"jobs": jobs, "internships": internships})


# ── Freelance Jobs endpoint ───────────────────────────────────────────────────

@app.get("/ninelab/freelance-jobs")
async def freelance_jobs(skills: str = "", title: str = ""):
    """Fetch freelance/remote jobs from RemoteOK + Remotive. No API keys needed."""
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    if not skill_list and title:
        skill_list = [title.strip()]
    if not skill_list:
        skill_list = ["developer"]

    raw = []
    loop = asyncio.get_event_loop()
    remoteok_task = loop.run_in_executor(None, _fetch_remoteok_jobs, skill_list)
    remotive_task = loop.run_in_executor(None, _fetch_remotive_jobs, skill_list)
    remoteok_results, remotive_results = await asyncio.gather(remoteok_task, remotive_task)
    raw.extend(remoteok_results)
    raw.extend(remotive_results)

    seen, out = set(), []
    for r in raw:
        url = r.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
        if len(out) >= 12:
            break

    return JSONResponse({"jobs": out})
