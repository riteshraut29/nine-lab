import re
def extract_section(text: str, header: str) -> str:
    pattern = rf'(?:^|\n)\s*(?:\d+\.\s*)?{re.escape(header)}[:\s]*\n?(.*?)(?=\n\s*(?:\d+\.\s*)?[A-Z][A-Z\s/&]{{3,}}\s*:|$)'
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
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

    return strip_md(m.group(1).strip()) if m else ""

def section_bullets(text: str) -> list:
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith(('===', '---'))]
    return [l.lstrip('\u2022•-*0123456789. ') for l in lines if len(l) > 3]

plan_text = """
PHASE 3: COMPANY-SPECIFIC POLISH
Some details

TOP 10 TECHNICAL INTERVIEW QUESTIONS WITH ANSWERS
1. What is closure in javascript? A closure is a function having access...
2. Explain event loop. The event loop is...

TOP 5 HR QUESTIONS WITH ANSWERS
1. Why our company? Because...
"""

tech_q = extract_section(plan_text, "TOP 10 TECHNICAL INTERVIEW QUESTIONS WITH ANSWERS")
print(f"Tech Q:\n{tech_q}")

flashcards = []
if tech_q:
    for b in section_bullets(tech_q)[:5]:
        parts = re.split(r'\?|\n|- |—|:|Answer:', b, maxsplit=1)
        if len(parts) > 1:
            flashcards.append({"q": parts[0].strip() + "?", "a": parts[1].strip()})
        else:
            flashcards.append({"q": "Interview Question", "a": b.strip()})
print(flashcards)
