import re
import io
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import streamlit as st
import streamlit_authenticator as stauth
from pypdf import PdfReader
from docx import Document as DocxDocument

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as pdf_canvas


# =========================
# App Identity
# =========================
APP_NAME = "DocuLite"
APP_TAGLINE = "Explain • Highlight • Questions • Glossary • Flashcards • Export"
st.set_page_config(page_title=APP_NAME, page_icon="📄", layout="wide")


# =========================
# Plans & Limits
# =========================
@dataclass
class Plan:
    name: str
    max_docs_per_day: int
    max_pages: int
    max_mb: int

OWNER_PLAN = Plan("OWNER", max_docs_per_day=10_000, max_pages=10_000, max_mb=300)
FREE_PLAN  = Plan("FREE",  max_docs_per_day=2,      max_pages=150,    max_mb=100)
# (Paid later: you can add PRO_PLAN)


# =========================
# Text helpers (fast, no NLTK)
# =========================
STOPWORDS = set("""
a an the and or but if while to of in on for with without at by from as is are was were be been being
this that these those it its into about over under between among than then so because therefore
i you he she they we them us our your my mine theirs his her
""".split())

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n").replace("\u00a0", " ")
    # fix hyphen line-break: inter-\nnational -> international
    s = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', s)
    # collapse spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    # Fix spaced letters: "C O N T E N T S" -> "CONTENTS"
    def _join(match):
        return match.group(0).replace(" ", "")
    s = re.sub(r'(?<!\w)(?:[A-Za-z]\s){2,}[A-Za-z](?!\w)', _join, s)

    # Convert single newlines to spaces, keep paragraphs
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘])', text)
    return [p.strip() for p in parts if p.strip()]

def tokenize_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]

def jaccard(a: str, b: str) -> float:
    sa = set(tokenize_words(a))
    sb = set(tokenize_words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    """Fast RAKE-lite: unigram + bigram scoring."""
    text = text or ""
    if len(text) < 60:
        return []
    words = tokenize_words(text)
    if not words:
        return []

    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    bigrams: Dict[str, int] = {}
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        bigrams[bg] = bigrams.get(bg, 0) + 1

    scored: List[Tuple[str, float]] = []
    for w, c in freq.items():
        scored.append((w, c * 1.0))
    for bg, c in bigrams.items():
        scored.append((bg, c * 1.8))

    scored.sort(key=lambda x: x[1], reverse=True)

    out, seen = [], set()
    for term, _ in scored:
        t = term.strip().lower()
        if t in seen:
            continue
        seen.add(t)
        out.append(term)
        if len(out) >= top_k:
            break
    return out

def summarize(text: str, n_sent: int, user_type: str) -> List[str]:
    sents = sentence_split(text)
    if not sents:
        return []

    if user_type == "Primary":
        n_sent = max(3, min(n_sent, 5))
    elif user_type == "High school":
        n_sent = max(4, min(n_sent, 7))
    elif user_type == "Varsity":
        n_sent = max(6, min(n_sent, 10))
    elif user_type == "Trader":
        n_sent = max(6, min(n_sent, 10))
    else:
        n_sent = max(5, min(n_sent, 9))

    kws = set([k.split()[0].lower() for k in extract_keywords(text, top_k=24) if k])
    scored = []
    for idx, s in enumerate(sents):
        w = tokenize_words(s)
        hit = sum(1 for x in w if x in kws)
        density = hit / max(1, len(w))
        length_penalty = 1.0 if len(s) <= 220 else max(0.6, 220 / len(s))
        score = (hit + 2.0 * density) * length_penalty
        scored.append((idx, s, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    picked = sorted(scored[: min(n_sent, len(scored))], key=lambda x: x[0])
    return [re.sub(r"\s+", " ", s).strip() for _, s, _ in picked]

def key_takeaways(text: str, k: int = 6) -> List[str]:
    candidates = summarize(text, n_sent=min(16, max(10, k * 2)), user_type="Varsity")
    out: List[str] = []
    for s in candidates:
        if len(out) >= k:
            break
        if all(jaccard(s, prev) < 0.55 for prev in out):
            out.append(s)
    return out

def generate_questions(text: str, user_type: str, n: int = 10) -> List[str]:
    kws = extract_keywords(text, top_k=max(24, n * 3))
    takes = key_takeaways(text, k=6)

    if user_type == "Primary":
        templates = [
            "What does '{k}' mean?",
            "Give an example of '{k}'.",
            "Explain '{k}' in your own words.",
            "Write one fact about '{k}'.",
        ]
    elif user_type == "High school":
        templates = [
            "Define '{k}'.",
            "Why is '{k}' important here?",
            "Compare '{k}' with a related idea from the text.",
            "Give an example and explain '{k}'.",
        ]
    elif user_type == "Trader":
        templates = [
            "Explain '{k}' and how to apply it in a trading plan (steps).",
            "What mistake happens if '{k}' is misunderstood?",
            "Create a checklist that includes '{k}'.",
            "When might '{k}' fail? Give one condition.",
        ]
    else:  # Varsity / General
        templates = [
            "Critically evaluate '{k}' based on the text.",
            "How does '{k}' connect to the main argument?",
            "Identify assumptions behind '{k}'.",
            "Apply '{k}' to a real-world scenario and justify.",
        ]

    base: List[str] = []
    for i, k in enumerate(kws):
        base.append(templates[i % len(templates)].format(k=k))
    for t in takes:
        base.append(f"Explain this statement and why it matters: “{t[:130]}…”")

    picked: List[str] = []
    for q in base:
        if len(picked) >= n:
            break
        if all(jaccard(q, prev) < 0.55 for prev in picked):
            picked.append(q)
    return picked[:n]

def highlight_html(text: str, keywords: List[str], max_chars: int = 120000) -> str:
    text = (text or "")[:max_chars]
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    keywords = sorted([k.strip() for k in (keywords or []) if k.strip()], key=len, reverse=True)
    for kw in keywords[:25]:
        pat = re.compile(rf"(?i)\b({re.escape(kw)})\b")
        safe = pat.sub(r"<mark>\1</mark>", safe)
    return f"<div style='white-space:pre-wrap; line-height:1.65'>{safe}</div>"

def glossary_search(term: str, text: str, limit: int = 8) -> List[str]:
    term = (term or "").strip()
    if not term:
        return []
    hits = []
    for s in sentence_split(text):
        if re.search(rf"(?i)\b{re.escape(term)}\b", s):
            hits.append(s.strip())
        if len(hits) >= limit:
            break
    return hits

def make_flashcards(text: str, keywords: List[str], n_cards: int = 10) -> List[Dict[str, Any]]:
    sents = [s for s in sentence_split(text) if len(s) >= 60]
    pool = [k for k in (keywords or []) if 3 <= len(k) <= 35]
    pool = list(dict.fromkeys(pool))

    cards, used = [], set()
    for sent in sents:
        if len(cards) >= n_cards:
            break
        chosen = None
        for k in pool:
            if re.search(rf"(?i)\b{re.escape(k)}\b", sent):
                chosen = k
                break
        if not chosen:
            continue

        key = (sent[:80] + chosen).lower()
        if key in used:
            continue
        used.add(key)

        blanked = re.sub(rf"(?i)\b{re.escape(chosen)}\b", "_____", sent, count=1)

        if len(cards) % 2 == 0:
            cards.append({"type": "typed", "q": f"Fill in the missing term:\n\n{blanked}", "a": chosen})
        else:
            distractors = [x for x in pool if x.lower() != chosen.lower()][:3]
            options = [chosen] + distractors
            options = list(dict.fromkeys(options))
            cards.append({"type": "mcq", "q": f"Which term best completes the sentence?\n\n{blanked}", "a": chosen, "options": options})
    return cards


# =========================
# File extraction
# =========================
def bytes_mb(n: int) -> float:
    return n / (1024 * 1024)

def extract_docx_text(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return normalize_text("\n".join(paras))

def extract_txt_text(file_bytes: bytes) -> str:
    try:
        return normalize_text(file_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return ""

def pdf_total_pages(file_bytes: bytes) -> int:
    reader = PdfReader(io.BytesIO(file_bytes))
    return len(reader.pages)

def extract_pdf_range(file_bytes: bytes, start_page1: int, end_page1: int, progress_cb=None) -> Tuple[str, List[str], int]:
    reader = PdfReader(io.BytesIO(file_bytes))
    total = len(reader.pages)
    start0 = max(0, min(total - 1, start_page1 - 1))
    end0 = max(0, min(total - 1, end_page1 - 1))
    if end0 < start0:
        start0, end0 = end0, start0

    pages_to_read = end0 - start0 + 1
    per_page, out = [], []

    for j, i in enumerate(range(start0, end0 + 1), start=1):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        t = normalize_text(t)
        per_page.append(t)
        if t:
            out.append(t)
        if progress_cb:
            progress_cb(j, pages_to_read)

    return normalize_text("\n\n".join(out)), per_page, pages_to_read


# =========================
# Export
# =========================
def export_pdf(title: str, summary: List[str], takeaways: List[str], questions: List[str]) -> bytes:
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x = 2.0 * cm
    y = height - 2.0 * cm

    def draw_wrapped(txt: str, y: float, font="Helvetica", size=11, leading=14) -> float:
        c.setFont(font, size)
        max_w = width - 4.0 * cm
        words = txt.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, font, size) <= max_w:
                line = test
            else:
                c.drawString(x, y, line)
                y -= leading
                line = w
                if y < 2.0 * cm:
                    c.showPage()
                    y = height - 2.0 * cm
                    c.setFont(font, size)
        if line:
            c.drawString(x, y, line)
            y -= leading
        return y

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 24

    c.setFont("Helvetica", 10)
    y = draw_wrapped(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", y, size=10)
    y -= 8

    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Summary")
    y -= 18
    for s in summary or ["(No summary)"]:
        y = draw_wrapped(f"- {s}", y)

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Key Takeaways")
    y -= 18
    for i, t in enumerate(takeaways or ["(No takeaways)"], start=1):
        y = draw_wrapped(f"{i}. {t}", y)

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Questions")
    y -= 18
    for i, q in enumerate(questions or ["(No questions)"], start=1):
        y = draw_wrapped(f"{i}. {q}", y)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# =========================
# Session state (Free limits per day, per user)
# NOTE: Without a DB this resets if Streamlit restarts.
# =========================
def init_state():
    if "usage" not in st.session_state:
        st.session_state.usage = {}  # {username: {"date":"YYYY-MM-DD","docs":0}}
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "cards" not in st.session_state:
        st.session_state.cards = []
    if "card_answers" not in st.session_state:
        st.session_state.card_answers = {}

init_state()


# =========================
# AUTH (Step B)
# =========================
# Put AUTH_CONFIG in Streamlit Secrets (see below)
auth_cfg = st.secrets.get("AUTH_CONFIG", None)
if auth_cfg is None:
    st.error("Missing AUTH_CONFIG in Streamlit Secrets. Add it first (see secrets template).")
    st.stop()

authenticator = stauth.Authenticate(
    auth_cfg["credentials"],
    auth_cfg["cookie"]["name"],
    auth_cfg["cookie"]["key"],
    auth_cfg["cookie"]["expiry_days"],
)

st.title(f"📄 {APP_NAME}")
st.caption(APP_TAGLINE)

name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("Incorrect username or password.")
    st.stop()
if auth_status is None:
    st.info("Please log in to use the app.")
    st.stop()

# Identify owner (unlimited)
OWNER_USERNAME = str(st.secrets.get("OWNER_USERNAME", "")).strip().lower()
is_owner = (str(username).lower() == OWNER_USERNAME)
plan = OWNER_PLAN if is_owner else FREE_PLAN

with st.sidebar:
    st.success(f"Logged in as: {name} ({username})")
    st.write(f"Plan: **{plan.name}**")
    authenticator.logout("Logout", "sidebar")
    st.divider()
    st.caption("Limits apply to FREE users only (owner is unlimited).")


# =========================
# Free usage check
# =========================
today = datetime.now().strftime("%Y-%m-%d")
u = st.session_state.usage.get(username, {"date": today, "docs": 0})
if u.get("date") != today:
    u = {"date": today, "docs": 0}
st.session_state.usage[username] = u

def can_process_one_more() -> bool:
    if plan.name == "OWNER":
        return True
    return u["docs"] < plan.max_docs_per_day


# =========================
# UI Controls
# =========================
colA, colB = st.columns(2)
with colA:
    user_type = st.selectbox("Who are you?", ["Varsity", "High school", "Primary", "Trader", "General"], index=0)
with colB:
    doc_mode = st.selectbox("Document type", ["Notes/Book (PDF/DOCX/TXT)", "Slides (PDF only)"], index=0)

summary_len = st.slider("Summary length (sentences)", 3, 12, 7)
keyword_count = st.slider("Keywords to highlight", 5, 30, 12)
question_count = st.slider("Questions to generate", 5, 25, 12)
flashcard_count = st.slider("Flashcards to generate", 5, 25, 10)

st.divider()


# =========================
# Upload Gate (Free users)
# =========================
if not can_process_one_more():
    st.error(f"Daily limit reached (FREE: {plan.max_docs_per_day} docs/day). Try tomorrow or upgrade later.")
    st.stop()

help_text = f"FREE limits: {FREE_PLAN.max_mb}MB, {FREE_PLAN.max_pages} pages, {FREE_PLAN.max_docs_per_day} docs/day." if not is_owner else "OWNER: unlimited usage."
uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"], help=help_text)

if not uploaded:
    st.stop()

file_bytes = uploaded.getvalue()
size_mb = bytes_mb(len(file_bytes))
ext = uploaded.name.lower().split(".")[-1]

if size_mb > plan.max_mb:
    st.error(f"File too large: {size_mb:.1f}MB. Limit: {plan.max_mb}MB.")
    st.stop()

# For slides mode: force PDF
if doc_mode.startswith("Slides") and ext != "pdf":
    st.error("Slides mode requires a PDF (export slides to PDF first).")
    st.stop()

start_page = 1
end_page = 1
total_pages = None

if ext == "pdf":
    try:
        total_pages = pdf_total_pages(file_bytes)
    except Exception:
        st.error("Could not read PDF.")
        st.stop()

    st.subheader("⚡ PDF Range (faster)")
    c1, c2 = st.columns(2)
    with c1:
        start_page = st.number_input("Start page", min_value=1, max_value=total_pages, value=1, step=1)
    with c2:
        default_end = min(10, total_pages)
        end_page = st.number_input("End page", min_value=1, max_value=total_pages, value=default_end, step=1)

    span = abs(int(end_page) - int(start_page)) + 1
    if span > plan.max_pages:
        st.error(f"Page range too large: {span} pages. Limit: {plan.max_pages} pages for your plan.")
        st.stop()

if st.button("⚡ Process Now", type="primary"):
    prog = st.progress(0, text="Starting...")

    def progress_cb(done: int, total: int):
        pct = int((done / max(1, total)) * 100)
        prog.progress(pct, text=f"Extracting… {done}/{total}")

    # Extract
    if ext == "pdf":
        text, per_page, pages_read = extract_pdf_range(file_bytes, int(start_page), int(end_page), progress_cb=progress_cb)
    elif ext == "docx":
        prog.progress(35, text="Reading DOCX…")
        text = extract_docx_text(file_bytes)
        per_page, pages_read = [], None
        prog.progress(100, text="Done.")
    else:
        prog.progress(35, text="Reading TXT…")
        text = extract_txt_text(file_bytes)
        per_page, pages_read = [], None
        prog.progress(100, text="Done.")

    if not text or len(text) < 120:
        prog.empty()
        st.error("No readable text extracted. If your PDF is scanned/image-only, it needs OCR (we can add later).")
        st.stop()

    # Analyze
    prog.progress(85, text="Analyzing…")
    keywords = extract_keywords(text, top_k=keyword_count)
    summary = summarize(text, n_sent=summary_len, user_type=user_type)
    takeaways = key_takeaways(text, k=min(10, max(5, summary_len)))
    questions = generate_questions(text, user_type=user_type, n=question_count)

    result = {
        "filename": uploaded.name,
        "ext": ext,
        "doc_mode": doc_mode,
        "user_type": user_type,
        "range": (int(start_page), int(end_page)) if ext == "pdf" else None,
        "size_mb": size_mb,
        "keywords": keywords,
        "summary": summary,
        "takeaways": takeaways,
        "questions": questions,
        "text": text,
        "per_page": per_page if ext == "pdf" else [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.session_state.last_result = result
    st.session_state.history.insert(0, result)

    # Count usage for free
    if plan.name != "OWNER":
        u["docs"] += 1
        st.session_state.usage[username] = u

    prog.progress(100, text="Done ✅")
    time.sleep(0.2)
    prog.empty()


res = st.session_state.last_result
if not res:
    st.stop()

st.subheader("✅ Results")
meta = f"File: {res['filename']} • Size: {res['size_mb']:.1f}MB • User: {res['user_type']}"
if res.get("range"):
    meta += f" • Pages: {res['range'][0]}-{res['range'][1]}"
st.caption(meta)

tabs = st.tabs(["✅ Explanation", "🟣 Highlights", "❓ Questions", "📘 Glossary", "🃏 Flashcards", "🆚 Compare", "🕘 History", "⬇️ Export"])

with tabs[0]:
    st.markdown("### Summary (readable)")
    st.write("\n".join([f"- {s}" for s in res["summary"]]))

    st.markdown("### Key Takeaways")
    for t in res["takeaways"]:
        st.markdown(f"- {t}")

with tabs[1]:
    st.markdown("### Keywords")
    st.write(", ".join(res["keywords"]) if res["keywords"] else "(none)")

    st.markdown("### Highlighted inside original text")
    if res["doc_mode"].startswith("Slides") and res["ext"] == "pdf":
        st.info("Slides mode: page-by-page highlights.")
        per = res.get("per_page") or []
        base_page = res["range"][0] if res.get("range") else 1
        for i, ptxt in enumerate(per, start=base_page):
            if not ptxt:
                continue
            st.markdown(f"#### Slide/Page {i}")
            kws_slide = extract_keywords(ptxt, top_k=min(10, keyword_count))
            st.markdown(highlight_html(ptxt, kws_slide, max_chars=25000), unsafe_allow_html=True)
    else:
        st.markdown(highlight_html(res["text"], res["keywords"]), unsafe_allow_html=True)

with tabs[2]:
    st.markdown("### Questions (non-repeating)")
    for i, q in enumerate(res["questions"], start=1):
        st.markdown(f"**{i}.** {q}")

with tabs[3]:
    st.markdown("### Glossary / Definitions (search inside doc)")
    term = st.text_input("Type a word/term to search", key="gloss_term")
    if st.button("Search term"):
        hits = glossary_search(term, res["text"], limit=8)
        if not hits:
            st.warning("No matches found.")
        else:
            for h in hits:
                st.markdown(f"- {h}")

with tabs[4]:
    st.markdown("### Flashcards (practice mode)")
    if st.button("Generate flashcards"):
        st.session_state.cards = make_flashcards(res["text"], res["keywords"], n_cards=flashcard_count)
        st.session_state.card_answers = {}
        st.success(f"Generated {len(st.session_state.cards)} flashcards!")

    cards = st.session_state.cards
    if not cards:
        st.info("Generate flashcards to start.")
    else:
        for idx, c in enumerate(cards):
            st.markdown(f"#### Card {idx+1}")
            st.write(c["q"])
            if c["type"] == "mcq":
                choice = st.radio("Choose", c["options"], key=f"mcq_{idx}")
                st.session_state.card_answers[idx] = choice
            else:
                typed = st.text_input("Type your answer", key=f"typed_{idx}")
                st.session_state.card_answers[idx] = typed
            st.divider()

        if st.button("Check answers"):
            score = 0
            for idx, c in enumerate(cards):
                user_ans = (st.session_state.card_answers.get(idx) or "").strip()
                correct = (c["a"] or "").strip()
                ok = user_ans.lower() == correct.lower() if c["type"] == "mcq" else (correct.lower() in user_ans.lower())
                if ok:
                    score += 1
                    st.success(f"Card {idx+1}: Correct ✅")
                else:
                    st.error(f"Card {idx+1}: Wrong ❌ (Correct: {correct})")
            st.info(f"Score: {score}/{len(cards)}")

with tabs[5]:
    st.markdown("### Compare (last two documents)")
    if len(st.session_state.history) < 2:
        st.info("Process at least 2 docs to compare.")
    else:
        a = st.session_state.history[0]
        b = st.session_state.history[1]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### A: {a['filename']}")
            st.write("\n".join([f"- {s}" for s in a["summary"]]))
        with c2:
            st.markdown(f"#### B: {b['filename']}")
            st.write("\n".join([f"- {s}" for s in b["summary"]]))

        common = sorted(set([k.lower() for k in a["keywords"]]) & set([k.lower() for k in b["keywords"]]))
        st.markdown("#### Common keywords")
        st.write(", ".join(common) if common else "(none)")

with tabs[6]:
    st.markdown("### History (this session)")
    if not st.session_state.history:
        st.info("No history yet.")
    else:
        for h in st.session_state.history[:10]:
            meta = f"{h['created_at']} • {h['user_type']} • {h['size_mb']:.1f}MB"
            if h.get("range"):
                meta += f" • pages {h['range'][0]}-{h['range'][1]}"
            st.markdown(f"**{h['filename']}**")
            st.caption(meta)
            st.write(", ".join(h["keywords"][:12]) if h.get("keywords") else "")
            st.divider()

        export = {"exported_at": datetime.now().isoformat(timespec="seconds"), "history": st.session_state.history}
        st.download_button(
            "⬇️ Download History (JSON)",
            data=json.dumps(export, indent=2),
            file_name="doculite_history.json",
            mime="application/json"
        )

with tabs[7]:
    st.markdown("### Export Summary + Questions")
    pdf_bytes = export_pdf(APP_NAME, res["summary"], res["takeaways"], res["questions"])
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="doculite_export.pdf", mime="application/pdf")

st.caption("Note: scanned/image PDFs need OCR to extract text. We can add OCR later.")
