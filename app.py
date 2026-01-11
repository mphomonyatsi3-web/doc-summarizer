import os
import re
import html
import time
from io import BytesIO

import numpy as np
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Export
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


# -----------------------------
# UI polish
# -----------------------------
CUSTOM_CSS = """
<style>
.main .block-container { padding-top: 1.6rem; max-width: 1150px; }
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.95rem; }
.card {
  background: white;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 16px 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
  margin-bottom: 14px;
}
.kwd { background: #fff3bf; padding: 0px 4px; border-radius: 6px; }
.pill {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10); margin: 4px 6px 0 0; font-size: 0.9rem;
}
</style>
"""
st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Secrets / PIN roles (set these in Streamlit Secrets)
# -----------------------------
OWNER_PIN = os.getenv("OWNER_PIN", "")
TRIAL_PIN = os.getenv("TRIAL_PIN", "")
PAID_PIN  = os.getenv("PAID_PIN", "")

# Brute-force protection
MAX_PIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 10 * 60  # 10 minutes

# Limits (✅ Paid max upload now 300MB)
LIMITS = {
    "trial": {
        "max_mb": 100,
        "max_pdf_pages": 150,
        "docs_per_hour": 2,
    },
    "paid": {
        "max_mb": 300,          # ✅ 300MB cap
        "max_pdf_pages": 200,
        "docs_per_hour": 10,
    },
    "owner": {
        "max_mb": 300,          # ✅ keep owner within server cap (Streamlit maxUploadSize=300)
        "max_pdf_pages": 2000,
        "docs_per_hour": 999999,
    }
}


def is_locked_out():
    until = st.session_state.get("lockout_until")
    return until is not None and time.time() < until

def register_failed_attempt():
    attempts = st.session_state.get("pin_attempts", 0) + 1
    st.session_state["pin_attempts"] = attempts
    if attempts >= MAX_PIN_ATTEMPTS:
        st.session_state["lockout_until"] = time.time() + LOCKOUT_SECONDS

def reset_attempts():
    st.session_state["pin_attempts"] = 0
    st.session_state["lockout_until"] = None


def get_role_from_pin(pin: str):
    if OWNER_PIN and pin == OWNER_PIN:
        return "owner"
    if PAID_PIN and pin == PAID_PIN:
        return "paid"
    if TRIAL_PIN and pin == TRIAL_PIN:
        return "trial"
    return None


def auth_gate():
    if not OWNER_PIN and not TRIAL_PIN and not PAID_PIN:
        st.error("No PINs configured. Add OWNER_PIN / TRIAL_PIN / PAID_PIN in Streamlit Secrets.")
        st.stop()

    st.sidebar.subheader("🔒 Access")

    if is_locked_out():
        remaining = int(st.session_state["lockout_until"] - time.time())
        st.sidebar.error(f"Too many attempts. Try again in {remaining//60}m {remaining%60}s.")
        st.stop()

    pin = st.sidebar.text_input("Enter PIN", type="password")
    if not pin:
        st.stop()

    role = get_role_from_pin(pin)
    if role:
        st.session_state["role"] = role
        reset_attempts()
        if role == "owner":
            st.sidebar.success("Owner access ✅")
        elif role == "paid":
            st.sidebar.success("Paid access ✅")
        else:
            st.sidebar.success("Free access ✅")
        return

    register_failed_attempt()
    st.sidebar.error("Wrong PIN")
    st.stop()


def role():
    return st.session_state.get("role", "trial")

def limits():
    return LIMITS.get(role(), LIMITS["trial"])


def enforce_rate_limit():
    # per-session hourly window
    now = time.time()
    window = 60 * 60
    hist = st.session_state.get("process_times", [])
    hist = [t for t in hist if now - t < window]

    lim = limits()
    if len(hist) >= lim["docs_per_hour"]:
        st.error("Usage limit reached for this hour. Please try again later.")
        st.stop()

    st.session_state["process_times"] = hist


def record_processed():
    hist = st.session_state.get("process_times", [])
    hist.append(time.time())
    st.session_state["process_times"] = hist


# -----------------------------
# Text extraction + helpers
# -----------------------------
def split_into_sentences(text: str):
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 25]

def extract_text_from_pdf(file, max_pages: int) -> str:
    reader = PdfReader(file)
    out = []
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages)
    for i in range(limit):
        t = reader.pages[i].extract_text()
        if t:
            out.append(t)
    return "\n".join(out)

def extract_text_from_docx(file) -> str:
    doc = DocxDocument(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# -----------------------------
# Smarter summarization (MMR)
# -----------------------------
def select_summary_sentences_mmr(sentences, k, lambda_param=0.70):
    if len(sentences) <= k:
        return list(range(len(sentences)))

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)

    centroid = X.mean(axis=0)
    rel = cosine_similarity(X, centroid).flatten()

    selected = []
    candidates = list(range(len(sentences)))

    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_score = -1e9
        best_idx = None
        for c in candidates:
            redundancy = max(cosine_similarity(X[c], X[s]).item() for s in selected)
            mmr = lambda_param * rel[c] - (1 - lambda_param) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = c
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected

def build_summary(text: str, k: int, keep_original_order: bool, mmr_lambda: float):
    sentences = split_into_sentences(text)
    if not sentences:
        return "", []

    idx = select_summary_sentences_mmr(sentences, k, lambda_param=mmr_lambda)
    if keep_original_order:
        idx = sorted(idx)

    chosen = [sentences[i] for i in idx]
    return " ".join(chosen), chosen


def extract_keywords(text: str, top_n: int, allow_phrases: bool):
    ngram_range = (1, 2) if allow_phrases else (1, 1)
    vec = TfidfVectorizer(stop_words="english", max_features=top_n, ngram_range=ngram_range)
    vec.fit([text])
    terms = list(vec.get_feature_names_out())

    cleaned, seen = [], set()
    for t in terms:
        t2 = t.strip().lower()
        if t2 and t2 not in seen and len(t2) >= 3:
            seen.add(t2)
            cleaned.append(t)
    return cleaned

def build_takeaways(summary_sentences, max_points=5):
    out = []
    for s in summary_sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > 150:
            s = s[:147].rstrip() + "..."
        out.append(s)
        if len(out) >= max_points:
            break
    return out

def generate_questions(mode: str, summary_sentences, k: int):
    qs = []
    for s in summary_sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 30:
            continue

        if mode == "Student":
            qs.append(f"Explain this in your own words: “{s}”")
            qs.append(f"What are 2 key points from: “{s}”?")
            qs.append(f"Give an example related to: “{s}”.")
        elif mode == "Business":
            qs.append(f"What is the main takeaway here: “{s}”?")
            qs.append(f"What action should be taken based on: “{s}”?")
            qs.append(f"What risk/implication is suggested by: “{s}”?")
        else:
            qs.append(f"How would you explain this clearly in an interview: “{s}”?")
            qs.append(f"What follow-up question could be asked from: “{s}”?")
            qs.append(f"How would you apply this idea in real life based on: “{s}”?")

        if len(qs) >= k:
            break
    return qs[:k]

def highlight_text_with_keywords(text, keywords):
    safe = html.escape(text)
    kws = sorted(set([k for k in keywords if len(k) >= 3]), key=len, reverse=True)
    for k in kws:
        escaped = html.escape(k)
        pattern = re.compile(rf"(?i)\b({re.escape(escaped)})\b")
        safe = pattern.sub(r'<span class="kwd">\1</span>', safe)
    return safe


# -----------------------------
# Export: PDF + DOCX
# -----------------------------
def build_pdf_bytes(title: str, summary_paragraph: str, takeaways, keywords, questions):
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(html.escape(summary_paragraph), styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Key Takeaways", styles["Heading2"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(html.escape(t), styles["BodyText"])) for t in takeaways],
        bulletType="bullet"
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Keywords", styles["Heading2"]))
    story.append(Paragraph(", ".join([html.escape(k) for k in keywords]), styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Questions", styles["Heading2"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(html.escape(q), styles["BodyText"])) for q in questions],
        bulletType="bullet"
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def build_docx_bytes(title: str, summary_paragraph: str, takeaways, keywords, questions):
    doc = DocxDocument()
    doc.add_heading(title, level=1)

    doc.add_heading("Summary", level=2)
    doc.add_paragraph(summary_paragraph)

    doc.add_heading("Key Takeaways", level=2)
    for t in takeaways:
        doc.add_paragraph(t, style="List Bullet")

    doc.add_heading("Keywords", level=2)
    doc.add_paragraph(", ".join(keywords))

    doc.add_heading("Questions", level=2)
    for q in questions:
        doc.add_paragraph(q, style="List Bullet")

    out = BytesIO()
    doc.save(out)
    out.seek(0)
    return out.getvalue()


# -----------------------------
# MAIN APP
# -----------------------------
auth_gate()

st.title("📄 Document Summarizer")
st.markdown('<div class="small-muted">Owner = unlimited tools. Free/Paid = protected limits.</div>', unsafe_allow_html=True)

lim = limits()

with st.sidebar:
    if st.button("🧹 Clear session / remove document"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    r = role()
    st.write("Role:", "👑 Owner" if r == "owner" else ("💳 Paid" if r == "paid" else "🆓 Free"))
    st.caption(f"Limits: {lim['max_mb']}MB • {lim['max_pdf_pages']} pages • {lim['docs_per_hour']}/hour")

    st.divider()
    mode = st.selectbox("Mode", ["Student", "Business", "Interview"])
    use_tabs = st.toggle("Use tabs layout", value=True)

    st.divider()
    summary_sentences = st.slider("Summary length (sentences)", 3, 12, 6)
    summary_format = st.selectbox("Summary format", ["Clean paragraph", "Slides / bullets"])
    keep_order = st.toggle("Keep original order", value=True)
    mmr_lambda = st.slider("Relevance ↔ Variety (MMR)", 0.45, 0.90, 0.70)

    st.divider()
    show_takeaways = st.toggle("Show Key Takeaways", value=True)
    takeaway_count = st.slider("Takeaways", 3, 10, 5)

    st.divider()
    allow_phrases = st.toggle("Keyword phrases (1–2 words)", value=True)
    keyword_count = st.slider("Keywords", 5, 25, 12)

    st.divider()
    question_count = st.slider("Questions", 3, 15, 8)

    st.divider()
    show_highlighted_text = st.toggle("Show highlighted original text", value=True)
    preview_chars = st.slider("Highlighted preview (chars)", 2000, 15000, 8000, step=1000)


uploaded = st.file_uploader("Upload a PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if not uploaded:
    st.info("Upload a document to begin.")
    st.stop()

# Rate limit check (per session)
enforce_rate_limit()

# File size limit
file_bytes = uploaded.getvalue()
size_mb = len(file_bytes) / (1024 * 1024)
if size_mb > lim["max_mb"]:
    st.error(f"File too large: {size_mb:.1f}MB. Max allowed is {lim['max_mb']}MB for your plan.")
    st.stop()

# Process button prevents heavy reruns
if "processed_key" not in st.session_state:
    st.session_state["processed_key"] = None

process_key = (
    uploaded.name, len(file_bytes), uploaded.type,
    mode, summary_sentences, keep_order, float(mmr_lambda),
    show_takeaways, takeaway_count,
    allow_phrases, keyword_count,
    question_count,
    show_highlighted_text, preview_chars,
    summary_format, use_tabs
)

if st.button("⚡ Process Document"):
    st.session_state["processed_key"] = process_key
    st.session_state["file_bytes"] = file_bytes
    st.session_state["file_type"] = uploaded.type
    st.session_state["file_name"] = uploaded.name

if st.session_state["processed_key"] != process_key:
    st.warning("Click **⚡ Process Document** to generate results.")
    st.stop()

try:
    with st.spinner("Reading document..."):
        ftype = st.session_state["file_type"]
        name = st.session_state["file_name"]
        data = st.session_state["file_bytes"]

        if ftype == "application/pdf":
            raw = extract_text_from_pdf(BytesIO(data), max_pages=lim["max_pdf_pages"])
        elif ftype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw = extract_text_from_docx(BytesIO(data))
        else:
            raw = data.decode("utf-8", errors="ignore")

    raw = (raw or "").strip()
    if len(raw) < 200:
        st.error("Not enough readable text. This might be a scanned PDF (image-only). OCR can be added later.")
        st.stop()

    with st.spinner("Summarizing & generating insights..."):
        summary_paragraph, summary_sents = build_summary(raw, summary_sentences, keep_order, float(mmr_lambda))
        keywords = extract_keywords(raw, keyword_count, allow_phrases)
        takeaways = build_takeaways(summary_sents, max_points=takeaway_count)
        questions = generate_questions(mode, summary_sents, k=question_count)

    record_processed()

    def render_summary():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Summary")
        if summary_format == "Clean paragraph":
            st.write(summary_paragraph)
        else:
            for s in summary_sents:
                st.write("•", s)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_takeaways():
        if not show_takeaways:
            return
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key Takeaways")
        for t in takeaways:
            st.write("•", t)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_keywords():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Keywords")
        for k in keywords:
            st.markdown(f'<span class="pill">{html.escape(k)}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_questions():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Questions")
        for q in questions:
            st.write("•", q)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_highlighted():
        if not show_highlighted_text:
            return
        st.subheader("Highlighted Original Text (Preview)")
        highlighted = highlight_text_with_keywords(raw[:preview_chars], keywords)
        st.markdown(f'<div class="card">{highlighted}</div>', unsafe_allow_html=True)

    def render_export():
        title = f"Summary Pack - {name}"
        pdf_bytes = build_pdf_bytes(title, summary_paragraph, takeaways, keywords, questions)
        docx_bytes = build_docx_bytes(title, summary_paragraph, takeaways, keywords, questions)

        st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="summary_pack.pdf", mime="application/pdf")
        st.download_button(
            "⬇️ Download Word (.docx)",
            data=docx_bytes,
            file_name="summary_pack.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    if use_tabs:
        t1, t2, t3, t4 = st.tabs(["🧠 Summary", "🔑 Keywords", "❓ Questions", "⬇️ Export"])
        with t1:
            render_summary()
            render_takeaways()
            render_highlighted()
        with t2:
            render_keywords()
        with t3:
            render_questions()
        with t4:
            render_export()
    else:
        render_summary()
        render_takeaways()
        render_keywords()
        render_questions()
        render_highlighted()
        st.subheader("⬇️ Export")
        render_export()

except Exception:
    st.error("Something went wrong. Try a smaller file or reduce pages.")
    st.stop()
