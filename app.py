import re
import math
import html
from io import BytesIO
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")

APP_TITLE = "📄 Document Summarizer"
APP_TAGLINE = "Owner = unlimited tools. Free/Paid = protected limits."

# Safety caps (cloud-friendly)
HARD_MAX_UPLOAD_MB = 300  # your request
HARD_CHAR_LIMIT_OWNER = 400_000  # owner can process more
HARD_CHAR_LIMIT_PAID = 250_000
HARD_CHAR_LIMIT_FREE = 180_000

# Prevent very long processing even if pages are high
HARD_MAX_PAGES_OWNER = 500
HARD_MAX_PAGES_PAID = 200
HARD_MAX_PAGES_FREE = 150

# Session limits (per session)
FREE_MAX_DOCS = 2
PAID_MAX_DOCS = 10
OWNER_MAX_DOCS = 10_000  # effectively unlimited

# -----------------------------
# PIN / Tier
# -----------------------------
def _get_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""

OWNER_PIN = _get_secret("OWNER_PIN")
TRIAL_PIN = _get_secret("TRIAL_PIN")
PAID_PIN = _get_secret("PAID_PIN")

def compute_tier(pin: str) -> str:
    pin = (pin or "").strip()
    if OWNER_PIN and pin == OWNER_PIN:
        return "owner"
    if PAID_PIN and pin == PAID_PIN:
        return "paid"
    if TRIAL_PIN and pin == TRIAL_PIN:
        return "free"
    return "none"

def limits_for_tier(tier: str) -> Dict[str, int]:
    if tier == "owner":
        return {
            "max_docs": OWNER_MAX_DOCS,
            "max_pages": HARD_MAX_PAGES_OWNER,
            "char_limit": HARD_CHAR_LIMIT_OWNER,
            "max_mb": HARD_MAX_UPLOAD_MB,
        }
    if tier == "paid":
        return {
            "max_docs": PAID_MAX_DOCS,
            "max_pages": HARD_MAX_PAGES_PAID,
            "char_limit": HARD_CHAR_LIMIT_PAID,
            "max_mb": HARD_MAX_UPLOAD_MB,
        }
    if tier == "free":
        return {
            "max_docs": FREE_MAX_DOCS,
            "max_pages": HARD_MAX_PAGES_FREE,
            "char_limit": HARD_CHAR_LIMIT_FREE,
            "max_mb": HARD_MAX_UPLOAD_MB,
        }
    return {
        "max_docs": 0,
        "max_pages": 0,
        "char_limit": 0,
        "max_mb": 0,
    }

def auth_gate():
    st.sidebar.subheader("🔐 Access")
    if not (OWNER_PIN or TRIAL_PIN or PAID_PIN):
        st.warning("No PINs configured. Add OWNER_PIN / TRIAL_PIN / PAID_PIN in Streamlit Secrets.")
        st.stop()

    if "tier" not in st.session_state:
        st.session_state.tier = "none"
    if "docs_used" not in st.session_state:
        st.session_state.docs_used = 0

    pin = st.sidebar.text_input("Enter PIN", type="password", placeholder="••••••")
    tier = compute_tier(pin)
    st.session_state.tier = tier

    if tier == "none":
        st.sidebar.error("No access. Enter a valid PIN.")
        st.stop()

    st.sidebar.success(f"Access: {tier.upper()}")

    lim = limits_for_tier(tier)
    st.sidebar.caption(f"Limits this session: {st.session_state.docs_used}/{lim['max_docs']} docs")

    if st.session_state.docs_used >= lim["max_docs"]:
        st.error("Document limit reached for this session. Upgrade / use a paid PIN / owner PIN.")
        st.stop()

    return lim, tier

# -----------------------------
# Text utilities (fast, no NLTK)
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def split_sentences_fast(text: str) -> List[str]:
    """
    Lightweight sentence splitting without NLTK.
    Works well for normal documents. Safe fallback for edge cases.
    """
    text = normalize_text(text)
    if not text:
        return []
    # Replace newlines with spaces for sentence splitting, but keep paragraph breaks for later
    t = re.sub(r"\s*\n\s*", " ", text)
    # Split on ., !, ?, ; followed by space and a capital/number/quote
    parts = re.split(r'(?<=[\.\!\?\;])\s+(?=[A-Z0-9"\'])', t)
    # If it didn't split much (e.g., all caps, weird PDF), fallback to chunking by length
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 3 and len(t) > 1200:
        # fallback: split by approximate length
        chunk = 350
        parts = [t[i:i+chunk].strip() for i in range(0, len(t), chunk)]
    return parts

def tokenize_words(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    words = [w for w in text.split() if len(w) >= 3 and w not in ENGLISH_STOP_WORDS]
    return words

# -----------------------------
# File extraction (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_text_cached(file_bytes: bytes, max_pages: int, char_limit: int) -> Tuple[str, int]:
    """
    Returns (text, pages_processed). Stops early at char_limit for speed & stability.
    """
    reader = PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)
    limit_pages = min(total_pages, max_pages)

    out = []
    collected = 0
    pages_done = 0

    for i in range(limit_pages):
        try:
            page_text = reader.pages[i].extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            out.append(page_text)
            collected += len(page_text)
        pages_done = i + 1

        if collected >= char_limit:
            break

    return normalize_text("\n".join(out)), pages_done

@st.cache_data(show_spinner=False)
def extract_docx_text_cached(file_bytes: bytes, char_limit: int) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    chunks = []
    collected = 0
    for p in doc.paragraphs:
        if p.text.strip():
            chunks.append(p.text)
            collected += len(p.text)
            if collected >= char_limit:
                break
    return normalize_text("\n".join(chunks))

@st.cache_data(show_spinner=False)
def extract_txt_cached(file_bytes: bytes, char_limit: int) -> str:
    # Try utf-8 then fallback
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes.decode(errors="ignore")
    return normalize_text(text[:char_limit])

def looks_scanned_or_empty(text: str) -> bool:
    # If extraction produced almost nothing, it's likely scanned or protected PDF
    return len(text.strip()) < 300

# -----------------------------
# Summarization: TextRank-style (free & better than plain TF-IDF)
# -----------------------------
def textrank_summary(sentences: List[str], top_n: int = 6) -> List[str]:
    if not sentences:
        return []
    if len(sentences) <= top_n:
        return sentences

    # TF-IDF over sentences
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    # Similarity graph
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    # PageRank
    scores = np.ones(len(sentences))
    damping = 0.85
    for _ in range(20):
        prev = scores.copy()
        # Normalize rows to sum=1 to avoid explosion
        row_sums = sim.sum(axis=1)
        norm_sim = np.divide(sim, row_sums[:, None] + 1e-12)
        scores = (1 - damping) + damping * norm_sim.T.dot(prev)
        if np.abs(scores - prev).sum() < 1e-6:
            break

    # Pick top sentences by score, but keep original order in output
    idx = np.argsort(scores)[::-1][:top_n]
    idx_sorted = sorted(idx.tolist())
    return [sentences[i] for i in idx_sorted]

def tfidf_keywords(text: str, k: int = 12) -> List[str]:
    words = tokenize_words(text)
    if not words:
        return []
    # Build pseudo-doc by joining words to keep it simple
    doc = " ".join(words)

    # Unigram + bigram keywords
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform([doc])
    feats = np.array(vec.get_feature_names_out())
    scores = X.toarray().ravel()

    if scores.size == 0:
        return []

    top_idx = np.argsort(scores)[::-1]
    kws = []
    for i in top_idx:
        term = feats[i].strip()
        if not term:
            continue
        # Avoid very common generic words
        if term in ("chapter", "page", "section"):
            continue
        kws.append(term)
        if len(kws) >= k:
            break
    return kws

def key_takeaways_from_summary(summary_sents: List[str], max_items: int = 5) -> List[str]:
    takeaways = []
    for s in summary_sents:
        s = s.strip()
        if len(s) < 30:
            continue
        takeaways.append(s)
        if len(takeaways) >= max_items:
            break
    return takeaways

def generate_questions(summary_sents: List[str], keywords: List[str], n: int = 6) -> List[str]:
    qs = []
    kws = keywords[: max(3, min(8, len(keywords)))]
    # Keyword-based questions
    for kw in kws:
        qs.append(f"What does the document say about **{kw}**?")
        if len(qs) >= n:
            return qs

    # Sentence-based questions
    for s in summary_sents:
        clean = re.sub(r"\s+", " ", s).strip()
        if len(clean) > 20:
            qs.append(f"Explain this idea in your own words: “{clean[:140]}...”")
        if len(qs) >= n:
            break

    return qs[:n]

# -----------------------------
# Highlight keywords inside original text (safe HTML)
# -----------------------------
def highlight_keywords_in_text(text: str, keywords: List[str], max_len: int = 80_000) -> str:
    """
    Returns HTML with <mark> highlights. We cap length to keep browser fast.
    """
    if not text:
        return ""
    # Cap text to avoid rendering huge HTML
    t = text[:max_len]
    escaped = html.escape(t)

    # Build regex for keywords (longer first)
    kws = [k.strip() for k in keywords if k.strip()]
    kws = sorted(kws, key=len, reverse=True)[:30]
    if not kws:
        return f"<div style='white-space: pre-wrap; font-family: system-ui;'>{escaped}</div>"

    # Escape regex special chars
    patterns = [re.escape(k) for k in kws if len(k) >= 3]
    if not patterns:
        return f"<div style='white-space: pre-wrap; font-family: system-ui;'>{escaped}</div>"

    rx = re.compile(r"(" + "|".join(patterns) + r")", flags=re.IGNORECASE)
    highlighted = rx.sub(r"<mark>\1</mark>", escaped)

    return f"""
    <div style="white-space: pre-wrap; line-height: 1.55; font-family: system-ui;">
      {highlighted}
    </div>
    """

# -----------------------------
# Export
# -----------------------------
def build_pdf_bytes(title: str, summary: List[str], takeaways: List[str], keywords: List[str], questions: List[str]) -> bytes:
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    if summary:
        story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        for s in summary:
            story.append(Paragraph(s, styles["BodyText"]))
            story.append(Spacer(1, 0.15 * cm))
        story.append(Spacer(1, 0.3 * cm))

    if takeaways:
        story.append(Paragraph("<b>Key Takeaways</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(ListFlowable([ListItem(Paragraph(t, styles["BodyText"])) for t in takeaways], bulletType="bullet"))
        story.append(Spacer(1, 0.3 * cm))

    if keywords:
        story.append(Paragraph("<b>Keywords</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(", ".join(keywords), styles["BodyText"]))
        story.append(Spacer(1, 0.3 * cm))

    if questions:
        story.append(Paragraph("<b>Questions</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(ListFlowable([ListItem(Paragraph(q, styles["BodyText"])) for q in questions], bulletType="bullet"))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    return buf.getvalue()

def build_docx_bytes(title: str, summary: List[str], takeaways: List[str], keywords: List[str], questions: List[str]) -> bytes:
    doc = DocxDocument()
    doc.add_heading(title, 0)

    if summary:
        doc.add_heading("Summary", level=1)
        for s in summary:
            doc.add_paragraph(s)

    if takeaways:
        doc.add_heading("Key Takeaways", level=1)
        for t in takeaways:
            doc.add_paragraph(t, style="List Bullet")

    if keywords:
        doc.add_heading("Keywords", level=1)
        doc.add_paragraph(", ".join(keywords))

    if questions:
        doc.add_heading("Questions", level=1)
        for q in questions:
            doc.add_paragraph(q, style="List Bullet")

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()

# -----------------------------
# UI
# -----------------------------
def main():
    lim, tier = auth_gate()

    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    # Settings
    st.sidebar.subheader("⚙️ Settings")

    summary_sentences = st.sidebar.slider("Summary length (sentences)", 3, 12, 6)
    keywords_k = st.sidebar.slider("Keywords to highlight", 5, 25, 12)
    questions_n = st.sidebar.slider("Questions to generate", 3, 12, 6)

    quick_mode = st.sidebar.toggle("⚡ Quick mode (faster)", value=True)
    if quick_mode:
        default_pages = 50 if tier == "free" else 80
    else:
        default_pages = min(150, lim["max_pages"])

    pages_to_analyze = st.sidebar.slider(
        "Pages to analyze (PDF)",
        5,
        lim["max_pages"],
        min(default_pages, lim["max_pages"]),
        help="For very large PDFs, analyzing fewer pages keeps the app fast and stable.",
    )

    summary_style = st.sidebar.selectbox("Summary style", ["Clean paragraph", "Bullet points"], index=0)

    st.write("Upload a **PDF, DOCX, or TXT** file to generate a summary, keywords, highlights, questions, and export files.")

    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"],
        help=f"Limit {lim['max_mb']}MB per file",
    )

    if not uploaded:
        st.info("Upload a document to begin.")
        return

    # Validate size
    data = uploaded.getvalue()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > lim["max_mb"]:
        st.error(f"File too large: {size_mb:.1f}MB. Max allowed is {lim['max_mb']}MB.")
        return

    # Only process when button pressed
    if st.button("⚡ Process Document"):
        st.session_state.docs_used += 1

        with st.spinner("Reading and analyzing document..."):
            try:
                ext = uploaded.name.lower().split(".")[-1]
                char_limit = lim["char_limit"]

                # Extract text
                pages_done = None
                if ext == "pdf":
                    raw_text, pages_done = extract_pdf_text_cached(data, pages_to_analyze, char_limit)
                elif ext == "docx":
                    raw_text = extract_docx_text_cached(data, char_limit)
                else:
                    raw_text = extract_txt_cached(data, char_limit)

                if looks_scanned_or_empty(raw_text):
                    st.error(
                        "This file looks like a scanned/locked PDF (no selectable text). "
                        "Try a different PDF (selectable text), or export the text version. "
                        "OCR can be added later."
                    )
                    return

                sentences = split_sentences_fast(raw_text)

                if len(sentences) < 3:
                    st.error("Not enough readable text extracted to summarize. Try another file.")
                    return

                # Summarize + keywords + questions
                summary_sents = textrank_summary(sentences, top_n=summary_sentences)
                takeaways = key_takeaways_from_summary(summary_sents, max_items=min(6, summary_sentences))
                keywords = tfidf_keywords(raw_text, k=keywords_k)
                questions = generate_questions(summary_sents, keywords, n=questions_n)

            except Exception as e:
                st.error("Something went wrong while processing. Try a smaller file or fewer pages.")
                st.caption(f"Debug (safe): {type(e).__name__}")
                return

        # Results UI
        meta_cols = st.columns(4)
        meta_cols[0].metric("File", uploaded.name)
        meta_cols[1].metric("Size (MB)", f"{size_mb:.1f}")
        meta_cols[2].metric("Text chars used", f"{len(raw_text):,}")
        if pages_done is not None:
            meta_cols[3].metric("PDF pages read", str(pages_done))
        else:
            meta_cols[3].metric("Type", ext.upper())

        tab1, tab2, tab3, tab4 = st.tabs(["✅ Summary", "🟡 Highlights", "❓ Questions", "📄 Original text"])

        with tab1:
            st.subheader("Summary")
            if summary_style == "Clean paragraph":
                st.write(" ".join(summary_sents))
            else:
                for s in summary_sents:
                    st.write(f"• {s}")

            st.subheader("Key takeaways")
            for t in takeaways:
                st.write(f"✅ {t}")

            # Export buttons
            pdf_bytes = build_pdf_bytes(uploaded.name, summary_sents, takeaways, keywords, questions)
            docx_bytes = build_docx_bytes(uploaded.name, summary_sents, takeaways, keywords, questions)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Download PDF report",
                    data=pdf_bytes,
                    file_name="summary_report.pdf",
                    mime="application/pdf",
                )
            with c2:
                st.download_button(
                    "⬇️ Download Word report",
                    data=docx_bytes,
                    file_name="summary_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

        with tab2:
            st.subheader("Keywords")
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.info("No strong keywords detected (document may be too short or very repetitive).")

            st.subheader("Highlighted in text")
            st.caption("To keep the app fast, very long documents only show the first part of the text here.")
            html_block = highlight_keywords_in_text(raw_text, keywords)
            st.markdown(html_block, unsafe_allow_html=True)

        with tab3:
            st.subheader("Generated questions")
            for q in questions:
                st.write(f"• {q}")

        with tab4:
            st.subheader("Original extracted text")
            st.text_area("Extracted text", raw_text[:200_000], height=400)
            if len(raw_text) > 200_000:
                st.caption("Showing first 200,000 characters for speed.")

if __name__ == "__main__":
    main()
