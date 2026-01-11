import re
import html
import numpy as np
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# UI polish (lightweight CSS)
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
.kwd {
  background: #fff3bf;
  padding: 0px 4px;
  border-radius: 6px;
}
.pill {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10);
  margin: 4px 6px 0 0;
  font-size: 0.9rem;
}
</style>
"""
st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Text extraction
# -----------------------------
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    out = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            out.append(t)
    return "\n".join(out)

def extract_text_from_docx(file) -> str:
    doc = DocxDocument(file)
    return "\n".join([p.text for p in doc.paragraphs])

def split_into_sentences(text: str):
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
    return sentences


# -----------------------------
# Smarter summarization (MMR)
# Free, fast, cloud-safe
# -----------------------------
def select_summary_sentences_mmr(sentences, k, lambda_param=0.70):
    """
    MMR picks sentences that are BOTH:
    - relevant (importance)
    - diverse (non-repetitive)

    lambda_param:
      - higher => more relevance
      - lower  => more diversity
    """
    if len(sentences) <= k:
        return list(range(len(sentences)))

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)

    # Relevance score: similarity to overall topic centroid
    centroid = X.mean(axis=0)
    rel = cosine_similarity(X, centroid).flatten()

    selected = []
    candidates = list(range(len(sentences)))

    # pick best relevance first
    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)

    # iteratively pick next best MMR
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
    paragraph = " ".join(chosen)
    return paragraph, chosen


# -----------------------------
# Better keywords (single + phrases)
# -----------------------------
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


# -----------------------------
# Better questions (mode-based)
# -----------------------------
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
        else:  # Interview
            qs.append(f"How would you explain this clearly in an interview: “{s}”?")
            qs.append(f"What follow-up question could be asked from: “{s}”?")
            qs.append(f"How would you apply this idea in real life based on: “{s}”?")

        if len(qs) >= k:
            break

    return qs[:k]


# -----------------------------
# Takeaways
# -----------------------------
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


# -----------------------------
# Highlight keywords in original text (safe)
# -----------------------------
def highlight_text_with_keywords(text, keywords):
    safe = html.escape(text)
    kws = sorted(set([k for k in keywords if len(k) >= 3]), key=len, reverse=True)

    for k in kws:
        escaped = html.escape(k)
        pattern = re.compile(rf"(?i)\b({re.escape(escaped)})\b")
        safe = pattern.sub(r'<span class="kwd">\1</span>', safe)

    return safe


# -----------------------------
# APP UI
# -----------------------------
st.title("📄 Document Summarizer")
st.markdown(
    '<div class="small-muted">Smarter than TF-IDF (MMR), still free + fast + cloud-ready. Choose the style & format.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Settings")

    mode = st.selectbox("Style / Mode", ["Student", "Business", "Interview"])
    use_tabs = st.toggle("Organize results into Tabs", value=True)

    st.divider()
    summary_sentences = st.slider("Summary length (sentences)", 3, 12, 6)
    summary_format = st.selectbox("Summary format", ["Clean paragraph", "Slides / bullets"])
    keep_order = st.toggle("Keep original order (better flow)", value=True)

    # MMR tradeoff
    mmr_lambda = st.slider("Summary smartness: Relevance ↔ Variety (MMR)", 0.45, 0.90, 0.70)

    st.divider()
    show_takeaways = st.toggle("Show Key Takeaways", value=True)
    takeaway_count = st.slider("Takeaway bullets", 3, 10, 5)

    st.divider()
    allow_phrases = st.toggle("Keyword phrases (1–2 words)", value=True)
    keyword_count = st.slider("Keywords to highlight", 5, 25, 12)

    st.divider()
    question_count = st.slider("Questions to generate", 3, 15, 8)

    st.divider()
    show_highlighted_text = st.toggle("Show highlighted original text", value=True)
    preview_chars = st.slider("Original text preview (chars)", 2000, 15000, 8000, step=1000)

    st.divider()
    max_chars = st.slider("Max text to process (speed)", 20_000, 180_000, 100_000, step=10_000)


uploaded = st.file_uploader("Upload a PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if not uploaded:
    st.info("Upload a document to begin.")
    st.stop()

try:
    with st.spinner("Reading document..."):
        if uploaded.type == "application/pdf":
            raw = extract_text_from_pdf(uploaded)
        elif uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw = extract_text_from_docx(uploaded)
        else:
            raw = uploaded.read().decode("utf-8", errors="ignore")

    raw = (raw or "").strip()
    if len(raw) < 200:
        st.error("Not enough readable text. If it's a scanned PDF (image-only), it may need OCR.")
        st.stop()

    # speed guard
    raw = raw[:max_chars]

    with st.spinner("Creating summary & insights..."):
        summary_paragraph, summary_sents = build_summary(
            raw,
            k=summary_sentences,
            keep_original_order=keep_order,
            mmr_lambda=float(mmr_lambda)
        )
        keywords = extract_keywords(raw, top_n=keyword_count, allow_phrases=allow_phrases)
        takeaways = build_takeaways(summary_sents, max_points=takeaway_count)
        questions = generate_questions(mode, summary_sents, k=question_count)

    # ---- Render helpers
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

    def render_highlighted_text():
        if not show_highlighted_text:
            return
        st.subheader("Highlighted Original Text (Preview)")
        highlighted = highlight_text_with_keywords(raw[:preview_chars], keywords)
        st.markdown(f'<div class="card">{highlighted}</div>', unsafe_allow_html=True)
        st.caption("Preview only (for speed). Increase preview chars if needed.")

    # ---- Layout
    if use_tabs:
        t1, t2, t3 = st.tabs(["🧠 Summary", "🔑 Keywords", "❓ Questions"])
        with t1:
            render_summary()
            render_takeaways()
            render_highlighted_text()
        with t2:
            render_keywords()
        with t3:
            render_questions()
    else:
        render_summary()
        render_takeaways()
        render_keywords()
        render_questions()
        render_highlighted_text()

except Exception as e:
    st.error("Something went wrong while processing the document.")
    st.exception(e)
