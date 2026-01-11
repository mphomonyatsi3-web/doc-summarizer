import re
import math
import html
import numpy as np
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# PDF export (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


# -----------------------------
# Helpers
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return clean_text("\n".join(pages))

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(paras))

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return clean_text(file_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        return clean_text(file_bytes.decode("latin-1", errors="ignore"))

def split_into_sentences(text: str):
    return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 0]

def score_sentences_tfidf(sentences, max_features=5000):
    if len(sentences) < 3:
        return list(range(len(sentences))), [1.0] * len(sentences)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    ranked_idx = list(np.argsort(-scores))
    return ranked_idx, scores.tolist()

def make_summary(sentences, ranked_idx, ratio: float):
    n = len(sentences)
    keep = max(3, int(math.ceil(n * ratio)))
    keep = min(keep, n)
    chosen = sorted(ranked_idx[:keep])  # keep original order
    summary = " ".join([sentences[i] for i in chosen])
    return summary, chosen

def top_keywords(text: str, k=12):
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    if len(words) < 30:
        uniq = list(dict.fromkeys(words))
        return uniq[: min(k, len(uniq))]

    chunk_size = 80
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vectorizer.fit_transform(chunks)
    tfidf_sum = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())

    top_idx = np.argsort(-tfidf_sum)[:k]
    return vocab[top_idx].tolist()

def key_points_from_summary(summary: str, max_points=10):
    sents = split_into_sentences(summary)
    return sents[: min(max_points, len(sents))]

def generate_questions_offline(summary: str, keywords, n_mcq=5, n_short=5):
    keywords = [k for k in keywords if len(k) > 3]
    if len(keywords) < 6:
        keywords = top_keywords(summary, k=12)
        keywords = [k for k in keywords if len(k) > 3]

    # Short answer questions
    short_q = []
    for kw in keywords[:n_short]:
        short_q.append(f"Explain **{kw}** as described in the document.")

    # MCQ
    mcq = []
    pool = keywords[: max(12, n_mcq * 4)]
    for i in range(min(n_mcq, max(0, len(pool) - 4))):
        answer = pool[i]
        distractors = [d for d in pool if d != answer][:3]
        options = distractors + [answer]
        np.random.shuffle(options)
        mcq.append({
            "question": "Which keyword is most central to the document?",
            "options": options,
            "answer": answer
        })

    return mcq, short_q


# -----------------------------
# Real highlighting (HTML spans)
# -----------------------------
def highlight_text_html(text: str, highlight_sentences, highlight_keywords):
    """
    Returns HTML string where:
    - Important sentences are highlighted in one color
    - Keywords are highlighted in another color
    """
    safe = html.escape(text)

    # Highlight sentences first (longest first helps avoid partial overlaps)
    # We only highlight sentences that are not extremely long.
    sent_list = sorted(
        [s for s in highlight_sentences if 15 <= len(s) <= 350],
        key=len,
        reverse=True
    )

    for s in sent_list:
        s_esc = html.escape(s)
        # Replace exact escaped sentence matches (case sensitive)
        safe = safe.replace(
            s_esc,
            f'<span style="background: #fff3b0; padding: 2px 3px; border-radius: 4px;">{s_esc}</span>'
        )

    # Highlight keywords (case-insensitive word-boundary)
    # Avoid messing up already-inserted HTML by skipping inside tags (simple best-effort).
    for kw in sorted(set(highlight_keywords), key=len, reverse=True):
        if len(kw) < 3:
            continue
        kw_esc = html.escape(kw)

        # regex on the current HTML string can be risky; we do a conservative replace:
        # word boundary replacement not inside tags is hard; we'll do normal word boundaries anyway.
        pattern = re.compile(rf"\b({re.escape(kw_esc)})\b", flags=re.IGNORECASE)
        safe = pattern.sub(
            r'<span style="background: #c7f9cc; padding: 1px 3px; border-radius: 4px;">\1</span>',
            safe
        )

    return f"""
    <div style="white-space: pre-wrap; line-height: 1.6; font-size: 14px;">
      {safe}
    </div>
    """


# -----------------------------
# Export: PDF + DOCX
# -----------------------------
def build_pdf_bytes(title: str, summary: str, key_points, highlights, mcq, short_q):
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(html.escape(summary), styles["BodyText"]))
    story.append(Spacer(1, 10))

    if key_points:
        story.append(Paragraph("Key Points", styles["Heading2"]))
        lst = ListFlowable(
            [ListItem(Paragraph(html.escape(p), styles["BodyText"])) for p in key_points],
            bulletType="bullet"
        )
        story.append(lst)
        story.append(Spacer(1, 10))

    if highlights:
        story.append(Paragraph("Highlights", styles["Heading2"]))
        lst = ListFlowable(
            [ListItem(Paragraph(html.escape(h), styles["BodyText"])) for h in highlights],
            bulletType="bullet"
        )
        story.append(lst)
        story.append(Spacer(1, 10))

    story.append(Paragraph("Questions (MCQ)", styles["Heading2"]))
    for i, q in enumerate(mcq, 1):
        story.append(Paragraph(f"Q{i}. {html.escape(q['question'])}", styles["BodyText"]))
        opts = ListFlowable(
            [ListItem(Paragraph(html.escape(opt), styles["BodyText"])) for opt in q["options"]],
            bulletType="bullet"
        )
        story.append(opts)
        story.append(Paragraph(f"Answer: {html.escape(q['answer'])}", styles["Italic"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Questions (Short Answer)", styles["Heading2"]))
    lst = ListFlowable(
        [ListItem(Paragraph(html.escape(sq), styles["BodyText"])) for sq in short_q],
        bulletType="bullet"
    )
    story.append(lst)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def build_docx_bytes(title: str, summary: str, key_points, highlights, mcq, short_q):
    doc = DocxDocument()
    doc.add_heading(title, level=1)

    doc.add_heading("Summary", level=2)
    doc.add_paragraph(summary)

    if key_points:
        doc.add_heading("Key Points", level=2)
        for p in key_points:
            doc.add_paragraph(p, style="List Bullet")

    if highlights:
        doc.add_heading("Highlights", level=2)
        for h in highlights:
            doc.add_paragraph(h, style="List Bullet")

    doc.add_heading("Questions (MCQ)", level=2)
    for i, q in enumerate(mcq, 1):
        doc.add_paragraph(f"Q{i}. {q['question']}")
        for opt in q["options"]:
            doc.add_paragraph(opt, style="List Bullet")
        doc.add_paragraph(f"Answer: {q['answer']}")

    doc.add_heading("Questions (Short Answer)", level=2)
    for i, sq in enumerate(short_q, 1):
        doc.add_paragraph(f"Q{i}. {sq}", style="List Number")

    out = BytesIO()
    doc.save(out)
    out.seek(0)
    return out.getvalue()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Doc Summarizer", layout="wide")
st.title("📄 Document Summarizer (MVP + Highlights + Export)")
st.write("Upload a document → get summary, highlights, key points, questions, and export to PDF/Word.")

with st.sidebar:
    st.header("Settings")
    summary_len = st.selectbox("Summary length", ["Short", "Medium", "Long"], index=1)
    ratio_map = {"Short": 0.12, "Medium": 0.22, "Long": 0.35}
    ratio = ratio_map[summary_len]

    highlight_mode = st.selectbox(
        "Highlight style",
        ["Important sentences + keywords", "Keywords only", "Sentences only"]
    )
    keyword_count = st.slider("How many keywords to highlight?", 6, 25, 14)
    max_sentence_highlights = st.slider("How many sentence highlights?", 5, 25, 10)

uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded:
    file_bytes = uploaded.read()
    filename = uploaded.name.lower()

    with st.spinner("Extracting text..."):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        else:
            text = extract_text_from_txt(file_bytes)

    if len(text) < 50:
        st.error("Could not extract enough text from this file. Try another document.")
        st.stop()

    sentences = split_into_sentences(text)
    ranked_idx, _scores = score_sentences_tfidf(sentences)

    summary, _chosen_idx = make_summary(sentences, ranked_idx, ratio=ratio)
    keywords = top_keywords(text, k=keyword_count)
    points = key_points_from_summary(summary, max_points=10)

    # Sentence highlights: top-ranked sentences
    highlight_idx = ranked_idx[: min(max_sentence_highlights, len(sentences))]
    sentence_highlights = [sentences[i] for i in sorted(highlight_idx)]

    # Build highlights list (for the tab)
    highlights_list = sentence_highlights

    # Questions
    mcq, short_q = generate_questions_offline(summary, keywords, n_mcq=5, n_short=5)

    # Highlighted HTML view
    if highlight_mode == "Important sentences + keywords":
        html_view = highlight_text_html(text, sentence_highlights, keywords)
    elif highlight_mode == "Keywords only":
        html_view = highlight_text_html(text, [], keywords)
    else:
        html_view = highlight_text_html(text, sentence_highlights, [])

    left, right = st.columns([1, 1])

    with left:
        st.subheader("📌 Document with real highlights")
        st.caption("Yellow = important sentences, Green = keywords (depending on your highlight style).")
        st.markdown(html_view, unsafe_allow_html=True)

    with right:
        st.subheader("Results")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Key Points", "Highlights", "Questions", "Export"])

        with tab1:
            st.markdown("### Summary")
            st.write(summary)

        with tab2:
            st.markdown("### Key Points")
            for i, p in enumerate(points, 1):
                st.write(f"**{i}.** {p}")

        with tab3:
            st.markdown("### Important sentences (highlights)")
            for i, h in enumerate(highlights_list, 1):
                st.write(f"**{i}.** {h}")

        with tab4:
            st.markdown("### Multiple choice questions (MCQ)")
            for i, q in enumerate(mcq, 1):
                st.write(f"**Q{i}.** {q['question']}")
                for opt in q["options"]:
                    st.write(f"- {opt}")
                st.caption(f"Answer: {q['answer']}")
                st.write("")

            st.markdown("### Short-answer questions")
            for i, q in enumerate(short_q, 1):
                st.write(f"**Q{i}.** {q}")

        with tab5:
            st.markdown("### Download your summary pack")

            title = f"Summary Pack - {uploaded.name}"

            pdf_bytes = build_pdf_bytes(
                title=title,
                summary=summary,
                key_points=points,
                highlights=highlights_list,
                mcq=mcq,
                short_q=short_q
            )
            docx_bytes = build_docx_bytes(
                title=title,
                summary=summary,
                key_points=points,
                highlights=highlights_list,
                mcq=mcq,
                short_q=short_q
            )

            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name="summary_pack.pdf",
                mime="application/pdf"
            )

            st.download_button(
                "⬇️ Download Word (.docx)",
                data=docx_bytes,
                file_name="summary_pack.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

else:
    st.info("Upload a document to begin.")
