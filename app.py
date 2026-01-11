import re
import math
import html
import numpy as np
import streamlit as st
from io import BytesIO

from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# FAST UTILITIES (NO NLTK)
# -----------------------------

def split_into_sentences(text):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file):
    doc = DocxDocument(file)
    return "\n".join([p.text for p in doc.paragraphs])


def summarize_text(text, max_sentences=5):
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)

    scores = np.array(tfidf.sum(axis=1)).flatten()
    ranked = sorted(
        ((score, sent) for score, sent in zip(scores, sentences)),
        reverse=True
    )

    summary = [sent for _, sent in ranked[:max_sentences]]
    return summary


def highlight_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    tfidf = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()


def generate_questions(sentences, max_q=5):
    questions = []
    for s in sentences[:max_q]:
        q = re.sub(r'\b(is|are|was|were|has|have|had)\b', 'Explain', s, flags=re.I)
        questions.append(q if q.endswith("?") else q + "?")
    return questions


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Summarizer (Fast • Secure • Cloud-Ready)")
st.write(
    "Upload a **PDF, DOCX, or TXT** file to get:\n"
    "- A summary\n"
    "- Highlighted keywords\n"
    "- Auto-generated questions\n\n"
    "⚡ Fast processing • No NLTK • Works on phone"
)

with st.sidebar:
    st.header("Settings")
    summary_len = st.slider("Summary length (sentences)", 3, 10, 5)
    keyword_count = st.slider("Keywords to highlight", 5, 20, 10)
    question_count = st.slider("Questions to generate", 3, 10, 5)

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    try:
        with st.spinner("Reading document..."):
            if uploaded_file.type == "application/pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = extract_text_from_docx(uploaded_file)
            else:
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if len(raw_text.strip()) < 200:
            st.error("Document is too short to summarize.")
            st.stop()

        with st.spinner("Analyzing content..."):
            sentences = split_into_sentences(raw_text)
            summary = summarize_text(raw_text, summary_len)
            keywords = highlight_keywords(raw_text, keyword_count)
            questions = generate_questions(summary, question_count)

        st.subheader("🧠 Summary")
        for s in summary:
            st.write("•", s)

        st.subheader("🔑 Key Terms")
        st.write(", ".join(keywords))

        st.subheader("❓ Practice Questions")
        for q in questions:
            st.write("•", q)

    except Exception as e:
        st.error("Something went wrong while processing the document.")
        st.exception(e)

else:
    st.info("Upload a document to begin.")
