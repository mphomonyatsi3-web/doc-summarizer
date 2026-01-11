import re
import io
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# App Config
# =========================
st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")
APP_VERSION = "v2.5 (Section-only PDF Processing ✅)"


# =========================
# Limits / Tiers
# =========================
@dataclass
class Tier:
    name: str
    max_docs: int
    max_pages: int
    max_file_mb: int

FREE_TIER = Tier("Free", max_docs=2, max_pages=150, max_file_mb=300)
PAID_TIER = Tier("Paid", max_docs=10, max_pages=200, max_file_mb=300)
OWNER_TIER = Tier("Owner", max_docs=50, max_pages=9999, max_file_mb=300)


# =========================
# Text cleanup
# =========================
def fix_spaced_letters(text: str) -> str:
    if not text:
        return text

    def _join(match):
        return match.group(0).replace(" ", "")

    # Fix "p s y c h o l o g y" -> "psychology"
    text = re.sub(r'(?<!\w)(?:[A-Za-z]\s){2,}[A-Za-z](?!\w)', _join, text)
    # Fix hyphen line breaks: "exam-\nple" -> "example"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    # Collapse spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    s = fix_spaced_letters(s)
    return s


def remove_toc_noise(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.search(r"\.{3,}", s):  # "....."
            continue
        if len(re.findall(r"\d", s)) >= 7 and len(s) <= 90:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘])', text)
    return [p.strip() for p in parts if p.strip()]


def safe_hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    chunk = 1024 * 1024
    for i in range(0, len(b), chunk):
        h.update(b[i:i+chunk])
    return h.hexdigest()


# =========================
# Headings detection (for section-only mode)
# =========================
HEADING_PATTERNS = [
    r'^\s*chapter\s+\d+\b.*$',           # Chapter 1 ...
    r'^\s*\d+(\.\d+)*\s+[A-Z].*$',       # 1.2 Title
    r'^\s*[A-Z][A-Z0-9\s:,-]{6,}$',      # ALL CAPS headings
]

def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 6 or len(s) > 120:
        return False
    for pat in HEADING_PATTERNS:
        if re.match(pat, s, flags=re.IGNORECASE):
            return True
    return False


@st.cache_data(show_spinner=False)
def scan_pdf_headings(file_id: str, file_bytes: bytes, scan_pages: int) -> List[Tuple[str, int]]:
    """
    Fast scan: reads only first scan_pages to find headings.
    Returns list of (heading_title, page_index0).
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    scan_pages = min(scan_pages, total_pages)

    headings: List[Tuple[str, int]] = []
    seen = set()

    for i in range(scan_pages):
        try:
            page_text = reader.pages[i].extract_text() or ""
        except Exception:
            page_text = ""

        page_text = normalize_text(page_text)
        for ln in page_text.splitlines():
            t = ln.strip()
            if not t:
                continue
            if looks_like_heading(t):
                key = re.sub(r"\s+", " ", t).lower()
                if key in seen:
                    continue
                seen.add(key)
                headings.append((t, i))
        # small optimization: if we already found many headings early, stop
        if len(headings) >= 35:
            break

    # If nothing found, fallback "Whole Document"
    if not headings:
        return [("Whole Document (no headings detected)", 0)]

    # Ensure the very start exists as an option
    if headings[0][1] > 0:
        headings.insert(0, ("Start (before first heading)", 0))

    return headings


def compute_heading_ranges(headings: List[Tuple[str, int]], total_pages: int) -> List[Tuple[str, int, int]]:
    """
    Converts (title, start_page0) into (title, start_page1, end_page1)
    end_page1 is inclusive.
    """
    ranges = []
    for idx, (title, start0) in enumerate(headings):
        start1 = start0 + 1
        if idx + 1 < len(headings):
            next_start0 = headings[idx + 1][1]
            end1 = max(start1, next_start0)  # end at page before next heading start
        else:
            end1 = total_pages
        # end1 should be inclusive; if next heading starts at same page, keep end=start
        end1 = max(start1, end1)
        ranges.append((title, start1, end1))
    return ranges


# =========================
# PDF extraction with page-range support (SECTION-ONLY ✅)
# =========================
def extract_pdf_text_range(file_bytes: bytes, start_page1: int, end_page1: int, progress_cb=None) -> Tuple[str, int]:
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)

    start0 = max(0, start_page1 - 1)
    end0 = min(total_pages - 1, end_page1 - 1)
    pages_to_read = max(0, end0 - start0 + 1)

    out = []
    for j, i in enumerate(range(start0, end0 + 1), start=1):
        try:
            page_text = reader.pages[i].extract_text() or ""
        except Exception:
            page_text = ""
        out.append(page_text)
        if progress_cb:
            progress_cb(j, pages_to_read)

    text = "\n".join(out)
    return normalize_text(remove_toc_noise(text)), pages_to_read


def extract_docx_text(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return normalize_text("\n".join(paras))


def extract_txt_text(file_bytes: bytes) -> str:
    try:
        return normalize_text(file_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


# =========================
# Summaries / Keywords
# =========================
def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 60:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
        X = vec.fit_transform([text])
        feats = np.array(vec.get_feature_names_out())
        scores = X.toarray().flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        kws = [feats[i] for i in idx if scores[i] > 0]
        # unique
        seen = set()
        out = []
        for k in kws:
            k2 = k.lower().strip()
            if k2 in seen:
                continue
            seen.add(k2)
            out.append(k.strip())
        return out
    except Exception:
        return []


def summarize_sentences(text: str, n_sent: int = 6) -> List[str]:
    sents = sentence_split(text)
    if not sents:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=9000)
        X = vec.fit_transform(sents)
        scores = np.asarray(X.sum(axis=1)).ravel()
        top_idx = np.argsort(scores)[::-1][:min(n_sent, len(sents))]
        top_idx_sorted = sorted(top_idx)
        return [sents[i] for i in top_idx_sorted]
    except Exception:
        return sents[:n_sent]


def highlight_html(text: str, keywords: List[str]) -> str:
    if not text:
        return ""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    keywords = sorted([k.strip() for k in keywords if k.strip()], key=lambda x: len(x), reverse=True)
    if not keywords:
        return f"<div style='white-space:pre-wrap; line-height:1.6'>{safe}</div>"
    for kw in keywords:
        pat = re.compile(rf"(?i)\b({re.escape(kw)})\b")
        safe = pat.sub(r"<mark>\1</mark>", safe)
    return f"<div style='white-space:pre-wrap; line-height:1.6'>{safe}</div>"


# =========================
# Access / PIN logic
# =========================
def resolve_tier(pin_owner: str, pin_paid: str, pin_trial: str) -> Tuple[Tier, str]:
    secrets = st.secrets if hasattr(st, "secrets") else {}
    OWNER = secrets.get("OWNER_PIN", "")
    PAID = secrets.get("PAID_PIN", "")
    TRIAL = secrets.get("TRIAL_PIN", "")

    if OWNER and pin_owner and pin_owner == OWNER:
        return OWNER_TIER, "Owner"
    if (PAID and pin_paid and pin_paid == PAID) or (TRIAL and pin_trial and pin_trial == TRIAL):
        return PAID_TIER, "Paid"
    return FREE_TIER, "Free"


# =========================
# Session state
# =========================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "access" not in st.session_state:
        st.session_state.access = {"tier": FREE_TIER, "mode": "Free"}

init_state()


# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Settings")
st.sidebar.caption(APP_VERSION)

owner_pin = st.sidebar.text_input("Owner PIN (optional)", type="password")
paid_pin = st.sidebar.text_input("Paid PIN (optional)", type="password")
trial_pin = st.sidebar.text_input("Trial PIN (optional)", type="password")

tier, mode = resolve_tier(owner_pin, paid_pin, trial_pin)
st.session_state.access = {"tier": tier, "mode": mode}

st.sidebar.success(f"Access: {mode} • Max docs: {tier.max_docs} • Max pages: {tier.max_pages} • Max file: {tier.max_file_mb}MB")

summary_sentences = st.sidebar.slider("Summary length (sentences)", 3, 12, 6)
keyword_count = st.sidebar.slider("Keywords to highlight", 5, 30, 12)


# =========================
# Main
# =========================
st.title("📄 Document Summarizer")
st.caption("Fast • No NLTK • Now supports Section-only processing for PDFs ✅")

tabs = st.tabs(["✅ Process", "🕘 History"])

with tabs[0]:
    st.subheader("Upload & Process")
    st.write("Upload PDF / DOCX / TXT.")

    uploaded = st.file_uploader(
        f"Upload up to {tier.max_docs} document(s)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    # SECTION-ONLY MODE UI:
    section_only = False
    scan_pages = 30
    selected_range = None  # (title, start1, end1)

    if uploaded and len(uploaded) == 1:
        f = uploaded[0]
        if f.name.lower().endswith(".pdf") or ("pdf" in (f.type or "").lower()):
            st.markdown("### ⚡ Section-only PDF processing (recommended for big books)")
            section_only = st.checkbox("Enable Section-only processing (scan headings first)", value=True)
            scan_pages = st.slider("How many pages to scan for headings (fast)", 5, 80, 30)

            if section_only:
                file_bytes = f.getvalue()
                size_mb = len(file_bytes) / (1024 * 1024)
                if size_mb > tier.max_file_mb:
                    st.error(f"File too large: {size_mb:.1f}MB. Limit is {tier.max_file_mb}MB.")
                    st.stop()

                file_id = safe_hash_bytes(file_bytes)
                reader = PdfReader(io.BytesIO(file_bytes))
                total_pages = len(reader.pages)

                with st.spinner("Scanning headings…"):
                    headings = scan_pdf_headings(file_id, file_bytes, scan_pages)
                ranges = compute_heading_ranges(headings, total_pages)

                labels = [f"{t}  (pages {s}-{e})" for (t, s, e) in ranges]
                pick = st.selectbox("Choose the section to process", labels)
                selected_range = ranges[labels.index(pick)]

                st.info("✅ This will extract ONLY the selected page range, not the full PDF.")

    if uploaded:
        if len(uploaded) > tier.max_docs:
            st.error(f"Too many files. Your limit is {tier.max_docs}.")
            st.stop()

        if st.button("⚡ Process Now"):
            results = []
            for f in uploaded:
                file_bytes = f.getvalue()
                size_mb = len(file_bytes) / (1024 * 1024)

                if size_mb > tier.max_file_mb:
                    st.error(f"{f.name}: File too large ({size_mb:.1f}MB). Limit {tier.max_file_mb}MB.")
                    continue

                st.info(f"Processing: {f.name}")

                progress = st.progress(0, text="Preparing...")
                status = st.empty()

                def prog(done, total):
                    pct = int((done / max(total, 1)) * 100)
                    progress.progress(pct, text=f"Extracting… {done}/{total}")
                    status.info(f"Reading page {done} of {total}")

                try:
                    name = f.name.lower()
                    mime = (f.type or "").lower()
                    file_id = safe_hash_bytes(file_bytes)

                    # PDF section-only extraction
                    if (name.endswith(".pdf") or "pdf" in mime) and section_only and selected_range is not None:
                        title, start1, end1 = selected_range
                        # enforce tier page limit
                        span = end1 - start1 + 1
                        if span > tier.max_pages:
                            end1 = start1 + tier.max_pages - 1
                            st.warning(f"Section was long. Limited to {tier.max_pages} pages: {start1}-{end1}")

                        status.info(f"Extracting selected section pages {start1}-{end1}…")
                        text, pages_read = extract_pdf_text_range(file_bytes, start1, end1, progress_cb=prog)
                        doc_meta = {"id": file_id, "name": f.name, "type": "pdf", "pages": pages_read, "size_mb": size_mb, "section": title, "range": (start1, end1)}

                    # Normal PDF (full up to max_pages)
                    elif name.endswith(".pdf") or "pdf" in mime:
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)
                        pages_to_read = min(total_pages, tier.max_pages)

                        out = []
                        for i in range(pages_to_read):
                            try:
                                page_text = reader.pages[i].extract_text() or ""
                            except Exception:
                                page_text = ""
                            out.append(page_text)
                            prog(i + 1, pages_to_read)

                        text = normalize_text(remove_toc_noise("\n".join(out)))
                        pages_read = pages_to_read
                        doc_meta = {"id": file_id, "name": f.name, "type": "pdf", "pages": pages_read, "size_mb": size_mb}

                    # DOCX
                    elif name.endswith(".docx") or "word" in mime:
                        progress.progress(30, text="Reading DOCX…")
                        text = extract_docx_text(file_bytes)
                        progress.progress(100, text="Done.")
                        doc_meta = {"id": file_id, "name": f.name, "type": "docx", "pages": None, "size_mb": size_mb}

                    # TXT
                    elif name.endswith(".txt") or "text" in mime:
                        progress.progress(30, text="Reading TXT…")
                        text = extract_txt_text(file_bytes)
                        progress.progress(100, text="Done.")
                        doc_meta = {"id": file_id, "name": f.name, "type": "txt", "pages": None, "size_mb": size_mb}

                    else:
                        raise ValueError("Unsupported file type.")

                    progress.progress(100, text="Analyzing…")
                    if not text or len(text) < 60:
                        raise ValueError("No readable text found. This may be a scanned PDF (image-based).")

                    kws = extract_keywords(text, top_k=keyword_count)
                    summary = summarize_sentences(text, n_sent=summary_sentences)

                    result = {
                        "doc": doc_meta,
                        "text": text,
                        "keywords": kws,
                        "summary": summary,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    results.append(result)

                    status.success("Done ✅")

                except Exception as e:
                    st.error(f"Failed on {f.name}: {e}")

            if results:
                st.session_state.last_result = results
                for r in results:
                    st.session_state.history.insert(0, {
                        "name": r["doc"]["name"],
                        "created_at": r["created_at"],
                        "summary": r["summary"],
                        "keywords": r["keywords"],
                        "size_mb": r["doc"]["size_mb"],
                        "pages": r["doc"]["pages"],
                        "section": r["doc"].get("section"),
                        "range": r["doc"].get("range"),
                    })

    # Show last results
    last = st.session_state.last_result
    if last:
        st.divider()
        st.subheader("Results")
        for r in last:
            doc = r["doc"]
            st.markdown(f"### {doc['name']}")
            extra = ""
            if doc.get("section"):
                extra = f" • Section: **{doc['section']}** • Pages: {doc.get('range')[0]}-{doc.get('range')[1]}"
            st.caption(f"Type: {doc['type']} • Size: {doc['size_mb']:.1f}MB • Pages read: {doc['pages'] if doc['pages'] else 'N/A'}{extra}")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("#### Summary")
                st.write("\n".join([f"- {s}" for s in r["summary"]]) if r["summary"] else "No summary.")
                st.markdown("#### Keywords")
                st.write(", ".join(r["keywords"]) if r["keywords"] else "No keywords.")

            with c2:
                st.markdown("#### Highlights (inside original text)")
                st.markdown(highlight_html(r["text"][:25000], r["keywords"]), unsafe_allow_html=True)
                st.caption("Showing first ~25k characters for speed.")


with tabs[1]:
    st.subheader("History (this session)")
    hist = st.session_state.history
    if not hist:
        st.info("No history yet.")
    else:
        for h in hist[:15]:
            st.markdown(f"### {h['name']}")
            sec = ""
            if h.get("section"):
                sec = f" • Section: {h['section']} • Range: {h['range'][0]}-{h['range'][1]}"
            st.caption(f"{h['created_at']} • Size: {h['size_mb']:.1f}MB • Pages: {h['pages'] if h['pages'] else 'N/A'}{sec}")
            st.write("\n".join([f"- {s}" for s in h["summary"]]) if h["summary"] else "No summary.")
            st.write("Keywords: " + (", ".join(h["keywords"]) if h["keywords"] else "None"))
            st.divider()

        export = {"exported_at": time.strftime("%Y-%m-%d %H:%M:%S"), "history": hist}
        st.download_button(
            "⬇️ Download History (JSON)",
            data=json.dumps(export, indent=2),
            file_name="doc_summarizer_history.json",
            mime="application/json"
        )

st.caption("Note: Scanned PDFs (images) need OCR to extract text. We can add OCR later.")
