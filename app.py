import re
import io
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# App Config
# =========================
APP_NAME = "DocSum"
APP_VERSION = "v3.0 (MASTER: All features + TOC + Manual Range + PIN Gate + User Type ✅)"
st.set_page_config(page_title=APP_NAME, page_icon="📄", layout="wide")


# =========================
# Limits / Tiers
# =========================
@dataclass
class Tier:
    name: str
    max_docs: int
    max_pages: int
    max_file_mb: int

FREE_TIER  = Tier("Free",  max_docs=2,  max_pages=150, max_file_mb=300)
PAID_TIER  = Tier("Paid",  max_docs=10, max_pages=200, max_file_mb=300)
OWNER_TIER = Tier("Owner", max_docs=50, max_pages=9999, max_file_mb=300)


# =========================
# Utilities
# =========================
def safe_hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    chunk = 1024 * 1024
    for i in range(0, len(b), chunk):
        h.update(b[i:i+chunk])
    return h.hexdigest()


def fix_spaced_letters(text: str) -> str:
    """Fix 'p s y c h o l o g y' -> 'psychology' and hyphen line breaks."""
    if not text:
        return text

    def _join(match):
        return match.group(0).replace(" ", "")

    text = re.sub(r'(?<!\w)(?:[A-Za-z]\s){2,}[A-Za-z](?!\w)', _join, text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
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
    """Remove table-of-contents-like dotted lines and heavy page-number lines."""
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.search(r"\.{3,}", s):
            continue
        if len(re.findall(r"\d", s)) >= 7 and len(s) <= 90:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def sentence_split(text: str) -> List[str]:
    """Fast sentence splitter (no NLTK)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘])', text)
    return [p.strip() for p in parts if p.strip()]


# =========================
# PIN / Access
# =========================
def get_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""


REQUIRE_PIN = get_secret("REQUIRE_PIN").lower() in ("1", "true", "yes", "y")
OWNER_PIN = get_secret("OWNER_PIN")
PAID_PIN  = get_secret("PAID_PIN")
FREE_PIN  = get_secret("FREE_PIN")


def tier_from_pin(pin: str) -> Tuple[Tier, str]:
    pin = (pin or "").strip()
    if OWNER_PIN and pin == OWNER_PIN:
        return OWNER_TIER, "Owner"
    if PAID_PIN and pin == PAID_PIN:
        return PAID_TIER, "Paid"
    if FREE_PIN and pin == FREE_PIN:
        return FREE_TIER, "Free"
    return Tier("Locked", 0, 0, 0), "Locked"


# =========================
# Heading detection
# (1) TOC / bookmarks
# (2) scan first X pages fallback
# =========================
HEADING_PATTERNS = [
    r'^\s*chapter\s+\d+\b.*$',
    r'^\s*\d+(\.\d+)*\s+[A-Z].*$',
    r'^\s*[A-Z][A-Z0-9\s:,-]{6,}$',
]

def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 6 or len(s) > 120:
        return False
    for pat in HEADING_PATTERNS:
        if re.match(pat, s, flags=re.IGNORECASE):
            return True
    return False


def _flatten_outline(outline: Any) -> List[Any]:
    items = []
    if outline is None:
        return items
    if isinstance(outline, list):
        for x in outline:
            items.extend(_flatten_outline(x))
    else:
        items.append(outline)
    return items


def _outline_title(obj: Any) -> Optional[str]:
    for attr in ("title", "Title"):
        if hasattr(obj, attr):
            t = getattr(obj, attr)
            if isinstance(t, str) and t.strip():
                return t.strip()
    if isinstance(obj, dict):
        t = obj.get("/Title") or obj.get("Title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


@st.cache_data(show_spinner=False)
def toc_headings(file_id: str, file_bytes: bytes) -> List[Tuple[str, int]]:
    """Try to read PDF TOC/bookmarks. Return list of (title, page0)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        outline = getattr(reader, "outline", None)
        if outline is None:
            return []
        flat = _flatten_outline(outline)

        found: List[Tuple[str, int]] = []
        seen = set()

        for item in flat:
            try:
                title = _outline_title(item)
                if not title:
                    continue
                page0 = reader.get_destination_page_number(item)
                if page0 is None:
                    continue
                key = re.sub(r"\s+", " ", title).lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append((title, int(page0)))
            except Exception:
                continue

        found.sort(key=lambda x: x[1])
        if found and found[0][1] > 0:
            found.insert(0, ("Start (before first TOC item)", 0))
        return found
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def scan_pdf_headings(file_id: str, file_bytes: bytes, scan_pages: int) -> List[Tuple[str, int]]:
    """Scan first scan_pages pages for headings."""
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
        if len(headings) >= 40:
            break

    if not headings:
        return [("Whole Document (no headings detected)", 0)]
    if headings[0][1] > 0:
        headings.insert(0, ("Start (before first heading)", 0))
    return headings


def compute_heading_ranges(headings: List[Tuple[str, int]], total_pages: int) -> List[Tuple[str, int, int]]:
    """(title, start0) -> (title, start1, end1 inclusive)"""
    ranges = []
    for idx, (title, start0) in enumerate(headings):
        start1 = start0 + 1
        if idx + 1 < len(headings):
            next_start0 = headings[idx + 1][1]
            end1 = max(start1, next_start0)
        else:
            end1 = total_pages
        end1 = max(start1, end1)
        ranges.append((title, start1, end1))
    return ranges


# =========================
# Extraction (with progress)
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


def extract_text_any(uploaded_file, tier: Tier) -> Dict:
    """Normal extract for non-PDF or multi-upload, with progress for PDF."""
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > tier.max_file_mb:
        raise ValueError(f"File too large: {size_mb:.1f}MB. Limit {tier.max_file_mb}MB.")

    name = uploaded_file.name.lower()
    mime = (uploaded_file.type or "").lower()
    file_id = safe_hash_bytes(file_bytes)

    progress = st.progress(0, text="Preparing…")
    status = st.empty()

    def prog(done, total):
        pct = int((done / max(total, 1)) * 100)
        progress.progress(pct, text=f"Extracting pages… {done}/{total}")
        status.info(f"Reading page {done} of {total}")

    if name.endswith(".pdf") or "pdf" in mime:
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
        progress.progress(100, text="Done extracting.")
        status.success(f"Extracted {pages_to_read} page(s).")
        return {"id": file_id, "name": uploaded_file.name, "type": "pdf", "text": text, "pages": pages_to_read, "size_mb": size_mb}

    if name.endswith(".docx") or "word" in mime:
        progress.progress(40, text="Reading DOCX…")
        text = extract_docx_text(file_bytes)
        progress.progress(100, text="Done extracting.")
        status.success("DOCX extracted.")
        return {"id": file_id, "name": uploaded_file.name, "type": "docx", "text": text, "pages": None, "size_mb": size_mb}

    if name.endswith(".txt") or "text" in mime:
        progress.progress(40, text="Reading TXT…")
        text = extract_txt_text(file_bytes)
        progress.progress(100, text="Done extracting.")
        status.success("TXT extracted.")
        return {"id": file_id, "name": uploaded_file.name, "type": "txt", "text": text, "pages": None, "size_mb": size_mb}

    raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")


# =========================
# Analysis / Output
# =========================
def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 60:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=7000)
        X = vec.fit_transform([text])
        feats = np.array(vec.get_feature_names_out())
        scores = X.toarray().flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        kws = [feats[i] for i in idx if scores[i] > 0]
        out, seen = [], set()
        for k in kws:
            k2 = k.lower().strip()
            if k2 in seen:
                continue
            seen.add(k2)
            out.append(k.strip())
        return out
    except Exception:
        return []


def summarize_sentences(text: str, n_sent: int, user_type: str) -> List[str]:
    sents = sentence_split(text)
    if not sents:
        return []

    # user-type tuning (keeps app fast but improves relevance)
    if user_type == "Primary learner":
        n_sent = max(3, min(n_sent, 5))
    elif user_type == "High school learner":
        n_sent = max(3, min(n_sent, 6))
    elif user_type == "Varsity/College":
        n_sent = max(5, min(n_sent, 10))
    elif user_type == "Professional":
        n_sent = max(5, min(n_sent, 10))
    elif user_type == "Trader":
        n_sent = max(5, min(n_sent, 10))

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=9000)
        X = vec.fit_transform(sents)
        scores = np.asarray(X.sum(axis=1)).ravel()
        top_idx = np.argsort(scores)[::-1][:min(n_sent, len(sents))]
        top_idx_sorted = sorted(top_idx)  # keep order
        picked = [re.sub(r"\s+", " ", sents[i].strip()) for i in top_idx_sorted]
        return picked
    except Exception:
        return sents[:n_sent]


def highlight_html(text: str, keywords: List[str]) -> str:
    if not text:
        return ""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    keywords = sorted([k.strip() for k in keywords if k.strip()], key=lambda x: len(x), reverse=True)
    for kw in keywords[:25]:
        pat = re.compile(rf"(?i)\b({re.escape(kw)})\b")
        safe = pat.sub(r"<mark>\1</mark>", safe)
    return f"<div style='white-space:pre-wrap; line-height:1.65'>{safe}</div>"


# =========================
# Glossary (user searches a term)
# =========================
DEF_PATTERNS = [
    r"\b{w}\b\s+(is|are|means|refers to|defined as)\s+(.+?)([.;]|$)",
    r"(.+?)\s+(is|are)\s+called\s+\b{w}\b([.;]|$)",
]

def glossary_define(word: str, text: str, max_results: int = 6) -> List[str]:
    if not word or not text:
        return []
    w = re.escape(word.strip())
    sents = sentence_split(text)
    hits = []

    for sent in sents:
        s = sent.strip()
        if len(s) < 15:
            continue
        if re.search(rf"(?i)\b{w}\b", s) is None:
            continue
        for pat in DEF_PATTERNS:
            m = re.search(pat.format(w=w), s, flags=re.IGNORECASE)
            if m:
                hits.append(s)
                break
        if len(hits) >= max_results:
            break

    if hits:
        return hits[:max_results]

    # fallback context sentences
    ctx = []
    for sent in sents:
        if re.search(rf"(?i)\b{w}\b", sent):
            ctx.append(sent.strip())
        if len(ctx) >= max_results:
            break
    return ctx


# =========================
# Flashcards
# =========================
def make_flashcards(text: str, keywords: List[str], n_cards: int = 10) -> List[Dict]:
    sents = sentence_split(text)
    if not sents:
        return []
    cand = [s for s in sents if len(s) >= 70] or sents

    pool = [k for k in keywords if 3 <= len(k) <= 35]
    pool = list(dict.fromkeys(pool))
    if len(pool) < 4:
        pool += extract_keywords(text, top_k=15)
        pool = list(dict.fromkeys(pool))

    rng = np.random.default_rng(7)
    rng.shuffle(cand)

    cards = []
    used = set()

    for sent in cand:
        if len(cards) >= n_cards:
            break
        chosen = None
        for k in pool:
            if re.search(rf"(?i)\b{re.escape(k)}\b", sent):
                chosen = k
                break
        if not chosen:
            continue

        key = (sent[:70] + chosen).lower()
        if key in used:
            continue
        used.add(key)

        blanked = re.sub(rf"(?i)\b{re.escape(chosen)}\b", "_____", sent, count=1)

        if len(cards) % 2 == 0:
            cards.append({"type": "typed", "question": f"Fill in the missing term:\n\n{blanked}", "answer": chosen.strip()})
        else:
            distractors = [x for x in pool if x.lower() != chosen.lower()]
            rng.shuffle(distractors)
            options = [chosen] + distractors[:3]
            rng.shuffle(options)
            cards.append({"type": "mcq", "question": f"Which term best completes the sentence?\n\n{blanked}", "answer": chosen.strip(), "options": options})

    return cards[:n_cards]


# =========================
# Session State
# =========================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "auth" not in st.session_state:
        st.session_state.auth = {"ok": False, "tier": FREE_TIER, "mode": "Free"}

init_state()


# =========================
# Sidebar: Access + User Type
# =========================
st.sidebar.title("🔐 Access")
st.sidebar.caption(APP_VERSION)

# If REQUIRE_PIN is true, enforce gate. If false, allow free without pin.
pin = st.sidebar.text_input("Enter PIN (if you have one)", type="password")
if st.sidebar.button("Apply PIN"):
    t, m = tier_from_pin(pin)
    if m == "Locked":
        st.sidebar.error("Wrong PIN.")
        if REQUIRE_PIN:
            st.session_state.auth = {"ok": False, "tier": Tier("Locked", 0, 0, 0), "mode": "Locked"}
        else:
            st.session_state.auth = {"ok": True, "tier": FREE_TIER, "mode": "Free"}
    else:
        st.session_state.auth = {"ok": True, "tier": t, "mode": m}
        st.sidebar.success(f"Unlocked: {m}")

# Default auth state on first load
if not st.session_state.auth.get("ok", False):
    if REQUIRE_PIN:
        st.session_state.auth = {"ok": False, "tier": Tier("Locked", 0, 0, 0), "mode": "Locked"}
    else:
        st.session_state.auth = {"ok": True, "tier": FREE_TIER, "mode": "Free"}

auth = st.session_state.auth

if REQUIRE_PIN and not auth["ok"]:
    st.title(f"📄 {APP_NAME}")
    st.error("🔒 Locked. Enter a valid PIN in the sidebar to use the app.")
    st.stop()

tier = auth["tier"]
mode = auth["mode"]

st.sidebar.success(f"Mode: {mode} • Max docs: {tier.max_docs} • Max pages: {tier.max_pages} • Max file: {tier.max_file_mb}MB")

st.sidebar.divider()
st.sidebar.title("👤 User Type")
user_type = st.sidebar.selectbox(
    "Choose your category",
    ["Primary learner", "High school learner", "Varsity/College", "Professional", "Trader"],
)

summary_sentences = st.sidebar.slider("Summary length (sentences)", 3, 12, 6)
keyword_count = st.sidebar.slider("Keywords to highlight", 5, 30, 12)
flashcard_count = st.sidebar.slider("Flashcards to generate", 5, 25, 10)

st.sidebar.caption("Tip: For big books, use Auto section or Manual page range.")


# =========================
# Main UI (ALL tabs preserved ✅)
# =========================
st.title(f"📄 {APP_NAME}")
st.caption(f"Access: {mode} • User Type: {user_type}")

tabs = st.tabs(["✅ Process", "🧩 Sections", "📘 Glossary", "🃏 Flashcards", "🆚 Compare", "🕘 History"])

# -------------------------
# Process Tab
# -------------------------
with tabs[0]:
    st.subheader("Upload & Process")
    st.write("Upload PDF / DOCX / TXT and generate summary, keywords, highlights.")

    uploaded = st.file_uploader(
        f"Upload up to {tier.max_docs} document(s)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    # For 1 PDF: allow TOC/Scan + Manual Range
    section_mode = None
    scan_pages = 30
    selected_range = None
    manual_start, manual_end = 1, 1

    if uploaded and len(uploaded) == 1:
        f = uploaded[0]
        name = f.name.lower()
        mime = (f.type or "").lower()

        if name.endswith(".pdf") or ("pdf" in mime):
            file_bytes = f.getvalue()
            size_mb = len(file_bytes) / (1024 * 1024)

            if size_mb > tier.max_file_mb:
                st.error(f"File too large: {size_mb:.1f}MB. Limit {tier.max_file_mb}MB.")
                st.stop()

            reader = PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)

            st.markdown("### ⚡ PDF: Select Pages (TOC / Scan / Manual)")
            section_mode = st.radio(
                "Choose a mode",
                ["Auto section (TOC → Scan headings)", "Manual page range", "Full (limited by tier)"],
                horizontal=True
            )

            if section_mode == "Auto section (TOC → Scan headings)":
                scan_pages = st.slider("If TOC fails, scan first pages for headings", 5, 120, 30)
                file_id = safe_hash_bytes(file_bytes)

                with st.spinner("Trying TOC/bookmarks…"):
                    toc = toc_headings(file_id, file_bytes)

                if toc:
                    st.success("✅ TOC found.")
                    headings = toc
                else:
                    st.warning("No TOC found. Scanning headings…")
                    headings = scan_pdf_headings(file_id, file_bytes, scan_pages)

                ranges = compute_heading_ranges(headings, total_pages)
                labels = [f"{t}  (pages {s}-{e})" for (t, s, e) in ranges]
                pick = st.selectbox("Choose section", labels)
                selected_range = ranges[labels.index(pick)]
                st.info("This will extract ONLY the selected range.")

            elif section_mode == "Manual page range":
                c1, c2 = st.columns(2)
                with c1:
                    manual_start = st.number_input("Start page", min_value=1, max_value=total_pages, value=1)
                with c2:
                    manual_end = st.number_input("End page", min_value=1, max_value=total_pages, value=min(total_pages, tier.max_pages))
                st.info("This will extract ONLY the pages you selected.")

            else:
                st.info("This will extract from page 1 up to your tier max pages.")

    if uploaded:
        if len(uploaded) > tier.max_docs:
            st.error(f"Too many files. Your limit is {tier.max_docs}.")
            st.stop()

        if st.button("⚡ Process Now"):
            results = []

            for f in uploaded:
                st.info(f"Processing: {f.name}")
                try:
                    name = f.name.lower()
                    mime = (f.type or "").lower()

                    # 1 PDF with page selection
                    if len(uploaded) == 1 and (name.endswith(".pdf") or ("pdf" in mime)) and section_mode is not None:
                        file_bytes = f.getvalue()
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)

                        start1, end1 = 1, min(total_pages, tier.max_pages)
                        section_label = None

                        if section_mode == "Auto section (TOC → Scan headings)" and selected_range is not None:
                            section_label, start1, end1 = selected_range

                        elif section_mode == "Manual page range":
                            section_label = "Manual Range"
                            start1, end1 = int(manual_start), int(manual_end)

                        # normalize
                        if start1 > end1:
                            start1, end1 = end1, start1

                        span = end1 - start1 + 1
                        if span > tier.max_pages:
                            end1 = start1 + tier.max_pages - 1
                            st.warning(f"Limited to {tier.max_pages} pages: {start1}-{end1}")

                        progress = st.progress(0, text="Preparing…")
                        status = st.empty()

                        def prog(done, total):
                            pct = int((done / max(total, 1)) * 100)
                            progress.progress(pct, text=f"Extracting pages… {done}/{total}")
                            status.info(f"Reading page {done} of {total}")

                        text, pages_read = extract_pdf_text_range(file_bytes, start1, end1, progress_cb=prog)
                        doc = {
                            "id": safe_hash_bytes(file_bytes),
                            "name": f.name,
                            "type": "pdf",
                            "text": text,
                            "pages": pages_read,
                            "size_mb": len(file_bytes) / (1024 * 1024),
                            "range": (start1, end1),
                            "section": section_label,
                        }
                    else:
                        # normal extraction (multi-files or non-pdf)
                        doc = extract_text_any(f, tier=tier)

                    text = doc["text"]
                    if not text or len(text) < 60:
                        raise ValueError("No readable text found. This may be a scanned PDF (image-based).")

                    kws = extract_keywords(text, top_k=keyword_count)
                    summary = summarize_sentences(text, n_sent=summary_sentences, user_type=user_type)

                    sections = split_into_sections(text)

                    doc_result = {
                        "doc": doc,
                        "text": text,
                        "summary": summary,
                        "keywords": kws,
                        "sections": sections,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "user_type": user_type,
                    }
                    results.append(doc_result)

                except Exception as e:
                    st.error(f"Failed on {f.name}: {e}")

            if results:
                st.session_state.last_result = results
                for r in results:
                    d = r["doc"]
                    st.session_state.history.insert(0, {
                        "name": d["name"],
                        "created_at": r["created_at"],
                        "summary": r["summary"],
                        "keywords": r["keywords"],
                        "size_mb": d.get("size_mb"),
                        "pages": d.get("pages"),
                        "range": d.get("range"),
                        "section": d.get("section"),
                        "user_type": r["user_type"],
                    })
                st.success("Done ✅")

    # Show last results
    last = st.session_state.last_result
    if last:
        st.divider()
        st.subheader("Results")
        for r in last:
            d = r["doc"]
            extra = ""
            if d.get("range"):
                extra += f" • Pages: {d['range'][0]}-{d['range'][1]}"
            if d.get("section"):
                extra += f" • Section: {d['section']}"
            st.markdown(f"### {d['name']}")
            st.caption(f"Type: {d['type']} • Size: {d.get('size_mb', 0):.1f}MB • Pages read: {d.get('pages','N/A')}{extra}")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Summary")
                st.write("\n".join([f"- {s}" for s in r["summary"]]) if r["summary"] else "No summary.")
                st.markdown("#### Keywords")
                st.write(", ".join(r["keywords"]) if r["keywords"] else "No keywords.")

            with col2:
                st.markdown("#### Highlights (inside original text)")
                st.markdown(highlight_html(r["text"][:25000], r["keywords"]), unsafe_allow_html=True)
                st.caption("Showing first ~25k characters for speed.")


# -------------------------
# Sections Tab
# -------------------------
def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Line-based section splitter (headings/chapters)."""
    lines = text.splitlines()
    sections = []
    current_title = "Start"
    buf = []
    for ln in lines:
        if looks_like_heading(ln):
            chunk = "\n".join(buf).strip()
            if chunk:
                sections.append((current_title, chunk))
            current_title = ln.strip()
            buf = []
        else:
            buf.append(ln)
    final = "\n".join(buf).strip()
    if final:
        sections.append((current_title, final))
    if len(sections) <= 1:
        return [("Whole Document", text.strip())]
    return sections


with tabs[1]:
    st.subheader("Section Splitter (choose a section to summarize)")
    last = st.session_state.last_result
    if not last:
        st.info("Process a document first in the ✅ Process tab.")
    else:
        doc_names = [r["doc"]["name"] for r in last]
        pick_doc = st.selectbox("Choose document", doc_names, key="sec_doc")
        r = next(x for x in last if x["doc"]["name"] == pick_doc)
        sections = r["sections"]

        titles = [f"{i+1}. {t}" for i, (t, _) in enumerate(sections)]
        pick_title = st.selectbox("Choose section", titles, key="sec_pick")
        idx = titles.index(pick_title)
        sec_title, sec_text = sections[idx]

        sec_kws = extract_keywords(sec_text, top_k=keyword_count)
        sec_summary = summarize_sentences(sec_text, n_sent=summary_sentences, user_type=user_type)

        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"### {sec_title}")
            st.markdown("#### Section Summary")
            st.write("\n".join([f"- {s}" for s in sec_summary]) if sec_summary else "No summary.")
            st.markdown("#### Section Keywords")
            st.write(", ".join(sec_kws) if sec_kws else "No keywords.")
        with right:
            st.markdown("#### Highlighted Section Text")
            st.markdown(highlight_html(sec_text[:50000], sec_kws), unsafe_allow_html=True)
            st.caption("Showing first ~50k characters for speed.")


# -------------------------
# Glossary Tab
# -------------------------
with tabs[2]:
    st.subheader("Glossary / Definitions (type a word and search)")
    last = st.session_state.last_result
    if not last:
        st.info("Process a document first in the ✅ Process tab.")
    else:
        doc_names = [r["doc"]["name"] for r in last]
        pick_doc = st.selectbox("Choose document for glossary", doc_names, key="gloss_doc")
        r = next(x for x in last if x["doc"]["name"] == pick_doc)

        query = st.text_input("Enter a word/term (e.g., 'cognition', 'liquidity sweep')", key="gloss_q")
        if st.button("🔎 Search", key="gloss_btn"):
            matches = glossary_define(query.strip(), r["text"], max_results=6)
            if query.strip() and matches:
                st.success(f"Found {len(matches)} relevant sentence(s):")
                for m in matches:
                    st.write(f"- {m}")
            elif not query.strip():
                st.warning("Type a word first.")
            else:
                st.warning("No definition found in this document. Try another term.")


# -------------------------
# Flashcards Tab
# -------------------------
with tabs[3]:
    st.subheader("Flashcards (MCQ + typed, auto-check)")
    last = st.session_state.last_result
    if not last:
        st.info("Process a document first in the ✅ Process tab.")
    else:
        doc_names = [r["doc"]["name"] for r in last]
        pick_doc = st.selectbox("Choose document for flashcards", doc_names, key="fc_doc")
        r = next(x for x in last if x["doc"]["name"] == pick_doc)

        if st.button("🃏 Generate Flashcards", key="fc_gen"):
            cards = make_flashcards(r["text"], keywords=r["keywords"], n_cards=flashcard_count)
            st.session_state["cards"] = cards
            st.session_state["card_answers"] = {}
            st.success(f"Generated {len(cards)} flashcards!")

        cards = st.session_state.get("cards", [])
        if cards:
            for i, c in enumerate(cards):
                st.markdown(f"### Card {i+1}")
                st.write(c["question"])

                if c["type"] == "mcq":
                    choice = st.radio("Choose an answer", c["options"], key=f"mcq_{i}")
                    st.session_state["card_answers"][i] = choice
                else:
                    typed = st.text_input("Type your answer", key=f"typed_{i}")
                    st.session_state["card_answers"][i] = typed

                st.divider()

            if st.button("✅ Check answers", key="fc_check"):
                score = 0
                for i, c in enumerate(cards):
                    user_ans = (st.session_state["card_answers"].get(i) or "").strip()
                    correct = c["answer"].strip()
                    if c["type"] == "mcq":
                        ok = user_ans.lower() == correct.lower()
                    else:
                        ok = correct.lower() in user_ans.lower() if len(correct) > 4 else user_ans.lower() == correct.lower()

                    if ok:
                        score += 1
                        st.success(f"Card {i+1}: Correct ✅")
                    else:
                        st.error(f"Card {i+1}: Wrong ❌ | Correct: {correct}")
                st.info(f"Score: {score}/{len(cards)}")


# -------------------------
# Compare Tab
# -------------------------
with tabs[4]:
    st.subheader("Multi-file Compare")
    last = st.session_state.last_result
    if not last or len(last) < 2:
        st.info("Process at least 2 documents first in the ✅ Process tab.")
    else:
        names = [r["doc"]["name"] for r in last]
        a = st.selectbox("Document A", names, index=0, key="cmp_a")
        b = st.selectbox("Document B", names, index=1, key="cmp_b")

        ra = next(x for x in last if x["doc"]["name"] == a)
        rb = next(x for x in last if x["doc"]["name"] == b)

        set_a = set([k.lower() for k in ra["keywords"]])
        set_b = set([k.lower() for k in rb["keywords"]])

        overlap = sorted(set_a.intersection(set_b))
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Summary A")
            st.write("\n".join([f"- {s}" for s in ra["summary"]]) if ra["summary"] else "No summary.")
        with c2:
            st.markdown("### Summary B")
            st.write("\n".join([f"- {s}" for s in rb["summary"]]) if rb["summary"] else "No summary.")

        st.divider()
        st.markdown("### Keyword Comparison")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("**Overlap**")
            st.write(", ".join(overlap) if overlap else "None")
        with k2:
            st.markdown("**Only in A**")
            st.write(", ".join(only_a[:30]) if only_a else "None")
        with k3:
            st.markdown("**Only in B**")
            st.write(", ".join(only_b[:30]) if only_b else "None")


# -------------------------
# History Tab
# -------------------------
with tabs[5]:
    st.subheader("Saved History (this session)")
    hist = st.session_state.history
    if not hist:
        st.info("No history yet. Process a document first.")
    else:
        for h in hist[:15]:
            st.markdown(f"### {h['name']}")
            sec = ""
            if h.get("range"):
                sec += f" • Pages: {h['range'][0]}-{h['range'][1]}"
            if h.get("section"):
                sec += f" • Section: {h['section']}"
            st.caption(f"{h['created_at']} • User Type: {h.get('user_type','-')} • Size: {h.get('size_mb',0):.1f}MB{sec}")
            st.write("\n".join([f"- {s}" for s in h["summary"]]) if h["summary"] else "No summary.")
            st.write("Keywords: " + (", ".join(h["keywords"]) if h["keywords"] else "None"))
            st.divider()

        export = {"exported_at": time.strftime("%Y-%m-%d %H:%M:%S"), "history": hist}
        st.download_button(
            "⬇️ Download History (JSON)",
            data=json.dumps(export, indent=2),
            file_name="docsum_history.json",
            mime="application/json"
        )

st.caption("Note: Scanned PDFs (image-only) need OCR to extract text (we can add OCR later).")
