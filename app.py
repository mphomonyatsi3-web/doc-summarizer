import re
import html
from io import BytesIO
from typing import List, Tuple, Dict
from difflib import SequenceMatcher

import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Export
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")

APP_TITLE = "📄 Clarity Summarizer"
APP_TAGLINE = "Pick who you are → the app adapts the summary, highlights, and questions."

HARD_MAX_UPLOAD_MB = 300

# Char caps keep processing fast/stable (esp. Streamlit Cloud)
CHAR_LIMIT_OWNER = 450_000
CHAR_LIMIT_PAID = 250_000
CHAR_LIMIT_FREE = 180_000

PAGES_OWNER = 500
PAGES_PAID = 250
PAGES_FREE = 150

FREE_MAX_DOCS = 2
PAID_MAX_DOCS = 10
OWNER_MAX_DOCS = 10_000


# -----------------------------
# Secrets / PIN / Tier
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
        return {"max_docs": OWNER_MAX_DOCS, "max_pages": PAGES_OWNER, "char_limit": CHAR_LIMIT_OWNER, "max_mb": HARD_MAX_UPLOAD_MB}
    if tier == "paid":
        return {"max_docs": PAID_MAX_DOCS, "max_pages": PAGES_PAID, "char_limit": CHAR_LIMIT_PAID, "max_mb": HARD_MAX_UPLOAD_MB}
    if tier == "free":
        return {"max_docs": FREE_MAX_DOCS, "max_pages": PAGES_FREE, "char_limit": CHAR_LIMIT_FREE, "max_mb": HARD_MAX_UPLOAD_MB}
    return {"max_docs": 0, "max_pages": 0, "char_limit": 0, "max_mb": 0}


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

    lim = limits_for_tier(tier)
    st.sidebar.success(f"Access: {tier.upper()}")
    st.sidebar.caption(f"Docs this session: {st.session_state.docs_used}/{lim['max_docs']}")
    if st.session_state.docs_used >= lim["max_docs"]:
        st.error("Document limit reached for this session.")
        st.stop()

    return lim, tier


# -----------------------------
# Fast text utilities (no NLTK)
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_sentences_fast(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    t = re.sub(r"\s*\n\s*", " ", text)
    parts = re.split(r'(?<=[\.\!\?\;])\s+(?=[A-Z0-9"\'])', t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 3 and len(t) > 1200:
        chunk = 350
        parts = [t[i:i+chunk].strip() for i in range(0, len(t), chunk)]
    return parts


def tokenize_words(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    words = [w for w in text.split() if len(w) >= 3 and w not in ENGLISH_STOP_WORDS]
    return words


# -----------------------------
# Cached extraction
# -----------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_text_cached(file_bytes: bytes, max_pages: int, char_limit: int) -> Tuple[str, int]:
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
        tx = (p.text or "").strip()
        if tx:
            chunks.append(tx)
            collected += len(tx)
            if collected >= char_limit:
                break
    return normalize_text("\n".join(chunks))


@st.cache_data(show_spinner=False)
def extract_txt_cached(file_bytes: bytes, char_limit: int) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes.decode(errors="ignore")
    return normalize_text(text[:char_limit])


def looks_scanned_or_empty(text: str) -> bool:
    return len(text.strip()) < 300


# -----------------------------
# Summarization (TextRank-ish) + concept mining
# -----------------------------
def textrank_summary(sentences: List[str], top_n: int = 6) -> List[str]:
    if not sentences:
        return []
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    scores = np.ones(len(sentences))
    damping = 0.85
    for _ in range(20):
        prev = scores.copy()
        row_sums = sim.sum(axis=1)
        norm_sim = np.divide(sim, row_sums[:, None] + 1e-12)
        scores = (1 - damping) + damping * norm_sim.T.dot(prev)
        if np.abs(scores - prev).sum() < 1e-6:
            break

    idx = np.argsort(scores)[::-1][:top_n]
    idx_sorted = sorted(idx.tolist())
    return [sentences[i] for i in idx_sorted]


def tfidf_keywords(text: str, k: int = 12) -> List[str]:
    words = tokenize_words(text)
    if not words:
        return []
    doc = " ".join(words)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
    X = vec.fit_transform([doc])
    feats = np.array(vec.get_feature_names_out())
    scores = X.toarray().ravel()
    if scores.size == 0:
        return []
    top_idx = np.argsort(scores)[::-1]
    kws = []
    seen = set()
    for i in top_idx:
        term = feats[i].strip()
        if not term or term in seen:
            continue
        if term in ("chapter", "page", "section"):
            continue
        seen.add(term)
        kws.append(term)
        if len(kws) >= k:
            break
    return kws


def pick_sentences_by_pattern(sentences: List[str], patterns: List[str], cap: int) -> List[str]:
    out = []
    rx_list = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    for s in sentences:
        if any(rx.search(s) for rx in rx_list):
            out.append(s.strip())
            if len(out) >= cap:
                break
    return out


def concept_sentences(sentences: List[str]) -> Dict[str, List[str]]:
    """
    Detect sentence roles (definitions, rules, risks, examples, cause/effect).
    """
    defs = pick_sentences_by_pattern(sentences, [r"\bis\b", r"\bmeans\b", r"\brefers to\b", r"\bdefined as\b"], 12)
    rules = pick_sentences_by_pattern(sentences, [r"\bmust\b", r"\bshould\b", r"\brequired\b", r"\balways\b", r"\bnever\b"], 12)
    risks = pick_sentences_by_pattern(sentences, [r"\bavoid\b", r"\brisk\b", r"\bwarning\b", r"\bfailure\b", r"\bloss\b", r"\bleads to\b", r"\bresults in\b"], 12)
    examples = pick_sentences_by_pattern(sentences, [r"\bfor example\b", r"\bsuch as\b", r"\be\.g\.\b"], 12)
    cause = pick_sentences_by_pattern(sentences, [r"\bbecause\b", r"\btherefore\b", r"\bhence\b", r"\bas a result\b"], 12)
    return {"definitions": defs, "rules": rules, "risks": risks, "examples": examples, "cause_effect": cause}


# -----------------------------
# Non-repeating, harder question engine
# -----------------------------
def _norm_q(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^a-z0-9\s]", "", q)
    return q


def dedupe_questions(questions: List[str], similarity_threshold: float = 0.90) -> List[str]:
    """
    Removes exact duplicates and near-duplicates (very similar questions).
    """
    kept = []
    kept_norm = []
    for q in questions:
        q2 = q.strip()
        if not q2:
            continue
        n = _norm_q(q2)
        if not n:
            continue
        # exact dup
        if n in kept_norm:
            continue
        # near-dup
        is_near = False
        for existing in kept:
            ratio = SequenceMatcher(None, _norm_q(existing), n).ratio()
            if ratio >= similarity_threshold:
                is_near = True
                break
        if is_near:
            continue

        kept.append(q2)
        kept_norm.append(n)
    return kept


def build_advanced_questions(
    persona: str,
    summary_sents: List[str],
    keywords: List[str],
    concepts: Dict[str, List[str]],
    n: int
) -> List[str]:
    """
    Persona-aware question generator that produces variety and avoids repeats.
    """
    # Pick “concept anchors”
    kws = keywords[: min(10, len(keywords))]
    defs = concepts.get("definitions", [])
    rules = concepts.get("rules", [])
    risks = concepts.get("risks", [])
    cause = concepts.get("cause_effect", [])

    def pick_anchor_text(lst: List[str], fallback: List[str]) -> str:
        if lst:
            return lst[0]
        if fallback:
            return fallback[0]
        return ""

    anchor_def = pick_anchor_text(defs, summary_sents)
    anchor_rule = pick_anchor_text(rules, summary_sents)
    anchor_risk = pick_anchor_text(risks, summary_sents)
    anchor_cause = pick_anchor_text(cause, summary_sents)

    out = []

    # --- Persona templates ---
    if persona == "Varsity student":
        # Harder: application, evaluation, comparison, scenario, critical thinking
        if anchor_def:
            out.append(f"Explain the concept in your own words: “{anchor_def}”")
        if anchor_cause:
            out.append(f"Explain the cause-and-effect relationship stated here: “{anchor_cause}”")
        if anchor_rule:
            out.append(f"State the rule clearly, then justify *why* it should be followed: “{anchor_rule}”")
        if anchor_risk:
            out.append(f"Identify the risk and propose a mitigation strategy: “{anchor_risk}”")

        # Application + scenario
        for kw in kws[:4]:
            out.append(f"Apply **{kw}** to a realistic scenario *not mentioned* in the document. Show steps and reasoning.")
        # Evaluation
        for kw in kws[4:6]:
            out.append(f"Critically evaluate **{kw}**: assumptions, limitations, and when it might fail.")
        # Comparison: pick two keywords if possible
        if len(kws) >= 2:
            out.append(f"Compare **{kws[0]}** vs **{kws[1]}**. Give similarities, differences, and when to use each.")

    elif persona == "High school learner":
        # Explain + connect + light application
        if anchor_def:
            out.append(f"Define this in simple terms, then give an example: “{anchor_def}”")
        if anchor_cause:
            out.append(f"Explain why this happens: “{anchor_cause}”")
        if anchor_rule:
            out.append(f"What does this rule mean in practice? “{anchor_rule}”")

        for kw in kws[:6]:
            out.append(f"Explain **{kw}** and give one example from real life or schoolwork.")
        out.append("Summarize the main idea in 5 bullet points without copying sentences.")

    elif persona == "Primary school learner":
        # Very simple comprehension questions
        if anchor_def:
            out.append(f"What does this mean? “{anchor_def}”")
        for kw in kws[:5]:
            out.append(f"What is **{kw}**? Explain like I’m 10 years old.")
        out.append("Tell the story of the document in 5 short sentences.")

    elif persona == "Trader":
        # Convert doc into rules, scenarios, mistakes, checklist
        out.append("Turn this document into a clear trading checklist (before entry, entry, management, exit).")
        if anchor_rule:
            out.append(f"Convert this rule into a strict checklist item + an example: “{anchor_rule}”")
        if anchor_risk:
            out.append(f"What mistake is being warned about here, and how do you prevent it? “{anchor_risk}”")
        for kw in kws[:6]:
            out.append(f"Using **{kw}**, create a scenario: conditions → entry trigger → invalidation → risk management → exit.")
        out.append("List 5 common failure points from this strategy and how to avoid each one.")

    elif persona == "Professional / Work":
        # Decision + action items + risks
        out.append("Extract 10 action items from this document (as tasks someone can execute).")
        if anchor_rule:
            out.append(f"Rewrite this requirement as a policy statement + compliance checklist: “{anchor_rule}”")
        if anchor_risk:
            out.append(f"Identify the risk here and propose controls/mitigation: “{anchor_risk}”")
        for kw in kws[:6]:
            out.append(f"What is the business impact of **{kw}**? Include risks and opportunities.")
        out.append("Write a 1-paragraph executive summary + 5 bullet recommendations.")

    else:  # default: "General"
        if anchor_def:
            out.append(f"Explain clearly: “{anchor_def}”")
        for kw in kws[:8]:
            out.append(f"Explain **{kw}** and why it matters.")
        out.append("What are the 5 most important takeaways and why?")

    # Add a few summary-sentence questions (variety)
    for s in summary_sents[: min(6, len(summary_sents))]:
        s2 = re.sub(r"\s+", " ", s).strip()
        if len(s2) > 35:
            out.append(f"What is the main claim in this sentence, and what evidence would support it? “{s2[:180]}...”")

    # Deduplicate strongly so user doesn't see repeats
    out = dedupe_questions(out, similarity_threshold=0.90)

    # Cap to requested number
    return out[:n]


# -----------------------------
# Concept-based highlights (colored spans)
# -----------------------------
def highlight_concepts(text: str, concepts: Dict[str, List[str]], max_len: int = 90_000) -> str:
    """
    Highlights concept patterns, not just keywords.
    Color meaning:
      Blue = definitions, Orange = rules, Red = risks, Green = examples, Purple = cause/effect
    """
    t = (text or "")[:max_len]
    esc = html.escape(t)

    # Regex patterns (concept roles)
    patterns = [
        (r"\b(is|means|refers to|defined as)\b", "#dbeafe"),      # definitions (blue-ish)
        (r"\b(must|should|required|always|never)\b", "#ffedd5"),  # rules (orange-ish)
        (r"\b(avoid|risk|warning|failure|loss|leads to|results in)\b", "#fee2e2"),  # risks (red-ish)
        (r"\b(for example|such as|e\.g\.)\b", "#dcfce7"),         # examples (green-ish)
        (r"\b(because|therefore|hence|as a result)\b", "#f3e8ff") # cause/effect (purple-ish)
    ]

    # Wrap matches
    for rx, color in patterns:
        esc = re.sub(
            rx,
            lambda m: f'<span style="background:{color}; padding:0 4px; border-radius:6px;">{m.group(0)}</span>',
            esc,
            flags=re.IGNORECASE
        )

    return f"""
    <div style="white-space: pre-wrap; line-height: 1.6; font-family: system-ui;">
      {esc}
    </div>
    """


# -----------------------------
# Export
# -----------------------------
def build_pdf_bytes(title: str, summary_block: str, takeaways: List[str], questions: List[str]) -> bytes:
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{html.escape(title)}</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>Explanation Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(html.escape(summary_block), styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    if takeaways:
        story.append(Paragraph("<b>Key Takeaways</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(ListFlowable([ListItem(Paragraph(html.escape(t), styles["BodyText"])) for t in takeaways], bulletType="bullet"))
        story.append(Spacer(1, 0.3 * cm))

    if questions:
        story.append(Paragraph("<b>Practice Questions</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(ListFlowable([ListItem(Paragraph(html.escape(q), styles["BodyText"])) for q in questions], bulletType="bullet"))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    return buf.getvalue()


def build_docx_bytes(title: str, summary_block: str, takeaways: List[str], questions: List[str]) -> bytes:
    doc = DocxDocument()
    doc.add_heading(title, 0)

    doc.add_heading("Explanation Summary", level=1)
    doc.add_paragraph(summary_block)

    if takeaways:
        doc.add_heading("Key Takeaways", level=1)
        for t in takeaways:
            doc.add_paragraph(t, style="List Bullet")

    if questions:
        doc.add_heading("Practice Questions", level=1)
        for q in questions:
            doc.add_paragraph(q, style="List Bullet")

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# -----------------------------
# Lecturer-style explanation summary (structured)
# -----------------------------
def build_explanatory_summary(persona: str, summary_sents: List[str], concepts: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """
    Returns (summary_block, takeaways)
    Uses a structured, lecturer-like format.
    """
    defs = concepts.get("definitions", [])
    rules = concepts.get("rules", [])
    risks = concepts.get("risks", [])
    cause = concepts.get("cause_effect", [])
    examples = concepts.get("examples", [])

    core = summary_sents[:2]
    explain = summary_sents[2:5]
    extra = summary_sents[5:8]

    # Make takeaways from strongest sentences (non-repeating)
    takeaways = dedupe_questions([*core, *explain, *rules[:2], *risks[:2]], similarity_threshold=0.92)[:6]

    # Persona tone
    if persona == "Primary school learner":
        header = "Simple Explanation"
        guide = "Focus: understand the meaning, one idea at a time."
    elif persona == "High school learner":
        header = "Clear Explanation"
        guide = "Focus: understand + connect ideas + prepare for tests."
    elif persona == "Varsity student":
        header = "Lecturer-Style Explanation"
        guide = "Focus: meaning, assumptions, implications, and exam-style understanding."
    elif persona == "Trader":
        header = "Practical Clarity"
        guide = "Focus: rules, mistakes, and how to apply it step-by-step."
    elif persona == "Professional / Work":
        header = "Work-Ready Summary"
        guide = "Focus: decisions, actions, and risk controls."
    else:
        header = "Explanation Summary"
        guide = "Focus: clarity and key points."

    parts = []
    parts.append(f"{header}\n{guide}\n")

    if defs:
        parts.append("1) Key Definitions\n- " + "\n- ".join(defs[:3]) + "\n")
    if core:
        parts.append("2) Core Idea\n- " + "\n- ".join(core) + "\n")
    if cause:
        parts.append("3) Cause → Effect (Why things happen)\n- " + "\n- ".join(cause[:3]) + "\n")
    if rules:
        parts.append("4) Rules / Requirements (What you MUST do)\n- " + "\n- ".join(rules[:3]) + "\n")
    if risks:
        parts.append("5) Risks / Mistakes (What can go wrong)\n- " + "\n- ".join(risks[:3]) + "\n")
    if examples:
        parts.append("6) Examples (How it looks in practice)\n- " + "\n- ".join(examples[:2]) + "\n")

    if explain or extra:
        combined = [*explain, *extra]
        combined = [c for c in combined if len(c.strip()) > 20][:4]
        if combined:
            parts.append("7) Important Notes\n- " + "\n- ".join(combined) + "\n")

    summary_block = "\n".join(parts).strip()
    return summary_block, takeaways


# -----------------------------
# Main UI
# -----------------------------
def main():
    lim, tier = auth_gate()

    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    # Scrollable role selector (selectbox scrolls when long)
    st.sidebar.subheader("🧠 Who are you?")
    persona = st.sidebar.selectbox(
        "Choose one (the app adapts)",
        [
            "Varsity student",
            "High school learner",
            "Primary school learner",
            "Trader",
            "Professional / Work",
            "General",
        ],
        index=0
    )

    persona_help = {
        "Varsity student": "Hard questions: apply, compare, evaluate, scenarios. Lecturer-style summary.",
        "High school learner": "Clear explanations + test-style questions + examples.",
        "Primary school learner": "Simpler language + gentle questions.",
        "Trader": "Turns docs into rules/checklist + scenarios + mistakes.",
        "Professional / Work": "Action items + decisions + risks.",
        "General": "Balanced summary + mixed questions."
    }
    st.sidebar.caption(persona_help.get(persona, ""))

    st.sidebar.subheader("⚙️ Output settings")
    summary_sentences = st.sidebar.slider("Summary strength (sentences)", 4, 14, 8)
    question_count = st.sidebar.slider("Question count", 6, 20, 12)
    keyword_count = st.sidebar.slider("Keyword count", 8, 30, 15)
    quick_mode = st.sidebar.toggle("⚡ Quick mode (faster)", value=True)

    # PDF pages slider (role-aware defaults)
    default_pages = 50 if quick_mode else min(120, lim["max_pages"])
    pages_to_analyze = st.sidebar.slider(
        "Pages to analyze (PDF)",
        5, lim["max_pages"], min(default_pages, lim["max_pages"])
    )

    st.write("Upload a **PDF, DOCX, or TXT**. The app will explain it, highlight what matters, and generate non-repeating questions.")
    uploaded = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])

    if not uploaded:
        st.info("Upload a document to begin.")
        return

    # size checks
    data = uploaded.getvalue()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > lim["max_mb"]:
        st.error(f"File too large: {size_mb:.1f}MB. Max allowed is {lim['max_mb']}MB.")
        return

    # Process button prevents rerun reprocessing
    if st.button("⚡ Process Document"):
        st.session_state.docs_used += 1

        with st.spinner("Reading document..."):
            ext = uploaded.name.lower().split(".")[-1]
            char_limit = lim["char_limit"]

            if ext == "pdf":
                raw_text, pages_done = extract_pdf_text_cached(data, pages_to_analyze, char_limit)
            elif ext == "docx":
                raw_text = extract_docx_text_cached(data, char_limit)
                pages_done = None
            else:
                raw_text = extract_txt_cached(data, char_limit)
                pages_done = None

        if looks_scanned_or_empty(raw_text):
            st.error(
                "This PDF looks scanned/locked (no selectable text). "
                "Try a selectable-text PDF or DOCX/TXT. (OCR can be added later.)"
            )
            return

        # Sentence + concept analysis
        sentences = split_sentences_fast(raw_text)
        if len(sentences) < 4:
            st.error("Not enough readable text extracted. Try another file.")
            return

        with st.spinner("Building explanation, highlights, and questions..."):
            summary_sents = textrank_summary(sentences, top_n=summary_sentences)
            concepts = concept_sentences(sentences)
            keywords = tfidf_keywords(raw_text, k=keyword_count)

            summary_block, takeaways = build_explanatory_summary(persona, summary_sents, concepts)

            questions = build_advanced_questions(
                persona=persona,
                summary_sents=summary_sents,
                keywords=keywords,
                concepts=concepts,
                n=question_count
            )

            # Final dedupe pass (safety)
            questions = dedupe_questions(questions, similarity_threshold=0.90)

        # Meta
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("File", uploaded.name)
        c2.metric("Size (MB)", f"{size_mb:.1f}")
        c3.metric("Chars used", f"{len(raw_text):,}")
        c4.metric("PDF pages read", str(pages_done) if pages_done is not None else ext.upper())

        t1, t2, t3, t4 = st.tabs(["✅ Explanation", "🟣 Highlights", "❓ Questions", "⬇️ Export"])

        with t1:
            st.subheader("Explanation (formal, simple, structured)")
            st.text_area("Explanation Summary", summary_block, height=320)
            st.subheader("Key Takeaways")
            for t in takeaways:
                st.write("•", t)

        with t2:
            st.subheader("Concept Highlights (not just keywords)")
            st.caption("Blue=definitions • Orange=rules • Red=risks • Green=examples • Purple=cause/effect")
            st.markdown(highlight_concepts(raw_text, concepts), unsafe_allow_html=True)

        with t3:
            st.subheader("Non-repeating Practice Questions")
            st.caption("These questions are generated in different types (apply/compare/evaluate/scenario) depending on who you selected.")
            for i, q in enumerate(questions, start=1):
                st.write(f"**{i}.** {q}")

        with t4:
            st.subheader("Export (PDF / Word)")
            pdf_bytes = build_pdf_bytes(uploaded.name, summary_block, takeaways, questions)
            docx_bytes = build_docx_bytes(uploaded.name, summary_block, takeaways, questions)

            st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="clarity_pack.pdf", mime="application/pdf")
            st.download_button(
                "⬇️ Download Word (.docx)",
                data=docx_bytes,
                file_name="clarity_pack.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    else:
        st.warning("Click **⚡ Process Document** to generate results.")


if __name__ == "__main__":
    main()
