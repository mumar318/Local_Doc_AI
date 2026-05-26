"""
app.py
------
Streamlit UI for the Local AI Document Understanding System.

Run with:
    streamlit run app.py
"""

import io
import json
import os
import tempfile

import streamlit as st

from classifier import classify_document
from document_processor import (
    process_documents,
    clean_text,
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx,
)
from extractor import extract_fields
from search import SemanticSearch, load_embedding_model, fingerprint

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local AI Document Understanding",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .class-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .badge-invoice {
        background:#dbeafe;
        color:#1d4ed8;
    }

    .badge-resume {
        background:#dcfce7;
        color:#15803d;
    }

    .badge-utility {
        background:#fef9c3;
        color:#a16207;
    }

    .badge-other {
        background:#f3f4f6;
        color:#374151;
    }

    .badge-unclass {
        background:#fee2e2;
        color:#b91c1c;
    }

    .field-table td {
        padding: 2px 10px 2px 0;
        vertical-align: top;
    }

    .field-table td:first-child {
        color: #6b7280;
        font-size: 0.85rem;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "Invoice": ("badge-invoice", "🧾"),
    "Resume": ("badge-resume", "👤"),
    "Utility Bill": ("badge-utility", "⚡"),
    "Other": ("badge-other", "📁"),
    "Unclassifiable": ("badge-unclass", "❓"),
}

FIELD_LABELS = {
    # Invoice
    "invoice_number": "Invoice #",
    "date": "Date",
    "company": "Company",
    "total_amount": "Total Amount",

    # Resume
    "name": "Name",
    "email": "Email",
    "phone": "Phone",
    "experience_years": "Experience (yrs)",
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "skills": "Skills",

    # Utility Bill
    "account_number": "Account #",
    "usage_kwh": "Usage (kWh)",
    "amount_due": "Amount Due",
}


def badge_html(doc_class: str) -> str:
    css, icon = CLASS_COLORS.get(doc_class, ("badge-other", "📄"))
    return f'<span class="class-badge {css}">{icon} {doc_class}</span>'


def render_fields(fields: dict) -> str:
    rows = ""

    for key, val in fields.items():
        if key == "class":
            continue

        label = FIELD_LABELS.get(key, key.replace("_", " ").title())

        display = (
            val if val is not None
            else "<span style='color:#9ca3af'>—</span>"
        )

        rows += f"""
        <tr>
            <td>{label}</td>
            <td><b>{display}</b></td>
        </tr>
        """

    return f"<table class='field-table'>{rows}</table>" if rows else ""


@st.cache_resource(show_spinner="Loading embedding model…")
def _warm_model():
    """Pre-load the embedding model once at startup so searches are instant."""
    return load_embedding_model()


@st.cache_resource(show_spinner="Building search index…")
def get_search_engine(doc_fingerprint: str, doc_texts: tuple, doc_names: tuple, _version: int = 2) -> SemanticSearch:
    """
    Cache the FAISS index keyed on a fingerprint of the document texts.
    The model itself is already loaded by _warm_model() so this only
    re-runs when the document set actually changes.
    """
    return SemanticSearch(list(doc_texts), list(doc_names))


def process_uploaded_files(uploaded_files) -> dict:
    """
    Save uploads to temp dir and process documents.
    """

    with tempfile.TemporaryDirectory() as tmpdir:

        for uf in uploaded_files:
            dest = os.path.join(tmpdir, uf.name)

            with open(dest, "wb") as f:
                f.write(uf.getbuffer())

        raw_docs = process_documents(tmpdir)

    results = {}

    for fname, text in raw_docs.items():
        doc_class = classify_document(text)
        fields = extract_fields(doc_class, text)

        results[fname] = {
            "class": doc_class,
            "text": text,
            **fields,
        }

    return results


def process_sample_docs() -> dict:
    """
    Process bundled sample_docs folder.
    """

    if not os.path.isdir("sample_docs"):
        return {}

    raw_docs = process_documents("sample_docs")

    results = {}

    for fname, text in raw_docs.items():
        doc_class = classify_document(text)
        fields = extract_fields(doc_class, text)

        results[fname] = {
            "class": doc_class,
            "text": text,
            **fields,
        }

    return results


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = {}

if "search_hits" not in st.session_state:
    st.session_state.search_hits = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    st.title("📄 Doc AI")
    st.caption("Local • Offline • Open-source")

    # Show model warm-up status
    with st.spinner("Warming up embedding model…"):
        _warm_model()
    st.caption("✅ Embedding model ready")

    st.divider()

    st.subheader("1 · Load Documents")

    source = st.radio(
        "Source",
        ["Upload files", "Use sample_docs folder"],
        label_visibility="collapsed"
    )

    # Upload files
    if source == "Upload files":

        uploaded = st.file_uploader(
            "Drop PDF, TXT or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )

        if st.button("Process", type="primary", disabled=not uploaded):

            with st.spinner("Processing documents…"):

                st.session_state.results = process_uploaded_files(uploaded)
                st.session_state.search_hits = []

            st.success(
                f"Processed {len(st.session_state.results)} document(s)"
            )

    # Sample docs
    else:

        if st.button("Process sample_docs", type="primary"):

            with st.spinner("Processing sample documents…"):

                st.session_state.results = process_sample_docs()
                st.session_state.search_hits = []

            st.success(
                f"Processed {len(st.session_state.results)} document(s)"
            )

    st.divider()

    # Filter
    st.subheader("2 · Filter by Class")

    all_classes = (
        sorted({v["class"] for v in st.session_state.results.values()})
        if st.session_state.results
        else []
    )

    selected_classes = st.multiselect(
        "Show only",
        all_classes,
        default=all_classes
    )

    st.divider()

    # Export
    if st.session_state.results:

        st.subheader("3 · Export")

        export_data = {
            k: {fk: fv for fk, fv in v.items() if fk != "text"}
            for k, v in st.session_state.results.items()
        }

        st.download_button(
            "⬇ Download output.json",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name="output.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Local AI Document Understanding System")

st.caption(
    "Classify · Extract · Search — 100% offline, no paid APIs"
)

if not st.session_state.results:
    st.info("👈 Load documents from the sidebar to get started.")
    st.stop()

results = st.session_state.results

filtered = {
    k: v
    for k, v in results.items()
    if v["class"] in (selected_classes or all_classes)
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_results, tab_search, tab_json = st.tabs([
    "📋 Results",
    "🔍 Semantic Search",
    "{ } JSON"
])

# ---------------------------------------------------------------------------
# TAB 1 - RESULTS
# ---------------------------------------------------------------------------
with tab_results:

    from collections import Counter

    counts = Counter(v["class"] for v in results.values())

    cols = st.columns(len(CLASS_COLORS))

    for col, (cls, (_, icon)) in zip(cols, CLASS_COLORS.items()):
        col.metric(f"{icon} {cls}", counts.get(cls, 0))

    st.divider()

    if not filtered:
        st.warning("No documents match the selected filter.")

    else:

        for fname, data in filtered.items():

            doc_class = data["class"]

            css, icon = CLASS_COLORS.get(
                doc_class,
                ("badge-other", "📄")
            )

            with st.expander(f"{icon} {fname}", expanded=False):

                col_left, col_right = st.columns([1, 2])

                # Left
                with col_left:

                    st.markdown(
                        badge_html(doc_class),
                        unsafe_allow_html=True
                    )

                    fields = {
                        k: v
                        for k, v in data.items()
                        if k not in ("class", "text")
                    }

                    if fields:
                        st.markdown(
                            render_fields(fields),
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption(
                            "No fields extracted for this class."
                        )

                # Right
                with col_right:

                    st.caption("Document text preview")

                    preview = data.get("text", "")[:800]

                    st.text_area(
                        "",
                        value=preview,
                        height=180,
                        disabled=True,
                        label_visibility="collapsed",
                    )

# ---------------------------------------------------------------------------
# TAB 2 - SEMANTIC SEARCH
# ---------------------------------------------------------------------------
with tab_search:

    st.subheader("Search documents by meaning")

    query = st.text_input(
        "Query",
        placeholder='e.g. "Python developer with machine learning experience"',
    )

    # Slider — hide when only 1 doc
    if len(results) == 1:
        top_k = 1
        st.info("Only 1 document available for search.")
    else:
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=len(results),
            value=min(5, len(results)),
        )

    if st.button("Search", type="primary", disabled=not query):

        with st.spinner("Searching…"):

            fnames = list(results.keys())
            texts = [results[f]["text"] for f in fnames]
            fp = fingerprint(texts)

            engine = get_search_engine(fp, tuple(texts), tuple(fnames))
            hits = engine.search(query, top_k=top_k)

            # Attach best passage for each hit
            hits_with_snippets = []
            for fname, score in hits:
                doc_text = results[fname]["text"]
                snippet = engine.best_passage(query, doc_text, window=350)
                hits_with_snippets.append((fname, score, snippet))

            st.session_state.search_hits = hits_with_snippets
            st.session_state.last_query = query

    if st.session_state.search_hits:

        st.divider()
        last_q = st.session_state.get("last_query", query)
        st.caption(f"Results for: **{last_q}**")

        for rank, item in enumerate(st.session_state.search_hits, 1):

            fname, score, snippet = item
            doc_class = results[fname]["class"]
            css, icon = CLASS_COLORS.get(doc_class, ("badge-other", "📄"))

            # Colour the score: green ≥ 0.45, amber ≥ 0.25, red below
            if score >= 0.45:
                score_color = "#15803d"
            elif score >= 0.25:
                score_color = "#a16207"
            else:
                score_color = "#b91c1c"

            score_pct = int(max(0.0, min(1.0, score)) * 100)

            with st.container():
                # Header row
                hcol1, hcol2, hcol3 = st.columns([0.4, 4, 1.5])
                hcol1.markdown(f"### #{rank}")
                hcol2.markdown(
                    f'<span class="class-badge {css}">{icon} {doc_class}</span> '
                    f"**{fname}**",
                    unsafe_allow_html=True,
                )
                hcol3.markdown(
                    f'<div style="text-align:right;font-size:1.1rem;'
                    f'font-weight:700;color:{score_color}">'
                    f'{score_pct}% match</div>',
                    unsafe_allow_html=True,
                )

                # Extracted fields summary
                fields = {
                    k: v
                    for k, v in results[fname].items()
                    if k not in ("class", "text") and v is not None
                }
                if fields:
                    field_parts = []
                    for k, v in fields.items():
                        label = FIELD_LABELS.get(k, k.replace("_", " ").title())
                        field_parts.append(f"**{label}:** {v}")
                    st.markdown("  ·  ".join(field_parts))

                # Best-matching passage snippet
                if snippet:
                    st.markdown(
                        f'<div style="background:#f8fafc;border-left:3px solid #6366f1;'
                        f'padding:8px 12px;border-radius:4px;font-size:0.88rem;'
                        f'color:#374151;margin:6px 0 12px 0">'
                        f'📌 {snippet}</div>',
                        unsafe_allow_html=True,
                    )

                st.divider()

# ---------------------------------------------------------------------------
# TAB 3 - JSON
# ---------------------------------------------------------------------------
with tab_json:

    st.subheader("output.json")

    export_data = {
        k: {fk: fv for fk, fv in v.items() if fk != "text"}
        for k, v in filtered.items()
    }

    st.json(export_data)