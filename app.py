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
from search import SemanticSearch

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
def get_search_engine(doc_texts: tuple, doc_names: tuple) -> SemanticSearch:
    """
    Cache the FAISS index so it isn't rebuilt every interaction.
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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    st.title("📄 Doc AI")
    st.caption("Local • Offline • Open-source")

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
        placeholder='e.g. "Find all documents mentioning payments due in January"',
    )

    # FIXED SLIDER ISSUE
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

        with st.spinner("Building semantic index and searching…"):

            fnames = list(results.keys())

            texts = [
                results[f]["text"]
                for f in fnames
            ]

            engine = get_search_engine(
                tuple(texts),
                tuple(fnames)
            )

            hits = engine.search(
                query,
                top_k=top_k
            )

            st.session_state.search_hits = hits

    if st.session_state.search_hits:

        st.divider()

        st.caption(f"Results for: **{query}**")

        for rank, (fname, score) in enumerate(
            st.session_state.search_hits,
            1
        ):

            doc_class = results[fname]["class"]

            _, icon = CLASS_COLORS.get(
                doc_class,
                ("", "📄")
            )

            bar_val = max(0.0, float(score))

            c1, c2, c3 = st.columns([0.5, 3, 1.5])

            c1.markdown(f"**#{rank}**")

            c2.markdown(
                f"{icon} **{fname}**  `{doc_class}`",
                unsafe_allow_html=True
            )

            c3.progress(
                bar_val,
                text=f"{score:.4f}"
            )

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