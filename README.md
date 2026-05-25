# Local AI Document Understanding System

A fully offline pipeline that ingests PDF, TXT, and DOCX documents, classifies
each one, extracts structured fields, and supports natural-language semantic
search — all using open-source libraries with no paid or hosted AI APIs.

---

## Features

| Capability | Details |
|---|---|
| Document ingestion | PDF (PyPDF2 + pdfminer fallback), TXT, DOCX |
| Classification | Invoice · Resume · Utility Bill · Other · Unclassifiable |
| Field extraction | Per-class structured fields via regex |
| Semantic search | SentenceTransformers embeddings + FAISS cosine similarity |
| Output | `output.json` with class + extracted fields per document |

---

## Project Structure

```
.
├── main.py                  # CLI entry point
├── document_processor.py    # Text extraction & cleaning
├── classifier.py            # Weighted keyword-scoring classifier
├── extractor.py             # Per-class regex field extractor
├── search.py                # SemanticSearch class (FAISS + SentenceTransformers)
├── requirements.txt
├── output.json              # Generated after running the pipeline
└── sample_docs/             # Sample dataset (10 documents)
    ├── invoice_1.txt
    ├── invoice_2.txt
    ├── invoice_3.txt
    ├── resume_1.txt
    ├── resume_2.txt
    ├── resume_3.txt
    ├── utility_bill_1.txt
    ├── utility_bill_2.txt
    ├── utility_bill_3.txt
    ├── other_1.txt
    ├── other_2.txt
    ├── other_3.txt
    └── unclassifiable_1.txt
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone or unzip the project
cd local-ai-doc-system

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model
> (~90 MB) on first use. After that, everything runs fully offline.

---

## Usage

### Process documents and generate `output.json`

```bash
python main.py --input_folder ./sample_docs --output_json output.json
```

### Process + semantic search in one command

```bash
python main.py --input_folder ./sample_docs \
               --output_json output.json \
               --search "Find all documents mentioning payments due in January"
```

### Semantic search only (skip classification)

```bash
python main.py --input_folder ./sample_docs \
               --search "electricity usage and energy charges" \
               --search_only
```

### Control the number of search results

```bash
python main.py --input_folder ./sample_docs \
               --search "invoice total amount" \
               --top_k 3
```

### Full argument reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_folder` | Yes | — | Folder with PDF/TXT/DOCX files |
| `--output_json` | No | `output.json` | Output file path |
| `--search` | No | — | Natural-language search query |
| `--top_k` | No | `5` | Number of search results |
| `--search_only` | No | `False` | Skip classification, only search |

---

## Output Format

`output.json` contains one entry per document:

```json
{
  "invoice_1.txt": {
    "class": "Invoice",
    "invoice_number": "INV-2025-0042",
    "date": "15/01/2025",
    "company": "ACME Corporation",
    "total_amount": 351.90
  },
  "resume_1.txt": {
    "class": "Resume",
    "name": "Jane Mitchell",
    "email": "jane.mitchell@email.com",
    "phone": "+1 (415) 555-0192",
    "experience_years": 7
  },
  "utility_bill_1.txt": {
    "class": "Utility Bill",
    "account_number": "PG-4471-8823",
    "date": "January 5, 2025",
    "usage_kwh": 682.0,
    "amount_due": 107.09
  },
  "other_1.txt": {
    "class": "Other"
  },
  "unclassifiable_1.txt": {
    "class": "Unclassifiable"
  }
}
```

Fields that could not be extracted are set to `null`.

---

## Libraries and Methods

### Document Processing — `document_processor.py`

| Library | Purpose |
|---|---|
| **PyPDF2** | Primary PDF text extraction |
| **pdfminer.six** | Fallback PDF extraction for complex/scanned PDFs |
| **python-docx** | DOCX text extraction |
| Built-in `re`, `os` | Text cleaning, file traversal |

Text is cleaned by normalising whitespace, removing non-printable characters,
and collapsing redundant newlines.

### Classification — `classifier.py`

No ML model is used. Each document category (Invoice, Resume, Utility Bill) has
a curated list of regex patterns with associated weights. The category with the
highest cumulative score wins, provided it exceeds a minimum threshold. Documents
below the threshold are labelled **Other**; documents with too little text are
labelled **Unclassifiable**.

This approach is fast, fully offline, and interpretable.

### Field Extraction — `extractor.py`

Per-class regex extractors with prioritised pattern lists. Each field tries
multiple patterns in order of specificity, returning the first match. Numeric
fields (amounts, years) are parsed to `float` or `int`.

### Semantic Search — `search.py`

| Library | Purpose |
|---|---|
| **sentence-transformers** | Encode documents and queries into dense vectors |
| **faiss-cpu** | Fast approximate nearest-neighbour search |
| **numpy** | Vector normalisation |

Model: `all-MiniLM-L6-v2` — a lightweight (90 MB) model that produces
384-dimensional embeddings. Vectors are L2-normalised so that FAISS inner-product
search is equivalent to cosine similarity. Scores range from -1 to 1; higher
means more semantically similar.

---

## Running Fully Offline

After the initial `pip install` and first model download, the system runs with
no internet connection. The SentenceTransformers model is cached locally in
`~/.cache/torch/sentence_transformers/`.

To pre-download the model explicitly:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')  # downloads and caches
```

---

## Optional Bonus: Local Question-Answering

The semantic search component can be extended into a full RAG (Retrieval-Augmented
Generation) pipeline using a local LLM such as Mistral or LLaMA via
[Ollama](https://ollama.com) or the `transformers` library:

```python
# Pseudocode — extend search.py
top_docs = engine.search(query, top_k=3)
context = "\n\n".join(docs[fname] for fname, _ in top_docs)
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
# Pass prompt to a local LLM (e.g., via ollama.chat or transformers pipeline)
```

This is not required for the core solution but demonstrates how the retrieval
layer integrates naturally with generative models.

---

## Technical Constraints

- **No paid or hosted AI APIs** — OpenAI, Claude, Gemini, etc. are not used.
- **All processing is local** — no data leaves the machine.
- **Python 3.8+** compatible.
