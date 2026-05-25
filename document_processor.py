"""
document_processor.py
---------------------
Handles reading and text extraction from PDF, TXT, and DOCX files.
Uses PyPDF2 as the primary PDF extractor with pdfminer.six as a fallback
for scanned or complex PDFs.
"""

import os
import re

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return " ".join(pages)
    except Exception as e:
        print(f"  [PyPDF2 failed for {pdf_path}]: {e}")
        return ""


def _extract_pdf_pdfminer(pdf_path: str) -> str:
    """Extract text using pdfminer.six (fallback)."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path) or ""
    except Exception as e:
        print(f"  [pdfminer failed for {pdf_path}]: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Try PyPDF2 first; fall back to pdfminer if the result is too short."""
    text = _extract_pdf_pypdf2(pdf_path)
    if len(text.strip()) < 50:
        text = _extract_pdf_pdfminer(pdf_path)
    return text


# ---------------------------------------------------------------------------
# TXT / DOCX extraction
# ---------------------------------------------------------------------------

def extract_text_from_txt(txt_path: str) -> str:
    """Read a plain-text file, trying UTF-8 then latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            with open(txt_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  [Error reading {txt_path}]: {e}")
            return ""
    return ""


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document
        doc = Document(docx_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"  [Error reading {docx_path}]: {e}")
        return ""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove non-printable characters while keeping
    punctuation that is useful for field extraction (e.g. @, -, /).
    """
    # Replace form-feeds, tabs, and multiple spaces/newlines with a single space
    text = re.sub(r"[\r\f\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Remove non-printable characters (keep standard ASCII + common unicode)
    text = re.sub(r"[^\x20-\x7E\n\u00C0-\u024F]", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def process_documents(folder: str) -> dict[str, str]:
    """
    Walk *folder* and extract cleaned text from every supported file.

    Returns
    -------
    dict mapping filename → cleaned text string
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Input folder does not exist: {folder}")

    docs: dict[str, str] = {}
    files = sorted(os.listdir(folder))

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        fpath = os.path.join(folder, fname)
        print(f"  Processing: {fname}")

        if ext == ".pdf":
            raw = extract_text_from_pdf(fpath)
        elif ext == ".txt":
            raw = extract_text_from_txt(fpath)
        elif ext == ".docx":
            raw = extract_text_from_docx(fpath)
        else:
            raw = ""

        docs[fname] = clean_text(raw)

    return docs
