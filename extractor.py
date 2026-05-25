"""
extractor.py
------------
Structured field extraction for classified documents.

Each document type has a dedicated extractor function that applies a
prioritised list of regex patterns.  The first pattern that matches wins,
which lets us handle many real-world layout variations without an ML model.
"""

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _first_match(patterns: list[str], text: str, flags: int = re.IGNORECASE) -> Optional[str]:
    """
    Try each pattern in order and return the first captured group that matches.
    Returns None if no pattern matches.
    """
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            val = m.group(1).strip()
            if val:
                return val
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    """Convert a string like '1,234.56' or '1.234,56' to a float."""
    if value is None:
        return None
    # Remove currency symbols and whitespace
    value = re.sub(r"[£€$\s]", "", value)
    # Handle European comma-as-decimal: 1.234,56 → 1234.56
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", value):
        value = value.replace(".", "").replace(",", ".")
    else:
        value = value.replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: Optional[str]) -> Optional[int]:
    """Convert a matched string to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(re.sub(r"[^\d]", "", value))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Invoice extractor
# ---------------------------------------------------------------------------

def _extract_invoice(text: str) -> dict[str, Any]:
    invoice_number = _first_match([
        r"invoice[\s_#-]?(?:no|number|num|#)[:\s#]*([A-Z0-9][\w/-]{2,20})",
        r"inv[\s_#-]?(?:no|number|num|#)[:\s#]*([A-Z0-9][\w/-]{2,20})",
        r"\bINV[-\s]?(\d{3,})\b",
        r"invoice[:\s]+([A-Z0-9][\w/-]{2,20})",
    ], text)

    date = _first_match([
        r"invoice\s+date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"date\s+of\s+invoice[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:date|issued)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:date|issued)[:\s]+(\w+ \d{1,2},?\s*\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
    ], text)

    company = _first_match([
        r"(?:from|billed?\s+by|seller|vendor|company)[:\s]+([A-Z][^\n,]{2,60})",
        r"(?:bill\s+from)[:\s]+([A-Z][^\n,]{2,60})",
    ], text, flags=re.IGNORECASE | re.MULTILINE)

    if not company:
        # Fall back: first line that ends with a known company suffix
        m = re.search(
            r"^([A-Z][A-Za-z0-9 &.,'-]{3,60}"
            r"(?:Ltd\.?|LLC|Inc\.?|Corp\.?|Co\.?|GmbH|PLC|Limited|Group|Services?|Consulting))\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            company = m.group(1).strip()

    if not company:
        # Last resort: first non-empty capitalised line (likely the letterhead)
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^[A-Z][A-Za-z0-9 &.,'-]{4,60}$", line):
                company = line
                break

    total_amount = _to_float(_first_match([
        r"total\s+(?:amount\s+)?(?:due|payable)[:\s$£€]*([\d,]+\.?\d*)",
        r"(?:grand\s+)?total[:\s$£€]*([\d,]+\.?\d*)",
        r"amount\s+(?:due|payable)[:\s$£€]*([\d,]+\.?\d*)",
        r"balance\s+due[:\s$£€]*([\d,]+\.?\d*)",
    ], text))

    return {
        "invoice_number": invoice_number,
        "date": date,
        "company": company,
        "total_amount": total_amount,
    }


# ---------------------------------------------------------------------------
# Resume extractor
# ---------------------------------------------------------------------------

def _extract_resume(text: str) -> dict[str, Any]:
    # Name: look for explicit label first, then fall back to the first
    # capitalised line that is NOT a known section header.
    _SECTION_HEADERS = re.compile(
        r"^(curriculum vitae|resume|objective|summary|profile|education|"
        r"skills?|certifications?|references?|experience|employment)$",
        re.IGNORECASE,
    )
    name = _first_match([
        # "Name: Dr. Amir Hassan" or "Name: Jane Mitchell"
        r"(?:full\s+)?name[:\s]+(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)?\s*([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})",
    ], text, flags=re.IGNORECASE | re.MULTILINE)

    if not name:
        # Scan lines for the first proper-cased full name (2–4 words),
        # also handling titles like Dr., Mr., Ms.
        for line in text.splitlines():
            line = line.strip()
            if _SECTION_HEADERS.match(line):
                continue
            # With optional title prefix
            m = re.match(
                r"^(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)?\s*"
                r"([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\s*$",
                line,
            )
            if m:
                name = m.group(1)
                break

    email = _first_match([
        r"([\w.+-]+@[\w-]+\.[\w.]{2,})",
    ], text)

    phone = _first_match([
        r"(?:phone|mobile|tel|cell)[:\s]*([\+\d][\d\s\-().]{7,20})",
        r"(\+?1?\s*[\(\-]?\d{3}[\)\-\s]?\s*\d{3}[\-\s]?\d{4})",
        r"(\+\d{1,3}[\s\-]?\d{6,14})",
    ], text)
    if phone:
        phone = re.sub(r"\s+", " ", phone).strip()

    experience_years = _to_int(_first_match([
        r"(\d+)\+?\s+years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience[:\s]+(\d+)\+?\s+years?",
        r"(\d+)\+?\s+years?\s+(?:in|of)\s+\w+",
    ], text))

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "experience_years": experience_years,
    }


# ---------------------------------------------------------------------------
# Utility Bill extractor
# ---------------------------------------------------------------------------

def _extract_utility_bill(text: str) -> dict[str, Any]:
    account_number = _first_match([
        r"account[\s_-]?(?:no|number|num|#)[:\s#]*([A-Z0-9][\w-]{3,20})",
        r"customer[\s_-]?(?:no|number|id)[:\s#]*([A-Z0-9][\w-]{3,20})",
        r"acct[\s_#.]*([A-Z0-9][\w-]{3,20})",
    ], text)

    date = _first_match([
        r"(?:bill|statement|invoice)\s+date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:billing\s+period)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:date)[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:date)[:\s]+(\w+ \d{1,2},?\s*\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
    ], text)

    usage_kwh = _to_float(_first_match([
        r"(?:total\s+)?(?:energy\s+)?usage[:\s]*([\d,]+\.?\d*)\s*kwh",
        r"([\d,]+\.?\d*)\s*kwh\s+(?:used|consumed|total)",
        r"kwh[:\s]*([\d,]+\.?\d*)",
        r"electricity\s+used[:\s]*([\d,]+\.?\d*)",
    ], text))

    amount_due = _to_float(_first_match([
        r"(?:total\s+)?amount\s+due[:\s$£€]*([\d,]+\.?\d*)",
        r"(?:please\s+pay)[:\s$£€]*([\d,]+\.?\d*)",
        r"balance\s+due[:\s$£€]*([\d,]+\.?\d*)",
        r"total\s+(?:charges?|bill)[:\s$£€]*([\d,]+\.?\d*)",
    ], text))

    return {
        "account_number": account_number,
        "date": date,
        "usage_kwh": usage_kwh,
        "amount_due": amount_due,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_fields(doc_class: str, text: str) -> dict[str, Any]:
    """
    Extract structured fields from *text* based on *doc_class*.

    Returns an empty dict for "Other" and "Unclassifiable" documents.
    """
    if doc_class == "Invoice":
        return _extract_invoice(text)
    if doc_class == "Resume":
        return _extract_resume(text)
    if doc_class == "Utility Bill":
        return _extract_utility_bill(text)
    return {}
