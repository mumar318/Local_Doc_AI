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
        r"core\s+technical\s+skills?|technical\s+skills?|skills?|certifications?|"
        r"references?|experience|employment|work\s+experience|professional\s+experience|"
        r"employment\s+history|projects?|achievements?|awards?|languages?|"
        r"interests?|hobbies|contact|personal\s+information|about\s+me)$",
        re.IGNORECASE,
    )

    # Also treat any ALL-CAPS line or a line with 3+ title-case words as a header
    def _is_header(line: str) -> bool:
        if _SECTION_HEADERS.match(line):
            return True
        # All-caps line (e.g. "WORK EXPERIENCE")
        if line.isupper() and len(line) > 3:
            return True
        # Lines that look like section headings: 3+ title-case words
        words = line.split()
        if len(words) >= 3 and all(w[0].isupper() for w in words if w.isalpha()):
            # Likely a heading like "Core Technical Skills" — skip unless it
            # could be a real name (max 4 words, no common heading keywords)
            heading_keywords = re.compile(
                r"\b(skills?|experience|education|summary|profile|"
                r"technical|professional|employment|history|core|"
                r"certifications?|projects?|achievements?)\b",
                re.IGNORECASE,
            )
            if heading_keywords.search(line):
                return True
        return False

    name = _first_match([
        # "Name: Dr. Amir Hassan" or "Name: Jane Mitchell"
        r"(?:full\s+)?name[:\s]+(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)?\s*([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})",
    ], text, flags=re.IGNORECASE | re.MULTILINE)

    if not name:
        # Scan lines for the first proper-cased full name (2–4 words),
        # also handling titles like Dr., Mr., Ms.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if _is_header(line):
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

    if not name:
        # PDF fallback: scan for a name-like pattern anywhere in the first
        # 800 chars — PDFs often don't have clean line breaks at the top.
        # Look for 2-3 capitalised words NOT preceded by common label words.
        for m in re.finditer(
            r"(?<![:\w])([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2})(?!\s*:)",
            text[:800],
        ):
            candidate = m.group(1).strip()
            words = candidate.split()
            # Must be 2-3 words, not a heading keyword
            heading_kw = re.compile(
                r"\b(skills?|experience|education|summary|profile|technical|"
                r"professional|employment|history|core|certifications?|"
                r"projects?|achievements?|resume|curriculum|vitae|objective)\b",
                re.IGNORECASE,
            )
            if 2 <= len(words) <= 3 and not heading_kw.search(candidate):
                name = candidate
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
        r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience[:\s]+(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s+(?:in|of)\s+\w+",
        r"(\d+)\+?\s*yrs?\s+(?:of\s+)?(?:professional\s+)?experience",
        # Handle PDF extraction artifacts like "5+ years" anywhere
        r"(\d+)\s*\+\s*years?",
    ], text))

    # LinkedIn profile URL
    linkedin = _first_match([
        r"linkedin\.com/in/([\w\-]+)",
        r"linkedin[:\s]+([\w\-./]+)",
    ], text)
    if linkedin and not linkedin.startswith("linkedin.com"):
        linkedin = f"linkedin.com/in/{linkedin}"

    # GitHub profile URL
    github = _first_match([
        r"github\.com/([\w\-]+)",
        r"github[:\s]+([\w\-./]+)",
    ], text)
    if github and not github.startswith("github.com"):
        github = f"github.com/{github}"

    # Top skills — grab the skills section content
    skills = None
    skills_match = re.search(
        r"(?:core\s+technical\s+skills?|technical\s+skills?|skills?)[:\s]*\n((?:.+\n?){1,10})",
        text,
        re.IGNORECASE,
    )
    if skills_match:
        raw_skills = skills_match.group(1).strip()
        # Flatten to a comma-separated list, removing bullets/dashes
        skill_items = re.split(r"[\n,|•\-–]+", raw_skills)
        skill_items = [s.strip() for s in skill_items if s.strip() and len(s.strip()) > 1]
        if skill_items:
            skills = ", ".join(skill_items[:10])  # cap at 10 items

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "experience_years": experience_years,
        "linkedin": linkedin,
        "github": github,
        "skills": skills,
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
