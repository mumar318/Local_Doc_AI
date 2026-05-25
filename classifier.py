"""
classifier.py
-------------
Rule-based document classifier using weighted keyword scoring.

Each category has a set of strong and weak signal patterns.  The category
with the highest cumulative score wins.  A minimum threshold must be met;
otherwise the document is labelled "Unclassifiable".

No external ML model is required — this keeps the classifier fast and fully
offline while still being robust to varied document layouts.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword signal definitions
# ---------------------------------------------------------------------------
# Each entry is (compiled_regex, weight).
# Positive weights add evidence; the category with the highest total wins.

_INVOICE_SIGNALS = [
    (re.compile(r"\binvoice\b", re.I), 5),
    (re.compile(r"\binvoice[\s_#-]?(?:no|number|num|#)\b", re.I), 6),
    (re.compile(r"\bbill\s+to\b", re.I), 4),
    (re.compile(r"\bship\s+to\b", re.I), 2),
    (re.compile(r"\bpurchase\s+order\b", re.I), 3),
    (re.compile(r"\bsubtotal\b", re.I), 3),
    (re.compile(r"\btax\b", re.I), 2),
    (re.compile(r"\btotal\s+(?:amount|due|payable)\b", re.I), 4),
    (re.compile(r"\bpayment\s+(?:terms|due|method)\b", re.I), 3),
    (re.compile(r"\bdue\s+date\b", re.I), 2),
    (re.compile(r"\bremit\s+to\b", re.I), 3),
    (re.compile(r"\bquantity\b", re.I), 2),
    (re.compile(r"\bunit\s+price\b", re.I), 3),
    (re.compile(r"\bvat\b", re.I), 2),
    (re.compile(r"\bINV[-\s]?\d+", re.I), 5),
]

_RESUME_SIGNALS = [
    (re.compile(r"\bresume\b", re.I), 5),
    (re.compile(r"\bcurriculum\s+vitae\b", re.I), 6),
    (re.compile(r"\bc\.?v\.?\b", re.I), 3),
    (re.compile(r"\bwork\s+experience\b", re.I), 5),
    (re.compile(r"\bprofessional\s+experience\b", re.I), 5),
    (re.compile(r"\bemployment\s+history\b", re.I), 5),
    (re.compile(r"\beducation\b", re.I), 3),
    (re.compile(r"\bskills?\b", re.I), 2),
    (re.compile(r"\bcertification\b", re.I), 2),
    (re.compile(r"\bobjective\b", re.I), 2),
    (re.compile(r"\bsummary\b", re.I), 1),
    (re.compile(r"\breferences?\b", re.I), 2),
    (re.compile(r"\blinkedin\.com\b", re.I), 4),
    (re.compile(r"\bgithub\.com\b", re.I), 3),
    (re.compile(r"\b\d+\s+years?\s+(?:of\s+)?experience\b", re.I), 4),
    (re.compile(r"\bproficiency\b", re.I), 2),
    (re.compile(r"\bresponsibilities\b", re.I), 3),
]

_UTILITY_SIGNALS = [
    (re.compile(r"\butility\s+bill\b", re.I), 6),
    (re.compile(r"\belectric(?:ity)?\s+bill\b", re.I), 6),
    (re.compile(r"\bgas\s+bill\b", re.I), 5),
    (re.compile(r"\bwater\s+bill\b", re.I), 5),
    (re.compile(r"\bkwh\b", re.I), 6),
    (re.compile(r"\bkilowatt[\s-]?hours?\b", re.I), 5),
    (re.compile(r"\benergy\s+usage\b", re.I), 4),
    (re.compile(r"\bcurrent\s+(?:meter\s+)?reading\b", re.I), 4),
    (re.compile(r"\bprevious\s+(?:meter\s+)?reading\b", re.I), 4),
    (re.compile(r"\baccount\s+(?:no|number|#)\b", re.I), 3),
    (re.compile(r"\bservice\s+address\b", re.I), 3),
    (re.compile(r"\bbilling\s+period\b", re.I), 4),
    (re.compile(r"\bamount\s+due\b", re.I), 3),
    (re.compile(r"\bpay\s+by\b", re.I), 2),
    (re.compile(r"\btariff\b", re.I), 3),
    (re.compile(r"\btherm\b", re.I), 4),
    (re.compile(r"\bcubic\s+(?:feet|meters?)\b", re.I), 4),
]

# Minimum total score required to assign a category (avoids false positives)
_MIN_SCORE = 5

# Minimum text length to attempt classification
_MIN_TEXT_LENGTH = 30


def _score(text: str, signals: list) -> int:
    """Sum the weights of all matching signals in *text*."""
    total = 0
    for pattern, weight in signals:
        if pattern.search(text):
            total += weight
    return total


def classify_document(text: str) -> str:
    """
    Classify *text* into one of:
      Invoice | Resume | Utility Bill | Other | Unclassifiable

    Returns the category name as a string.
    """
    if len(text.strip()) < _MIN_TEXT_LENGTH:
        return "Unclassifiable"

    scores = {
        "Invoice": _score(text, _INVOICE_SIGNALS),
        "Resume": _score(text, _RESUME_SIGNALS),
        "Utility Bill": _score(text, _UTILITY_SIGNALS),
    }

    best_class = max(scores, key=scores.__getitem__)
    best_score = scores[best_class]

    if best_score < _MIN_SCORE:
        return "Other"

    return best_class
