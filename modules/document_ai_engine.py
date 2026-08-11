from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence


CLAUSE_ID_RE = re.compile(r"\b([ABC]\d+(?:\.\d+)?)\b")
KEYS_TO_CHECK = [
    "Author",
    "PDF Producer",
    "PDF Application",
    "Created Date",
    "Modified Date",
    "Title",
    "Subject",
    "Keywords",
    "Vendor ID",
    "Reference Name",
    "Number of Pages",
]


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _lower(text: Any) -> str:
    return _norm(text).lower()


def _extract_clause_ids(policy_context: Optional[str]) -> List[str]:
    if not policy_context:
        return []
    ids = []
    seen = set()
    for match in CLAUSE_ID_RE.findall(policy_context):
        clause_id = match.upper()
        if clause_id not in seen:
            seen.add(clause_id)
            ids.append(clause_id)
    return ids


def _field(doc: Dict[str, Any], *names: str) -> str:
    for name in names:
        if name in doc and doc.get(name) not in (None, ""):
            return _norm(doc.get(name))
    return ""


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _make_signal(signal: str, evidence: str, why_it_matters: str) -> Dict[str, str]:
    return {
        "indicator": signal,
        "evidence": evidence,
        "why_it_matters": why_it_matters,
    }


def _collect_text(doc: Dict[str, Any]) -> str:
    for key in [
        "Extracted Text",
        "ExtractedText",
        "Text",
        "text",
        "Content",
        "content",
        "Body",
        "body",
    ]:
        value = doc.get(key)
        if value:
            return _norm(value)
    return ""


def _find_repeated_phrases(text_a: str, text_b: str) -> List[str]:
    if not text_a or not text_b:
        return []

    phrases_a = [p.strip() for p in re.split(r"[\n\r\.]+", text_a) if len(p.split()) >= 6]
    phrases_b = [p.strip() for p in re.split(r"[\n\r\.]+", text_b) if len(p.split()) >= 6]
    counts = Counter(phrases_a + phrases_b)
    repeated = [phrase for phrase, count in counts.items() if count > 1]
    return repeated[:5]


def analyze_document_authorship(
    doc_a: Dict[str, Any],
    doc_b: Dict[str, Any],
    case_id: Optional[str] = None,
    policy_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Heuristic authorship / falsification screening helper.

    Returns the shape expected by the Streamlit UI and keeps the policy grounding
    in the explanation and recommendation text.
    """
    author_a = _field(doc_a, "Author")
    author_b = _field(doc_b, "Author")
    producer_a = _field(doc_a, "PDF Producer", "Producer")
    producer_b = _field(doc_b, "PDF Producer", "Producer")
    app_a = _field(doc_a, "PDF Application", "Application")
    app_b = _field(doc_b, "PDF Application", "Application")
    created_a = _field(doc_a, "Created Date", "CreationDate")
    created_b = _field(doc_b, "Created Date", "CreationDate")
    modified_a = _field(doc_a, "Modified Date", "ModDate")
    modified_b = _field(doc_b, "Modified Date", "ModDate")
    title_a = _field(doc_a, "Title")
    title_b = _field(doc_b, "Title")
    subject_a = _field(doc_a, "Subject")
    subject_b = _field(doc_b, "Subject")
    pages_a = _field(doc_a, "Number of Pages", "Pages")
    pages_b = _field(doc_b, "Number of Pages", "Pages")

    text_a = _collect_text(doc_a)
    text_b = _collect_text(doc_b)

    indicators: List[Dict[str, str]] = []
    score = 0.0

    def add_signal(label: str, evidence: str, why: str, weight: float) -> None:
        nonlocal score
        indicators.append(_make_signal(label, evidence, why))
        score += weight

    if author_a and author_b and _lower(author_a) == _lower(author_b):
        add_signal(
            "Matching author metadata",
            f"Author is the same in both files: {author_a}",
            "Matching author metadata can indicate the same preparer or a shared export workflow.",
            0.30,
        )
    elif author_a and author_b and _string_similarity(author_a, author_b) >= 0.8:
        add_signal(
            "Similar author metadata",
            f"Author values are similar: '{author_a}' vs '{author_b}'",
            "Similar author metadata can be a soft indicator of a common preparation source.",
            0.20,
        )

    if producer_a and producer_b and _lower(producer_a) == _lower(producer_b):
        add_signal(
            "Matching PDF producer",
            f"PDF Producer is the same in both files: {producer_a}",
            "A shared producer can mean the documents passed through the same generation path.",
            0.20,
        )

    if app_a and app_b and _lower(app_a) == _lower(app_b):
        add_signal(
            "Matching PDF application",
            f"PDF Application is the same in both files: {app_a}",
            "The same export application can support a shared preparation workflow.",
            0.10,
        )

    if created_a and created_b and _lower(created_a) == _lower(created_b):
        add_signal(
            "Matching creation timestamp",
            f"Created Date matches: {created_a}",
            "The same creation timestamp can be consistent with batch generation or a shared source file.",
            0.10,
        )

    if modified_a and modified_b and _lower(modified_a) == _lower(modified_b):
        add_signal(
            "Matching modification timestamp",
            f"Modified Date matches: {modified_a}",
            "Shared modification timing can be a weak but useful corroborating clue.",
            0.05,
        )

    if title_a and title_b and _lower(title_a) == _lower(title_b):
        add_signal(
            "Matching title",
            f"Title is the same in both files: {title_a}",
            "Repeated titles can be innocuous, but they help confirm whether the files came from the same template family.",
            0.05,
        )

    if subject_a and subject_b and _lower(subject_a) == _lower(subject_b):
        add_signal(
            "Matching subject",
            f"Subject is the same in both files: {subject_a}",
            "Shared subject values are weak evidence, but they can support a template-reuse hypothesis.",
            0.05,
        )

    if pages_a and pages_b and _lower(pages_a) == _lower(pages_b):
        add_signal(
            "Matching page count",
            f"Number of pages matches: {pages_a}",
            "Page count alone is weak, but it can support a broader similarity pattern.",
            0.03,
        )

    text_sim = _string_similarity(text_a, text_b)
    if text_a and text_b and text_sim >= 0.72:
        add_signal(
            "High text similarity",
            f"Text similarity score: {text_sim:.2f}",
            "Substantial textual overlap can indicate template reuse, copy-paste behavior, or a shared preparer.",
            0.20,
        )
    elif text_a and text_b and text_sim >= 0.50:
        add_signal(
            "Moderate text similarity",
            f"Text similarity score: {text_sim:.2f}",
            "Moderate overlap is worth review when it appears alongside metadata matches.",
            0.10,
        )

    repeated_phrases = _find_repeated_phrases(text_a, text_b)
    if repeated_phrases:
        add_signal(
            "Repeated wording",
            "Several long phrases appear in both documents.",
            "Repeated wording may suggest shared drafting, a reused template, or an edited copy of an earlier file.",
            0.08,
        )

    clause_ids = _extract_clause_ids(policy_context)
    b_clauses = [c for c in clause_ids if c.startswith("B")]

    # Keep the band conservative and easy to explain.
    if score >= 0.65:
        same_preparer_likelihood = "high"
    elif score >= 0.32:
        same_preparer_likelihood = "medium"
    elif score > 0.0:
        same_preparer_likelihood = "low"
    else:
        same_preparer_likelihood = "unknown"

    if score >= 0.65:
        confidence = "high"
        risk_band = "High"
    elif score >= 0.32:
        confidence = "medium"
        risk_band = "Medium"
    else:
        confidence = "low"
        risk_band = "Low"

    if b_clauses:
        applicable_clause_text = ", ".join(b_clauses)
    else:
        applicable_clause_text = "B3 / B4 / B5" if risk_band != "Low" else "B5"

    explanation = (
        f"The current screening indicates a {same_preparer_likelihood} likelihood that the two files were prepared from the same or a closely related source. "
        f"Observed indicators include {', '.join(i['indicator'] for i in indicators[:3]) if indicators else 'no strong indicators'}. "
        f"This assessment is grounded in the retrieved policy clauses and should be treated as a screening aid only."
    )

    if risk_band == "High":
        recommendation = (
            f"Refer the case for human review under clause(s) {applicable_clause_text}; preserve the original files and full metadata before any further action."
        )
    elif risk_band == "Medium":
        recommendation = (
            f"Request the original files or a clarification from the submitter before proceeding, consistent with clause(s) {applicable_clause_text}."
        )
    else:
        recommendation = "No escalation is recommended on the current evidence alone; continue screening and record the result for reference."

    caution = (
        "Metadata and text similarity are indicators, not proof. Missing metadata can be caused by normal scan/export workflows, and the absence of a red flag is not proof of authenticity."
    )

    return {
        "case_id": case_id,
        "same_preparer_likelihood": same_preparer_likelihood,
        "confidence": confidence,
        "risk_band": risk_band,
        "explanation": explanation,
        "recommendation": recommendation,
        "caution": caution,
        "signals": indicators or [
            _make_signal(
                "Limited evidence",
                "No strong metadata or text indicators were available from the current inputs.",
                "Limited evidence lowers confidence and should be disclosed clearly in the review record.",
            )
        ],
        "repeated_phrases": repeated_phrases,
        "style_observations": [
            "The assessment is conservative and should be reviewed by a human before any procurement decision.",
            f"Policy grounding references clause(s) {applicable_clause_text}.",
        ],
        "policy_clauses_used": b_clauses or ["B5"],
    }