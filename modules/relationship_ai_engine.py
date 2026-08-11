from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set
from urllib.parse import urlparse


# ============================================================
# Configuration
# ============================================================

# Maximum lengths are used to prevent excessively large or malformed
# external inputs from entering the analysis pipeline.

MAX_SUPPLIER_NAME_LENGTH = 300
MAX_POLICY_CONTEXT_LENGTH = 8000
MAX_URL_LENGTH = 1000
MAX_SEARCH_RESULTS = 20
MAX_INDICATORS = 10


# Procurement policy clause identifier.
# Examples:
# A3
# A5
# B2.1
# C10.2

CLAUSE_ID_RE = re.compile(
    r"\b([ABC]\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


# ============================================================
# Input Sanitisation
# ============================================================

def _clean_text(text: Any, max_length: int) -> str:
    """
    Convert an arbitrary value into bounded plain text.

    This function does NOT make external content trustworthy.
    It simply prevents malformed/control-heavy input from propagating
    through the application.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove ASCII control characters except normal whitespace.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_length]


def _norm(text: Any) -> str:
    """
    Normalise text for deterministic comparisons.

    Important:
    This function is used only for analysis.
    It does not interpret or execute any text as instructions.
    """

    return _clean_text(
        text,
        MAX_SUPPLIER_NAME_LENGTH,
    ).lower()


# ============================================================
# Policy Processing
# ============================================================

def _extract_clause_ids(
    policy_context: Optional[str],
) -> List[str]:
    """
    Extract procurement policy clause IDs from policy context.

    The policy text is treated purely as reference data.
    No instructions contained inside the policy text are executed.
    """

    if not policy_context:
        return []

    policy_context = _clean_text(
        policy_context,
        MAX_POLICY_CONTEXT_LENGTH,
    )

    ids: List[str] = []
    seen: Set[str] = set()

    for match in CLAUSE_ID_RE.findall(policy_context):
        clause_id = match.upper()

        if clause_id not in seen:
            seen.add(clause_id)
            ids.append(clause_id)

    return ids


# ============================================================
# Web Search Processing
# ============================================================

def _extract_domains(
    search_results: Sequence[Dict[str, Any]],
) -> Counter:
    """
    Extract domains from search-result URLs.

    SECURITY DESIGN:
    - Search-result snippets are NOT analysed.
    - Search-result titles are NOT analysed.
    - Search-result text is NOT interpreted as instructions.
    - Only the URL hostname is used as a deterministic signal.

    This significantly reduces the prompt-injection surface because
    this module never sends arbitrary webpage text to an LLM.
    """

    domains = Counter()

    if not search_results:
        return domains

    # Bound the number of externally supplied results.
    for item in list(search_results)[:MAX_SEARCH_RESULTS]:

        if not isinstance(item, dict):
            continue

        raw_url = item.get("url", "")

        url = _clean_text(
            raw_url,
            MAX_URL_LENGTH,
        )

        if not url:
            continue

        try:
            parsed = urlparse(url)

            hostname = parsed.hostname

            if not hostname:
                continue

            hostname = hostname.lower().strip(".")

            # Remove a common "www." prefix so that:
            # www.example.com
            # example.com
            # are treated as the same domain.
            if hostname.startswith("www."):
                hostname = hostname[4:]

            # Basic hostname validation.
            if not re.fullmatch(
                r"[a-z0-9.-]+",
                hostname,
            ):
                continue

            domains[hostname] += 1

        except Exception:
            # Malformed URLs should never break the investigation.
            continue

    return domains


# ============================================================
# Supplier Name Processing
# ============================================================

NAME_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+"
)


def _tokenize_supplier(
    name: str,
) -> Set[str]:
    """
    Tokenise supplier names for deterministic similarity analysis.
    """

    name = _norm(name)

    tokens = {
        token
        for token in NAME_TOKEN_RE.findall(name)
        if len(token) > 2
    }

    stop = {
        "pte",
        "ltd",
        "limited",
        "company",
        "co",
        "corp",
        "inc",
        "the",
    }

    return {
        token
        for token in tokens
        if token not in stop
    }


# ============================================================
# Similarity Functions
# ============================================================

def _best_overlap(
    a: Set[str],
    b: Set[str],
) -> float:

    if not a or not b:
        return 0.0

    return len(a & b) / max(
        1,
        min(len(a), len(b)),
    )


def _string_similarity(
    a: str,
    b: str,
) -> float:

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def _pick_level(
    score: float,
) -> str:

    if score >= 0.72:
        return "high"

    if score >= 0.42:
        return "medium"

    if score > 0.0:
        return "low"

    return "none"


# ============================================================
# Indicator Construction
# ============================================================

def _make_indicator(
    label: str,
    evidence: str,
    why: str,
) -> Dict[str, str]:

    return {
        "indicator": _clean_text(label, 300),
        "evidence": _clean_text(evidence, 1000),
        "why_it_matters": _clean_text(why, 1000),
    }


# ============================================================
# Main Relationship Assessment
# ============================================================

def assess_relationship_with_search(
    supplier_a: str,
    supplier_b: str,
    search_results_a: Optional[
        List[Dict[str, Any]]
    ] = None,
    search_results_b: Optional[
        List[Dict[str, Any]]
    ] = None,
    case_id: Optional[str] = None,
    policy_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic relationship / conflict-of-interest screening helper.

    Security characteristics
    -------------------------
    1. No LLM is called by this function.
    2. External web content is NOT interpreted as instructions.
    3. Only web-result domains are used for web-based comparison.
    4. Input lengths are bounded.
    5. Malformed URLs are ignored safely.
    6. Policy context is treated as reference data only.
    7. Results are deterministic and reproducible.

    The output is an investigative screening signal and must not be
    interpreted as proof of fraud, collusion, corruption or conflict
    of interest.
    """

    # --------------------------------------------------------
    # Step 1 — Bound and sanitise inputs
    # --------------------------------------------------------

    supplier_a_clean = _clean_text(
        supplier_a,
        MAX_SUPPLIER_NAME_LENGTH,
    )

    supplier_b_clean = _clean_text(
        supplier_b,
        MAX_SUPPLIER_NAME_LENGTH,
    )

    case_id_clean = _clean_text(
        case_id,
        100,
    )

    policy_context_clean = _clean_text(
        policy_context,
        MAX_POLICY_CONTEXT_LENGTH,
    )

    # Only accept list-like search results.
    search_results_a = (
        search_results_a
        if isinstance(search_results_a, list)
        else []
    )

    search_results_b = (
        search_results_b
        if isinstance(search_results_b, list)
        else []
    )

    # --------------------------------------------------------
    # Step 2 — Normalise supplier names
    # --------------------------------------------------------

    a_norm = _norm(
        supplier_a_clean
    )

    b_norm = _norm(
        supplier_b_clean
    )

    a_tokens = _tokenize_supplier(
        supplier_a_clean
    )

    b_tokens = _tokenize_supplier(
        supplier_b_clean
    )

    # --------------------------------------------------------
    # Step 3 — Calculate deterministic signals
    # --------------------------------------------------------

    name_sim = _string_similarity(
        a_norm,
        b_norm,
    )

    token_overlap = _best_overlap(
        a_tokens,
        b_tokens,
    )

    a_domains = _extract_domains(
        search_results_a
    )

    b_domains = _extract_domains(
        search_results_b
    )

    domain_overlap = _best_overlap(
        set(a_domains),
        set(b_domains),
    )

    # --------------------------------------------------------
    # Step 4 — Weighted relationship score
    # --------------------------------------------------------

    # Strong signals are intentionally conservative.
    #
    # Name similarity: 35%
    # Token overlap:   35%
    # Domain overlap:  30%

    score = (
        (name_sim * 0.35)
        + (token_overlap * 0.35)
        + (domain_overlap * 0.30)
    )

    # Keep score inside expected numerical bounds.
    score = max(
        0.0,
        min(1.0, score),
    )

    # --------------------------------------------------------
    # Step 5 — Generate indicators
    # --------------------------------------------------------

    indicators: List[
        Dict[str, str]
    ] = []

    shared_tokens = (
        a_tokens & b_tokens
    )

    if shared_tokens:

        shared = ", ".join(
            sorted(shared_tokens)
        )

        indicators.append(
            _make_indicator(
                "Shared name tokens",
                f"Common tokens: {shared}",
                (
                    "Shared tokens can indicate a related legal entity, "
                    "alternate branding, or a close corporate relationship."
                ),
            )
        )

    shared_domains = (
        set(a_domains)
        & set(b_domains)
    )

    if shared_domains:

        indicators.append(
            _make_indicator(
                "Shared web footprint",
                (
                    "Overlapping domains: "
                    + ", ".join(
                        sorted(shared_domains)
                    )
                ),
                (
                    "A shared web footprint can suggest that suppliers "
                    "are connected or share a common source of content."
                ),
            )
        )

    if name_sim >= 0.80:

        indicators.append(
            _make_indicator(
                "Very similar supplier names",
                f"Name similarity score: {name_sim:.2f}",
                (
                    "Highly similar names warrant review to rule out "
                    "shared ownership, rebranding, or a related-party link."
                ),
            )
        )

    if not indicators:

        indicators.append(
            _make_indicator(
                "No strong public linkage found",
                (
                    "The current search results did not surface "
                    "an obvious direct connection."
                ),
                (
                    "Absence of a visible link is not proof of independence; "
                    "it only means that the current evidence is limited."
                ),
            )
        )

    # Hard limit in case additional indicators are added later.
    indicators = indicators[:MAX_INDICATORS]

    # --------------------------------------------------------
    # Step 6 — Determine relationship level
    # --------------------------------------------------------

    relationship_level = _pick_level(
        score
    )

    # Same-source likelihood is intentionally aligned with the
    # relationship level, but kept separate because a relationship
    # does not prove common document authorship.

    if relationship_level == "high":

        same_source_likelihood = "high"

    elif relationship_level == "medium":

        same_source_likelihood = "medium"

    elif relationship_level == "low":

        same_source_likelihood = "low"

    else:

        same_source_likelihood = "unknown"

    # --------------------------------------------------------
    # Step 7 — Determine confidence
    # --------------------------------------------------------

    if score >= 0.72:

        confidence = "high"

    elif score >= 0.42:

        confidence = "medium"

    else:

        confidence = "low"

    # --------------------------------------------------------
    # Step 8 — Extract applicable policy clauses
    # --------------------------------------------------------

    clause_ids = _extract_clause_ids(
        policy_context_clean
    )

    coi_clauses = [
        clause
        for clause in clause_ids
        if clause.startswith("A")
    ]

    if coi_clauses:

        applicable_clause_text = ", ".join(
            coi_clauses
        )

    else:

        applicable_clause_text = (
            "A3 / A5"
            if relationship_level
            in {"medium", "high"}
            else "A5"
        )

    # --------------------------------------------------------
    # Step 9 — Build explanation
    # --------------------------------------------------------

    supplier_a_display = (
        supplier_a_clean
        or "Supplier A"
    )

    supplier_b_display = (
        supplier_b_clean
        or "Supplier B"
    )

    indicator_names = ", ".join(
        indicator["indicator"]
        for indicator in indicators[:2]
    )

    explanation = (
        f"Current screening suggests a "
        f"{relationship_level} relationship signal between "
        f"{supplier_a_display} and {supplier_b_display}. "
        f"Observed evidence includes {indicator_names}. "
        f"This is a policy-grounded screening assessment, "
        f"not a finding of fact."
    )

    # --------------------------------------------------------
    # Step 10 — Recommendation
    # --------------------------------------------------------

    if relationship_level in {
        "medium",
        "high",
    }:

        recommendation = (
            "If the overlap remains material after review, "
            "route the matter to the approving authority under "
            f"clause(s) {applicable_clause_text}."
        )

    else:

        recommendation = (
            "Continue screening; no automatic escalation is "
            "recommended on the current evidence alone."
        )

    # --------------------------------------------------------
    # Step 11 — Human-review safeguard
    # --------------------------------------------------------

    caution = (
        "Do not treat this assessment as proof of a conflict of interest, "
        "fraud, collusion, corruption, or common ownership. "
        "Use the retrieved policy clauses and human review before "
        "making any procurement decision."
    )

    # --------------------------------------------------------
    # Step 12 — Return structured result
    # --------------------------------------------------------

    return {
        "case_id": case_id_clean,
        "relationship_level": relationship_level,
        "confidence": confidence,
        "same_source_likelihood": same_source_likelihood,
        "explanation": explanation,
        "recommendation": recommendation,
        "caution": caution,
        "indicators": indicators,
        "policy_clauses_used": (
            coi_clauses
            or ["A5"]
        ),
    }