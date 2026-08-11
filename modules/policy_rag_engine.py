from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Iterable

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CLAUSE_RE = re.compile(
    r"^(?P<clause>[ABC]\d+(?:\.\d+)?)\b[\s:.\-–—]*(?P<title>.*)$",
    re.IGNORECASE,
)


@dataclass
class PolicyClause:
    clause_id: str
    title: str
    text: str
    topic: str
    source_doc: str
    version: str
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _topic_from_clause(clause_id: str) -> str:
    clause_id = (clause_id or "").upper()
    
    # NEW FILTER: Catches AI templates in A6, A7, B5, plus all of Part C
    if clause_id in ["A6", "A7", "B5"] or clause_id.startswith("C"):
        return "technical"
        
    if clause_id.startswith("A"):
        return "coi"
    if clause_id.startswith("B"):
        return "falsification"
        
    return "general"


def iter_block_items(parent: _Document) -> Iterable[Any]:
    """
    Yield paragraphs and tables in document order.

    This is useful because policy clauses may appear in tables as well as
    regular paragraphs.
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element

    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def load_policy_clauses(
    docx_path: str,
    source_doc: str = "Procurement Shield Policy KB Pack",
    version: str = "Draft v0.1",
    last_updated: str = "2026-08-04",
) -> List[PolicyClause]:
    """
    Parse the policy DOCX into clause-level chunks.

    Expected clause headings:
      A1. Definition
      A5.1 Escalation Matrix
      B4. Risk Rating Framework

    Each clause becomes one retrievable chunk.
    """
    doc = Document(docx_path)

    clauses: List[PolicyClause] = []
    current_clause_id: Optional[str] = None
    current_title: str = ""
    current_parts: List[str] = []

    def flush_current_clause() -> None:
        nonlocal current_clause_id, current_title, current_parts
        if not current_clause_id:
            return

        body = _clean(" ".join(current_parts))
        if not body:
            return

        topic = _topic_from_clause(current_clause_id)
        
        # FILTER: Prevent technical AI guidelines from reaching the human user
        if topic == "technical":
            return

        clauses.append(
            PolicyClause(
                clause_id=current_clause_id,
                title=current_title or current_clause_id,
                text=f"{current_clause_id} {current_title}\n{body}".strip(),
                topic=topic,
                source_doc=source_doc,
                version=version,
                last_updated=last_updated,
            )
        )

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = _clean(block.text)
            if not text:
                continue

            match = CLAUSE_RE.match(text)
            if match:
                # New clause starts; flush the previous one first.
                flush_current_clause()
                current_clause_id = match.group("clause").upper()
                current_title = _clean(match.group("title"))
                current_parts = []
            else:
                if current_clause_id:
                    current_parts.append(text)

        elif isinstance(block, Table):
            table_lines = []
            for row in block.rows:
                cells = [_clean(cell.text) for cell in row.cells]
                cells = [c for c in cells if c]
                if cells:
                    table_lines.append(" | ".join(cells))
            if current_clause_id and table_lines:
                current_parts.append(" ".join(table_lines))

    flush_current_clause()
    return clauses


class PolicyRetriever:
    """
    Simple baseline retriever using TF-IDF + cosine similarity.

    This is a good first step before upgrading to hybrid retrieval
    (keyword + embeddings + reranking).
    """

    def __init__(self, clauses: List[PolicyClause]):
        self.clauses = clauses
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )

        if clauses:
            self.matrix = self.vectorizer.fit_transform([c.text for c in clauses])
        else:
            self.matrix = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top matching clauses as dictionaries with similarity scores.
        """
        query = (query or "").strip()
        if not query or not self.clauses or self.matrix is None:
            return []

        candidate_indices = list(range(len(self.clauses)))

        if topic:
            topic = topic.strip().lower()
            candidate_indices = [
                i for i, clause in enumerate(self.clauses)
                if clause.topic == topic
            ]

        if not candidate_indices:
            return []

        sub_matrix = self.matrix[candidate_indices]
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, sub_matrix).flatten()

        ranked = sorted(
            zip(candidate_indices, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            clause = self.clauses[idx].to_dict()
            clause["score"] = float(score)
            results.append(clause)

        return results


def format_policy_context(retrieved_clauses: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved clauses into a prompt-ready text block.
    """
    if not retrieved_clauses:
        return "No relevant policy clause was retrieved."

    blocks: List[str] = []
    for c in retrieved_clauses:
        blocks.append(
            f"[{c.get('clause_id', '')}] {c.get('title', '')} "
            f"(topic={c.get('topic', '')}, version={c.get('version', '')}, "
            f"score={float(c.get('score', 0.0)):.3f})\n"
            f"{c.get('text', '')}"
        )

    return "\n\n---\n\n".join(blocks)


def build_case_policy_query(
    supplier_a: str = "",
    supplier_b: str = "",
    extra_terms: Optional[List[str]] = None,
) -> str:
    """
    Create a simple query string from case context.
    """
    terms = [
        supplier_a or "",
        supplier_b or "",
        "procurement integrity screening",
        "possible conflict of interest",
        "possible falsification",
        "metadata similarity",
    ]
    if extra_terms:
        terms.extend(extra_terms)

    terms = [t.strip() for t in terms if t and t.strip()]
    return " ".join(terms)