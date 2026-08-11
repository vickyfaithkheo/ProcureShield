import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import inspect

from modules.forensics_engine import extract_pdf_info_from_upload
from modules.comparison_engine import build_pairwise_similarity_table
from modules.risk_engine import derive_risk_flags, add_overall_risk_score
from modules.document_ai_engine import analyze_document_authorship
from modules.web_search_engine import search_public_web
from modules.relationship_ai_engine import assess_relationship_with_search
from modules.policy_rag_engine import (
    load_policy_clauses,
    PolicyRetriever,
    format_policy_context,
    build_case_policy_query,
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ProcureShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #fbf8ff;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .procure-header {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #ffffff 0%, #f6f0ff 100%);
        border: 1px solid #ece4ff;
        box-shadow: 0 8px 22px rgba(61, 35, 102, 0.06);
        margin-bottom: 1rem;
    }

    .procure-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #24113f;
    }

    .procure-subtitle {
        font-size: 1.05rem;
        color: #6b5a84;
        margin-bottom: 0.65rem;
    }

    .procure-card {
        background: white;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #eee7fa;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
    }

    .procure-chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #f4edff;
        color: #6b2bd9;
        font-weight: 700;
        font-size: 0.88rem;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 700;
        border: 0;
        background: #ff4b4b;
        color: white;
    }

    .stMetric {
        background-color: white;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #eee7fa;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }

    .small-muted {
        color: #7a6d90;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session State Initialization
# -----------------------------
def init_session_state():
    defaults = {
        "case_id": f"PS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "uploaded_files": None,
        "pdf_df": None,
        "pairwise_df": None,
        "similarity_matrix": None,
        "analysis_ran": False,
        "relationship_ai": None,
        "document_ai": None,
        "relationship_web_evidence": {"a": [], "b": []},
        "policy_clauses": None,
        "retrieved_policy_clauses": [],
        "policy_version_used": None,
        "ai_insights_generated": False,  # Flag to control UI state and prevent tab resets
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -----------------------------
# Simple Assignment Login
# -----------------------------
USERS = {
    "Admin": {"password": "admin123", "role": "Admin"},
    "User": {"password": "user123", "role": "User"},
}

def require_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    if st.session_state.authenticated:
        return

    st.markdown(
        """
        <div class="procure-header" style="max-width:650px; margin:70px auto 20px auto; text-align:center;">
            <div class="procure-title">🛡 ProcureShield</div>
            <div class="procure-subtitle">Procurement Integrity Investigation Portal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.subheader("Sign in")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", use_container_width=True):
            account = USERS.get(username)
            if account and password == account["password"]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = account["role"]
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Demo accounts: Admin | User")
    st.stop()

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.rerun()

require_login()

# Callback to reset AI insights if the user changes the dropdown selection
def reset_ai_state():
    st.session_state.ai_insights_generated = False
    st.session_state.relationship_ai = None
    st.session_state.document_ai = None
    st.session_state.retrieved_policy_clauses = []

# -----------------------------
# Performance Optimized Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def build_metadata_comparison(doc_a, doc_b):
    fields = [
        ("Author", "Identity", "High", "Repeated authoring patterns across vendors may be notable."),
        ("PDF Application", "Software", "High", "Same creator application may suggest a similar preparation workflow."),
        ("PDF Producer", "Software", "High", "Same producer can indicate the same PDF generation path."),
        ("Created Date", "Timestamp", "Medium", "Similar timestamps may support a similar preparation timeline."),
        ("Modified Date", "Timestamp", "Medium", "Differences here may indicate later editing or handling."),
        ("Title", "Document", "Low", "Titles may be generic, but repeated titles can still be useful."),
        ("Subject", "Document", "Low", "Subject fields are often blank, but are still worth checking."),
        ("Keywords", "Content", "Low", "Keywords are optional, but repeated values can help support a pattern."),
        ("Vendor ID", "Identity", "Medium", "Useful for confirming whether the same identifier appears elsewhere."),
        ("Reference Name", "Identity", "Medium", "Can reveal whether the same reference pattern is reused."),
        ("Number of Pages", "Structure", "Low", "Page count alone is weak, but useful as supporting context."),
    ]

    rows = []
    for field, group, weight, why_it_matters in fields:
        a = str(doc_a.get(field, "")).strip() if pd.notna(doc_a.get(field)) else ""
        b = str(doc_b.get(field, "")).strip() if pd.notna(doc_b.get(field)) else ""

        if not a and not b:
            status, note = "MISSING", "No value found in either document."
        elif a == b:
            status, note = "MATCH", "Same value in both documents."
        elif not a or not b:
            status, note = "PARTIAL", "Present in one document only."
        else:
            status, note = "VARIANT", "Different values detected."

        rows.append({
            "Group": group, "Field": field, "Document A": a or "—", "Document B": b or "—",
            "Status": status, "Weight": weight, "Why it matters": why_it_matters, "Note": note,
        })

    meta_df = pd.DataFrame(rows)
    weight_order = {"High": 0, "Medium": 1, "Low": 2}
    status_order = {"VARIANT": 0, "PARTIAL": 1, "MISSING": 2, "MATCH": 3}
    meta_df["_w"] = meta_df["Weight"].map(weight_order)
    meta_df["_s"] = meta_df["Status"].map(status_order)
    return meta_df.sort_values(by=["_w", "_s", "Field"]).drop(columns=["_w", "_s"])

@st.cache_resource(show_spinner=False)
def init_policy_retriever():
    """Load the policy DOCX once and cache the retriever instance globally."""
    policy_path = "policy/Procurement_Shield_Policy_KB_Pack 1.docx"
    clauses = load_policy_clauses(
        docx_path=policy_path,
        source_doc="Procurement Shield Policy KB Pack",
        version="Draft v0.1",
        last_updated="2026-08-04",
    )
    return PolicyRetriever(clauses)

# -----------------------------
# Processing Functions
# -----------------------------
def process_uploaded_files(files):
    extracts = [x for x in (extract_pdf_info_from_upload(f) for f in files) if x]
    if not extracts:
        raise ValueError("No readable PDF files uploaded.")

    pdf_df = pd.DataFrame(extracts)
    pdf_df = derive_risk_flags(pdf_df)
    pairwise_df = build_pairwise_similarity_table(pdf_df, threshold=0.75)
    pdf_df = add_overall_risk_score(pdf_df, pairwise_df)

    return pdf_df, pairwise_df

def build_similarity_matrix(pdf_df, pairwise_df):
    if pdf_df is None or pdf_df.empty:
        return pd.DataFrame()
    names = pdf_df["Filename"].tolist()
    matrix = pd.DataFrame(1.0, index=names, columns=names)

    if pairwise_df is not None and not pairwise_df.empty:
        for _, row in pairwise_df.iterrows():
            a, b, s = row["File A"], row["File B"], float(row["Similarity"])
            if a in matrix.index and b in matrix.columns:
                matrix.loc[a, b] = matrix.loc[b, a] = s
    return matrix

def style_metadata_weight(meta_df):
    weight_styles = {
        "High": "background-color: #ffe1e1; color: #b30000; font-weight: 700;",
        "Medium": "background-color: #fff3d6; color: #b36b00; font-weight: 700;",
        "Low": "background-color: #eaf7ee; color: #1a7a3c; font-weight: 700;",
    }
    return meta_df.style.map(lambda val: weight_styles.get(val, ""), subset=["Weight"])

def build_similarity_gauge(similarity_index, threshold=0.75):
    value = round(similarity_index * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number", value=value,
            number={"suffix": "%", "font": {"size": 40, "color": "#24113f"}},
            title={"text": "Overall Similarity Index", "font": {"size": 16, "color": "#6b5a84"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#c9bce6"},
                "bar": {"color": "#6b2bd9"}, "bgcolor": "white", "borderwidth": 1, "bordercolor": "#eee7fa",
                "steps": [
                    {"range": [0, 50], "color": "#eaf7ee"},
                    {"range": [50, threshold * 100], "color": "#fff3d6"},
                    {"range": [threshold * 100, 100], "color": "#ffe1e1"},
                ],
                "threshold": {"line": {"color": "#ff4b4b", "width": 3}, "thickness": 0.85, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=25, r=25, t=55, b=15), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#24113f"})
    return fig

def render_indicator_list(title, items):
    st.markdown(f"### {title}")
    if not items:
        st.info("No indicators returned.")
        return
    for item in items:
        label = item.get("indicator") or item.get("signal") or "Indicator"
        st.markdown(
            f"""
            <div class="procure-card">
                <b>{label}</b><br>
                <span class="small-muted">{item.get('evidence', '')}</span><br><br>
                <span><b>Why it matters:</b> {item.get('why_it_matters', '')}</span>
            </div>
            """, unsafe_allow_html=True
        )

def get_supplier_label(doc):
    return doc.get("Supplier Name", "") or doc.get("Vendor Name", "") or doc.get("Reference Name", "") or doc.get("Vendor ID", "") or doc.get("Filename", "")

def dedupe_results(items):
    seen, out = set(), []
    for item in items:
        key = (str(item.get("title", "")).strip().lower(), str(item.get("url", "")).strip().lower())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

def call_ai_with_optional_policy(fn, *args, policy_context=None, **kwargs):
    try:
        sig = inspect.signature(fn)
        if "policy_context" in sig.parameters:
            return fn(*args, policy_context=policy_context, **kwargs)
    except Exception:
        pass
    return fn(*args, **kwargs)


# -----------------------------
# Documentation Pages
# -----------------------------
def render_about_us():
    st.markdown(
        '<div class="procure-header">'
        '<div class="procure-title">📘 About Us</div>'
        '<div class="procure-subtitle">Project scope, objectives, data sources and features</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## Project Overview")
    st.markdown(
        "**ProcureShield** is an AI-assisted procurement integrity screening platform "
        "designed to support procurement officers during quotation evaluation.\n\n"
        "The application provides a structured investigation workspace where users can "
        "upload quotation PDFs, inspect document metadata, compare quotations, identify "
        "similarity and risk signals, augment the investigation with public-web evidence, "
        "retrieve relevant procurement policies, and generate AI-assisted findings and "
        "an investigation report."
    )

    st.markdown("## Project Scope")
    st.markdown(
        "### In Scope\n"
        "- Upload and analyse multiple quotation PDFs.\n"
        "- Extract document-level forensic and metadata information.\n"
        "- Compare quotation metadata across documents.\n"
        "- Calculate pairwise quotation similarity.\n"
        "- Derive risk flags and an overall risk classification.\n"
        "- Retrieve relevant procurement-policy clauses.\n"
        "- Search public-web information related to selected suppliers.\n"
        "- Generate relationship and document-authorship intelligence.\n"
        "- Compile and download an investigation report.\n"
        "- Maintain a case-oriented investigation workspace and archive view."
    )

    st.markdown("## Project Objectives")
    objectives = [
        ("1", "Detect potential integrity signals", "Identify unusual document metadata, quotation similarities and other indicators that may warrant human review."),
        ("2", "Improve investigation efficiency", "Bring document forensics, comparison, external evidence and policy retrieval into a single workflow."),
        ("3", "Support evidence-based review", "Present comparison signals and policy context so investigators can understand why a case was flagged."),
        ("4", "Assist—not replace—human judgement", "Use AI to structure and explain findings while retaining human review and escalation as the decision point."),
        ("5", "Create an auditable output", "Compile key findings into a downloadable investigation report for follow-up and documentation."),
    ]
    for num, title, desc in objectives:
        st.markdown(
            f'<div class="procure-card"><b>{num}. {title}</b><br>'
            f'<span class="small-muted">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("## Data Sources")
    data_sources = [
        ("Quotation PDFs", "Primary user-provided investigation inputs. PDF information and metadata are extracted for comparison and risk analysis."),
        ("Procurement Policy Knowledge Base", "The local DOCX policy pack is loaded into the policy retriever and relevant clauses are retrieved for each AI investigation."),
        ("Public Web Search", "Publicly available supplier-related search results provide contextual evidence for relationship assessment."),
        ("Derived Analytical Data", "Risk flags, pairwise similarity scores, similarity matrices and overall risk classifications are calculated from uploaded documents."),
        ("AI Analysis Outputs", "The relationship and document-authorship AI engines produce structured investigation findings using documents, web evidence and policy context."),
    ]
    for title, desc in data_sources:
        st.markdown(
            f'<div class="procure-card"><b>{title}</b><br>'
            f'<span class="small-muted">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("## Key Features")
    feature_cols = st.columns(3)
    features = [
        ("📄", "PDF Forensics", "Extracts document information and metadata from uploaded quotations."),
        ("🔎", "Metadata Comparison", "Compares identity, software, timestamps, document and structural fields."),
        ("📊", "Similarity Analysis", "Builds pairwise similarity results and a similarity matrix."),
        ("🚩", "Risk Screening", "Derives risk flags and an overall risk classification."),
        ("🌐", "Public-Web Evidence", "Searches public sources for supplier-related contextual evidence."),
        ("📘", "Policy RAG", "Retrieves relevant procurement-policy clauses for the investigation."),
        ("🤖", "AI Intelligence", "Assesses supplier relationships and document authorship indicators."),
        ("📝", "Investigation Report", "Compiles findings into a downloadable text report."),
        ("🗂️", "Case Workspace", "Organises analysis around a generated case ID and archive view."),
    ]
    for idx, (icon, title, desc) in enumerate(features):
        with feature_cols[idx % 3]:
            st.markdown(
                f'<div class="procure-card"><div style="font-size:1.5rem;">{icon}</div>'
                f'<b>{title}</b><br><span class="small-muted">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.info(
        "ProcureShield is a screening and investigation-support tool. Its outputs are "
        "intended to support human review and should not be interpreted as automatic proof "
        "of fraud, misconduct or supplier disqualification."
    )


def render_methodology():
    st.markdown(
        '<div class="procure-header">'
        '<div class="procure-title">⚙️ Methodology</div>'
        '<div class="procure-subtitle">Data flow, implementation approach and use-case flowcharts</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## 1. Overall Methodology")
    st.markdown(
        "ProcureShield follows a **multi-stage evidence enrichment pipeline**:\n\n"
        "**Input → Extraction → Analytical Screening → Comparison → Evidence Enrichment "
        "→ Policy Retrieval → AI Assessment → Human Review → Report**\n\n"
        "The first stages operate on uploaded quotation documents. The later AI stage is "
        "optional and is triggered only after the user selects two documents and requests AI insights."
    )

    st.markdown("## 2. Core Data Flow")
    core_flow = (
        'digraph G { graph [rankdir=LR, bgcolor="transparent", nodesep=0.35, ranksep=0.55]; '
        'node [shape=box, style="rounded,filled", fillcolor="#f6f0ff", color="#6b2bd9", fontname="Arial", fontsize=11]; '
        'edge [color="#7a6d90", penwidth=1.4, arrowsize=0.7]; '
        'A [label="Quotation PDFs\\nUser Upload"]; '
        'B [label="PDF Forensics\\n& Metadata Extraction"]; '
        'C [label="Risk Flags"]; '
        'D [label="Pairwise Similarity\\nThreshold = 0.75"]; '
        'E [label="Overall Risk Score\\n& Similarity Matrix"]; '
        'F [label="Human Review\\nSelect Document Pair"]; '
        'G [label="AI Investigation"]; A -> B -> C -> D -> E -> F -> G; }'
    )
    st.graphviz_chart(core_flow, use_container_width=True)

    st.markdown("## 3. Use Case 1 — Quotation Screening & Comparison")
    st.markdown(
        "The investigator uploads quotation PDFs and starts an investigation. Each readable PDF "
        "is converted into a structured record. Risk flags are derived and the documents are "
        "compared pairwise. The dashboard then provides an initial screening picture before deeper AI investigation."
    )

    uc1_flow = (
        'digraph UC1 { graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.45]; '
        'node [shape=box, style="rounded,filled", fillcolor="#f6f0ff", color="#6b2bd9", fontname="Arial", fontsize=11]; '
        'edge [color="#7a6d90", penwidth=1.4, arrowsize=0.7]; '
        'A [label="1. Upload quotation PDFs"]; '
        'B [label="2. Extract PDF information\\n& metadata"]; '
        'C [label="3. Build structured DataFrame"]; '
        'D [label="4. Derive risk flags"]; '
        'E [label="5. Build pairwise similarity table"]; '
        'F [label="6. Add overall risk score"]; '
        'G [label="7. Build similarity matrix"]; '
        'H [label="8. Display screening results"]; '
        'I [label="9. Investigator selects\\nhigh-priority document pair"]; '
        'A -> B -> C -> D -> E -> F -> G -> H -> I; }'
    )
    st.graphviz_chart(uc1_flow, use_container_width=True)

    st.markdown("### Implementation Details")
    st.markdown(
        "1. **File ingestion:** Streamlit's multi-file uploader accepts PDF quotations.\n"
        "2. **Forensics extraction:** `extract_pdf_info_from_upload()` converts each readable upload into structured document information.\n"
        "3. **Risk screening:** `derive_risk_flags()` adds document-level indicators.\n"
        "4. **Pairwise comparison:** `build_pairwise_similarity_table()` compares documents using a configured similarity threshold of **0.75**.\n"
        "5. **Risk aggregation:** `add_overall_risk_score()` combines screening signals into an overall risk classification.\n"
        "6. **Visualisation:** Plotly displays the overall similarity gauge while Streamlit tables expose document-level and pairwise results."
    )

    st.markdown("## 4. Use Case 2 — AI Intelligence Investigation")
    st.markdown(
        "This workflow begins after the investigator selects two different quotations. "
        "ProcureShield enriches the investigation using public-web supplier evidence and the "
        "local procurement-policy knowledge base. These inputs are then passed into the "
        "relationship and document-authorship AI engines."
    )

    uc2_flow = (
        'digraph UC2 { graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.45]; '
        'node [shape=box, style="rounded,filled", fillcolor="#f6f0ff", color="#6b2bd9", fontname="Arial", fontsize=11]; '
        'edge [color="#7a6d90", penwidth=1.4, arrowsize=0.7]; '
        'A [label="1. Select Document A\\n& Document B"]; '
        'B [label="2. Identify supplier labels"]; '
        'C1 [label="3A. Public Web Search"]; '
        'C2 [label="3B. Policy Knowledge Base"]; '
        'D1 [label="4A. Supplier relationship\\nassessment"]; '
        'D2 [label="4B. Document authorship\\nassessment"]; '
        'E [label="5. Combine AI findings\\nwith policy context"]; '
        'F [label="6. Display relationship,\\nconfidence & authorship findings"]; '
        'G [label="7. Compile Investigation Report"]; '
        'H [label="8. Human Review\\n& Follow-up"]; '
        'A -> B; B -> C1; B -> C2; C1 -> D1; C2 -> D1; C2 -> D2; D1 -> E; D2 -> E; E -> F -> G -> H; }'
    )
    st.graphviz_chart(uc2_flow, use_container_width=True)

    st.markdown("### AI Investigation Data Flow")
    ai_flow = (
        'digraph AIFLOW { graph [rankdir=LR, bgcolor="transparent", nodesep=0.35, ranksep=0.55]; '
        'node [shape=box, style="rounded,filled", fillcolor="#f6f0ff", color="#6b2bd9", fontname="Arial", fontsize=11]; '
        'edge [color="#7a6d90", penwidth=1.4, arrowsize=0.7]; '
        'DOC [label="Selected quotation\\nmetadata"]; WEB [label="Public web evidence"]; '
        'POL [label="Retrieved policy\\nclauses"]; REL [label="Relationship AI"]; '
        'AUTH [label="Document AI"]; OUT [label="Investigation findings"]; REP [label="Downloadable report"]; '
        'DOC -> REL; DOC -> AUTH; WEB -> REL; POL -> REL; POL -> AUTH; REL -> OUT; AUTH -> OUT; OUT -> REP; }'
    )
    st.graphviz_chart(ai_flow, use_container_width=True)

    st.markdown("## 5. Policy Retrieval Method")
    st.markdown(
        "The policy knowledge base is loaded from a local DOCX file and cached using Streamlit's "
        "resource cache. When AI investigation is requested, the system:\n\n"
        "1. Identifies the two supplier labels.\n"
        "2. Builds a case-specific policy query.\n"
        "3. Retrieves the top **4** relevant policy clauses.\n"
        "4. Formats the retrieved clauses into a policy context.\n"
        "5. Supplies that context to the AI assessment functions.\n"
        "6. Displays the applicable operational guidance to the investigator.\n\n"
        "The implementation explicitly presents policy guidance stating that the system is intended "
        "to **flag for human review rather than automatically reject or disqualify**."
    )

    st.markdown("## 6. Risk & Decision Logic")
    st.markdown(
        "- **Low Risk:** No significant high-priority screening signals are present.\n"
        "- **Medium Risk:** Medium-priority signals are present and may require procurement-lead review.\n"
        "- **High Risk:** High-risk document signals or high-similarity pairs trigger heightened review.\n\n"
        "The application is a **decision-support layer**. Final action, escalation, clarification "
        "or procurement decisions remain with the appropriate human authority."
    )

    st.markdown("## 7. Technology / Implementation Stack")
    stack = [
        ("Streamlit", "Web application framework and interactive UI."),
        ("Python", "Application logic and orchestration."),
        ("Pandas", "Structured document-analysis dataframes and tabular comparisons."),
        ("Plotly", "Similarity gauge visualisation."),
        ("PDF Forensics Engine", "Extracts structured information from uploaded quotations."),
        ("Comparison Engine", "Builds pairwise similarity results."),
        ("Risk Engine", "Derives risk flags and overall risk scores."),
        ("Web Search Engine", "Retrieves public supplier-related evidence."),
        ("Policy RAG Engine", "Loads, retrieves and formats relevant policy clauses."),
        ("Relationship / Document AI Engines", "Produces AI-assisted investigation assessments."),
    ]
    st.table(pd.DataFrame(stack, columns=["Component", "Role"]))

    st.markdown("## 8. Limitations & Human Oversight")
    st.warning(
        "The current implementation is a screening prototype. A risk score, similarity signal, "
        "web result or AI assessment should be treated as an investigative lead, not as definitive "
        "evidence. Investigators should verify source documents and follow applicable procurement escalation procedures."
    )


# -----------------------------
# Sidebar Navigation
# -----------------------------

with st.sidebar:
    st.markdown("## 🛡 ProcureShield")
    st.caption("Procurement Integrity Investigation Portal")
    st.caption(f"Signed in as: {st.session_state.username} ({st.session_state.user_role})")
    if st.button("Logout", use_container_width=True):
        logout()

    with st.expander("⚠️ Important Notice"):
        st.markdown(
            """
            **IMPORTANT NOTICE**

            This web application is developed as a proof-of-concept prototype.
            The information provided here is **NOT intended for actual usage**
            and should not be relied upon for making any decisions, especially
            those related to financial, legal, or healthcare matters.

            **Furthermore, please be aware that the LLM may generate inaccurate
            or incorrect information. You assume full responsibility for how
            you use any generated output.**

            Always consult with qualified professionals for accurate and
            personalised advice.
            """
        )

    page = st.radio(
        "Navigate",
        ["Workspace", "Archive", "About Us", "Methodology"],
        label_visibility="visible",
    )

    st.divider()

    st.subheader("System")
    st.button("Settings")
    st.button("Support")

    st.divider()

    st.markdown("### Case")
    st.write(f"**{st.session_state.case_id}**")
# -----------------------------
# Page: Workspace
# -----------------------------
if page == "Workspace":
    st.markdown(
        f"""
        <div class="procure-header">
            <div class="procure-title">🛡 ProcureShield</div>
            <div class="procure-subtitle">Procurement Integrity Investigation Portal</div>
            <div class="small-muted">Case ID: <b>{st.session_state.case_id}</b></div>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("ProcureShield is an AI-assisted procurement integrity screening platform designed to support procurement officers during quotation evaluation.")
    
    col_upload, col_start = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader("Drag & drop quotation PDFs here", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"{len(uploaded_files)} file(s) uploaded.")

    with col_start:
        st.markdown(
            """<div class="procure-card"><b>Investigation Status</b><br><br>Ready to analyse uploaded quotations and generate a screening summary.</div>""", 
            unsafe_allow_html=True
        )
        if st.button("Start Investigation", type="primary"):
            if not st.session_state.uploaded_files:
                st.warning("Please upload at least one PDF first.")
                st.stop()
            try:
                with st.spinner("Analyzing forensics..."):
                    pdf_df, pairwise_df = process_uploaded_files(st.session_state.uploaded_files)
                    st.session_state.pdf_df = pdf_df.sort_values("Filename")
                    st.session_state.pairwise_df = pairwise_df
                    st.session_state.similarity_matrix = build_similarity_matrix(st.session_state.pdf_df, pairwise_df)
                    st.session_state.analysis_ran = True
                    reset_ai_state() # Reset downstream AI states on a new run
                st.success("Investigation completed.")
            except Exception as e:
                st.error(f"Investigation failed: {e}")
                st.stop()

    if st.session_state.analysis_ran and st.session_state.pdf_df is not None:
        pdf_df, pairwise_df = st.session_state.pdf_df, st.session_state.pairwise_df
        high_risk = int((pdf_df["Overall Risk"] == "High").sum())
        med_risk = int((pdf_df["Overall Risk"] == "Medium").sum())
        high_sim = int((pairwise_df["Flag"] == "TRUE").sum()) if pairwise_df is not None and not pairwise_df.empty else 0

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents Investigated", len(pdf_df))
        c2.metric("Cases Requiring Review", high_risk)
        c3.metric("Medium Priority Cases", med_risk)
        c4.metric("High Similarity Pairs", high_sim)
        st.divider()

        # Tabs Setup
        tab_overview, tab_metadata, tab_comparison, tab_ai, tab_report = st.tabs([
            "Overview", "Metadata Comparison", "Quotation Comparison", "AI Intelligence", "Investigation Report"
        ])

        with tab_overview:
            st.subheader("Executive Snapshot")
            snap1, snap2 = st.columns([2, 1])
            with snap1:
                st.plotly_chart(build_similarity_gauge(float(pairwise_df["Similarity"].mean()) if pairwise_df is not None and not pairwise_df.empty else 0.0), use_container_width=True)
            with snap2:
                overall_risk = "High" if high_risk > 0 or high_sim > 0 else "Medium" if med_risk > 0 else "Low"
                st.markdown(f"""<div class="procure-card"><b>Overall Case Risk</b><br><br><span style="font-size: 2rem; font-weight: 800;">{overall_risk}</span></div>""", unsafe_allow_html=True)
            st.dataframe(pdf_df[["Filename", "Vendor ID", "Overall Risk", "Metadata Flags Count", "High Similarity Pair Count"]], use_container_width=True)

        with tab_metadata:
            st.subheader("Metadata Comparison")
            if len(pdf_df) < 2:
                st.info("Upload at least 2 readable PDFs to compare metadata.")
            else:
                filenames = pdf_df["Filename"].tolist()
                c_left, c_right = st.columns(2)
                with c_left: doc_a_name = st.selectbox("Document A", filenames, index=0, key="meta_a")
                with c_right: doc_b_name = st.selectbox("Document B", filenames, index=1 if len(filenames) > 1 else 0, key="meta_b")

                if doc_a_name != doc_b_name:
                    doc_a = pdf_df[pdf_df["Filename"] == doc_a_name].iloc[0].to_dict()
                    doc_b = pdf_df[pdf_df["Filename"] == doc_b_name].iloc[0].to_dict()
                    meta_df = build_metadata_comparison(doc_a, doc_b)
                    
                    match_count = int((meta_df["Status"] == "MATCH").sum())
                    variant_count = int((meta_df["Status"] == "VARIANT").sum())
                    partial_count = int((meta_df["Status"] == "PARTIAL").sum())
                    missing_count = int((meta_df["Status"] == "MISSING").sum())
                    high_value_count = int((meta_df["Weight"] == "High").sum())

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Matches", match_count)
                    m2.metric("Variants", variant_count)
                    m3.metric("Partial", partial_count)
                    m4.metric("Missing", missing_count)
                    m5.metric("High-Value Clues", high_value_count)
                    
                    st.dataframe(style_metadata_weight(meta_df), use_container_width=True, hide_index=True)

        with tab_comparison:
            st.subheader("Quotation Comparison")
            if pairwise_df is None or pairwise_df.empty:
                st.info("No pairwise comparison available yet.")
            else:
                st.dataframe(pairwise_df, use_container_width=True)
                st.subheader("Similarity Matrix")
                st.dataframe(st.session_state.similarity_matrix, use_container_width=True)

        with tab_ai:
            st.subheader("AI Intelligence")
            st.write("Generate AI-based investigation findings for a selected pair of quotations.")
            filenames = pdf_df["Filename"].tolist()

            if len(filenames) < 2:
                st.warning("At least two readable PDF files are required for AI comparison.")
            else:
                c1, c2 = st.columns(2)
                # Attach the reset callback to ensure state refreshes if inputs change
                with c1: doc_a_name = st.selectbox("Document A", filenames, index=0, key="ai_a", on_change=reset_ai_state)
                with c2: doc_b_name = st.selectbox("Document B", filenames, index=1 if len(filenames) > 1 else 0, key="ai_b", on_change=reset_ai_state)

                if doc_a_name == doc_b_name:
                    st.warning("Please choose two different documents.")
                else:
                    doc_a = pdf_df[pdf_df["Filename"] == doc_a_name].iloc[0].to_dict()
                    doc_b = pdf_df[pdf_df["Filename"] == doc_b_name].iloc[0].to_dict()

                    # Using Session State to lock the layout so tabs don't reset upon re-rendering
                    if st.button("Generate AI Insights", type="primary"):
                        st.session_state.ai_insights_generated = True
                        with st.spinner("Searching the web, retrieving policy, and generating AI investigation..."):
                            supplier_a = get_supplier_label(doc_a)
                            supplier_b = get_supplier_label(doc_b)

                            search_a = dedupe_results(search_public_web(supplier_a, max_results=3))[:8]
                            search_b = dedupe_results(search_public_web(supplier_b, max_results=3))[:8]
                            st.session_state.relationship_web_evidence = {"a": search_a, "b": search_b}

                            # Cached Policy Retrieval execution
                            policy_retriever = init_policy_retriever()
                            policy_query = build_case_policy_query(supplier_a=supplier_a, supplier_b=supplier_b)
                            retrieved = policy_retriever.retrieve(query=policy_query, top_k=4)
                            
                            st.session_state.retrieved_policy_clauses = retrieved
                            policy_context = format_policy_context(retrieved)

                            st.session_state.relationship_ai = call_ai_with_optional_policy(
                                assess_relationship_with_search, supplier_a, supplier_b, search_a, search_b, st.session_state.case_id, policy_context=policy_context
                            )
                            st.session_state.document_ai = call_ai_with_optional_policy(
                                analyze_document_authorship, doc_a, doc_b, st.session_state.case_id, policy_context=policy_context
                            )

                    # Display logic controlled purely by state, preserving the UI flow
                    if st.session_state.ai_insights_generated and st.session_state.relationship_ai:
                        
                        # Formatted Policy Output for Human Reader
                        with st.expander("📘 Applicable Procurement Policies", expanded=True):
                            retrieved = st.session_state.retrieved_policy_clauses
                            if not retrieved:
                                st.info("No human-facing operational guidelines were matched for this search.")
                            else:
                                st.markdown("The following operational policies guide this assessment:")
                                for clause in retrieved:
                                    st.markdown(f"**{clause['clause_id']} — {clause['title']}**")
                                    
                                    if clause['clause_id'] == "A5":
                                        st.markdown("""
                                        * **Role of the System:** Do NOT auto-reject or auto-disqualify. The system flags for human review only.
                                        * **Preserve Evidence Trail:** Capture specific signal(s) triggering the flag before further processing.
                                        * **Route to Approving Authority:** Route to the Approving Authority, NOT the flagged officer.
                                        * **Apply Seniority-Based Escalation:** Use the matrix to determine who decides and document the outcome.
                                        * **Recuse / Replace:** Replace the officer for that procurement action if COI is confirmed.
                                        * **Resolve on Time:** Complete the review before the next irreversible step.
                                        """)
                                    elif clause['clause_id'] == "A5.1":
                                        st.markdown("""
                                        | Seniority of Flagged Officer | Who Decides |
                                        | :--- | :--- |
                                        | **Officer / Executive** | Immediate supervisor or procurement lead |
                                        | **Manager / Senior Manager** | Department Head / Director |
                                        | **VP / Director and above** | Next higher authority (e.g., Deputy CEO or Integrity/Audit committee) |
                                        | **Fraud/Corruption Suspected** | Refer to Internal Audit / Integrity Unit |
                                        """)
                                    elif clause['clause_id'] == "B2":
                                        st.markdown("""
                                        **Document Authenticity & Metadata Check:**
                                        We categorize findings into three review levels:
                                        * **Corroborating Signals (Class A):** Raises confidence in authenticity (e.g., matching institutional producer or valid cryptographic signature).
                                        * **High-Risk Red Flags (Class B):** Serious inconsistencies, such as a file showing it was edited *after* its creation date, or an official quotation created using consumer tools (Word, Canva, etc.).
                                        * **Minor Clues (Class C):** Weak supporting notes (like generic author names). *Note: Missing data is normal if scanned/printed.*
                                        """)
                                    elif clause['clause_id'] == "B4":
                                        st.markdown("""
                                        **Risk Rating Guide:**
                                        * **Low Risk:** Only minor Class C signals; no escalation needed.
                                        * **Medium Risk:** Clear Class B signals; flag for procurement lead and request clarification.
                                        * **High Risk:** Multiple Class B signals or tampering; **Stop immediately**, preserve files, and refer to Internal Audit.
                                        """)
                                    else:
                                        cleaned = clause['text'].replace("DocInfo", "file properties").replace("XMP", "hidden audit logs")
                                        st.markdown(f"> {cleaned}")
                                        
                                    st.divider()

                        rel = st.session_state.relationship_ai
                        st.markdown("### Relationship Intelligence")
                        x1, x2, x3 = st.columns(3)
                        x1.metric("Relationship Level", str(rel.get("relationship_level", "unknown")).title())
                        x2.metric("Confidence", str(rel.get("confidence", "unknown")).title())
                        x3.metric("Same Source Likelihood", str(rel.get("same_source_likelihood", "unknown")).title())
                        
                        st.markdown(f"""<div class="procure-card"><b>Assessment</b><br><br>{rel.get("explanation", "")}</div>""", unsafe_allow_html=True)

                        if st.session_state.document_ai:
                            doc = st.session_state.document_ai
                            st.markdown("### Document AI Intelligence")
                            y1, y2 = st.columns(2)
                            y1.metric("Same Preparer Likelihood", str(doc.get("same_preparer_likelihood", "unknown")).title())
                            y2.metric("Confidence", str(doc.get("confidence", "unknown")).title())
                            st.markdown(f"""<div class="procure-card"><b>Assessment</b><br><br>{doc.get("explanation", "")}</div>""", unsafe_allow_html=True)

        with tab_report:
            st.subheader("Investigation Report")
            
            if not st.session_state.ai_insights_generated:
                st.info("Run an AI Intelligence investigation first in the 'AI Intelligence' tab to compile the report.")
            else:
                st.write("Review the compiled summary below and click to download the investigation report.")
                
                # Compile report content from current analysis
                # Get current docs from AI Intelligence selectors
                doc_a = st.session_state.pdf_df[st.session_state.pdf_df["Filename"] == st.session_state.get("ai_a", st.session_state.pdf_df["Filename"].iloc[0])].iloc[0].to_dict()
                doc_b = st.session_state.pdf_df[st.session_state.pdf_df["Filename"] == st.session_state.get("ai_b", st.session_state.pdf_df["Filename"].iloc[1] if len(st.session_state.pdf_df)>1 else st.session_state.pdf_df["Filename"].iloc[0])].iloc[0].to_dict()
                meta_df = build_metadata_comparison(doc_a, doc_b)

                report_content = f"=========================================\n"
                report_content += f"ProcureShield Investigation Report\n"
                report_content += f"Case ID: {st.session_state.case_id}\n"
                report_content += f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                report_content += f"=========================================\n\n"
                
                report_content += "--- Metadata Analysis ---\n"
                for _, row in meta_df.iterrows():
                    report_content += f"{row['Field']}: {row['Status']} - {row['Note']}\n"
                
                report_content += "\n--- Relationship Intelligence ---\n"
                if st.session_state.relationship_ai:
                    rel = st.session_state.relationship_ai
                    report_content += f"Relationship Level: {str(rel.get('relationship_level', 'N/A')).title()}\n"
                    report_content += f"Confidence: {str(rel.get('confidence', 'N/A')).title()}\n"
                    report_content += f"Assessment: {rel.get('explanation', 'N/A')}\n\n"
                else:
                    report_content += "No relationship analysis generated.\n\n"
                
                report_content += "--- Document AI Intelligence ---\n"
                if st.session_state.document_ai:
                    doc = st.session_state.document_ai
                    report_content += f"Same Preparer Likelihood: {str(doc.get('same_preparer_likelihood', 'N/A')).title()}\n"
                    report_content += f"Confidence: {str(doc.get('confidence', 'N/A')).title()}\n"
                    report_content += f"Assessment: {doc.get('explanation', 'N/A')}\n\n"
                else:
                    report_content += "No document authorship analysis generated.\n\n"

                # Preview in UI
                st.text_area("Report Preview", report_content, height=280)

                # Download Button
                st.download_button(
                    label="📥 Download Investigation Report (.txt)",
                    data=report_content,
                    file_name=f"Investigation_Report_{st.session_state.case_id}.txt",
                    mime="text/plain"
                )

# -----------------------------
# Page: About Us
# -----------------------------
elif page == "About Us":
    render_about_us()

# -----------------------------
# Page: Methodology
# -----------------------------
elif page == "Methodology":
    render_methodology()

# -----------------------------
# Page: Archive
# -----------------------------
elif page == "Archive":
    st.markdown(
        """
        <div class="procure-header">
            <div class="procure-title">📁 Investigation Archive</div>
            <div class="procure-subtitle">Case history and outputs</div>
        </div>
        """, unsafe_allow_html=True
    )
    if st.session_state.pdf_df is None:
        st.info("No investigation has been run yet.")
    else:
        st.dataframe(st.session_state.pdf_df[["Filename", "Vendor ID", "Overall Risk"]], use_container_width=True)