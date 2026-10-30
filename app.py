"""MediLens AI — reference-matched Streamlit presentation layer.

The report-processing and Medical Q&A module contracts are preserved. This
file changes only the Streamlit presentation and the way existing results are
arranged on screen. It does not alter extraction, parsing, range checking,
Gemini requests, ChromaDB retrieval, or language handling.
"""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import importlib
import os
from math import isfinite
from pathlib import Path
from typing import Any

import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION / EXISTING APP CONSTANTS
# --------------------------------------------------
st.set_page_config(
    page_title="MediLens AI | Health Awareness",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _configure_cloud_secrets() -> None:
    """Pass Streamlit Cloud secrets to the existing modules via environment."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if api_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = str(api_key)


_configure_cloud_secrets()

LANGUAGES = {"English": "en", "Hindi": "hi", "Kannada": "kn"}

ANALYSIS_STATE_KEY = "medilens_analysis_result"
ANALYSIS_FILE_KEY = "medilens_analysis_file_hash"
CHAT_STATE_KEY = "medilens_chat_result"
PAGE_STATE_KEY = "medilens_active_page"
ASSET_DIR = Path(__file__).parent / "assets"


# --------------------------------------------------
# STYLES — visual layer only
# --------------------------------------------------
def inject_styles() -> None:
    css_path = ASSET_DIR / "medilens.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _asset_data_uri(filename: str) -> str:
    """Embed a local illustration without adding a frontend dependency."""
    asset_path = ASSET_DIR / filename
    raw = asset_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    suffix = asset_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/svg+xml"
    return f"data:{mime_type};base64,{encoded}"


def render_topbar(page_label: str, language_code: str) -> None:
    language_name = {"en": "English", "hi": "Hindi", "kn": "Kannada"}.get(language_code, "English")
    st.markdown(
        f"""
        <div class="ml-topbar">
          <div class="ml-topbar-title"><span class="ml-topbar-mark">✣</span>
            <strong>{html.escape(page_label)}</strong>
            <span class="ml-topbar-divider">|</span>
            <span class="ml-topbar-tagline">Ask. Understand. Take better care.</span>
          </div>
          <div class="ml-topbar-right">
            <span class="ml-language-pill">◎&nbsp; {html.escape(language_name)}</span>
            <span class="ml-profile-orb" aria-label="Profile">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="12" cy="8" r="3.4" stroke="#b6caff" stroke-width="1.6"/>
                <path d="M5 20c.55-3.55 3.05-5.4 7-5.4s6.45 1.85 7 5.4" stroke="#b6caff" stroke-width="1.6" stroke-linecap="round"/>
              </svg>
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, art: str, page_label: str) -> None:
    art_src = _asset_data_uri(art)
    safe_title = title
    st.markdown(
        f"""
        <section class="ml-hero">
          <div class="ml-hero-copy">
            <h1>{safe_title}</h1>
            <p>{html.escape(subtitle)}</p>
            <div class="ml-hero-meta">
              <span class="ml-hero-pill">✦ AI-assisted health awareness</span>
              <span class="ml-hero-pill">◎ English · Hindi · Kannada</span>
            </div>
          </div>
          <div class="ml-hero-art"><img src="{art_src}" alt="Healthcare AI illustration for {html.escape(page_label)}" /></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str]:
    """Reference-matched navigation, settings and project-status sections."""
    st.session_state.setdefault(PAGE_STATE_KEY, "report")
    with st.sidebar:
        st.markdown(
            """
            <div class="ml-brand">
              <div class="ml-brand-mark">
                <svg width="29" height="29" viewBox="0 0 32 32" fill="none" aria-hidden="true">
                  <path d="M8 4v8a6 6 0 0 0 12 0V4" stroke="white" stroke-width="2.2" stroke-linecap="round"/>
                  <path d="M5 4h6M17 4h6" stroke="white" stroke-width="2.2" stroke-linecap="round"/>
                  <path d="M14 18v3a7 7 0 0 0 14 0v-1" stroke="white" stroke-width="2.2" stroke-linecap="round"/>
                  <circle cx="28" cy="17" r="3" fill="#A5F3FC" stroke="white" stroke-width="1.5"/>
                  <path d="M11 4V2M20 4V2" stroke="#DDE6FF" stroke-width="2" stroke-linecap="round"/>
                </svg>
              </div>
              <div>
                <div class="ml-brand-name">MediLens <span style="color:#8da0ff">AI</span></div>
                <div class="ml-brand-sub">HEALTH INTELLIGENCE</div>
              </div>
            </div>
            <div class="ml-side-label">Workspace</div>
            """,
            unsafe_allow_html=True,
        )

        nav_items = [
            ("report", "📄  Report explainer"),
            ("qna", "💬  Medical Q&A"),
            ("about", "ℹ️  About"),
        ]
        for page, label in nav_items:
            if st.button(
                label,
                key=f"ml_nav_{page}",
                type="primary" if st.session_state[PAGE_STATE_KEY] == page else "secondary",
                use_container_width=True,
            ):
                st.session_state[PAGE_STATE_KEY] = page
                st.rerun()

        st.markdown('<div class="ml-side-label">⚙ Settings</div>', unsafe_allow_html=True)
        selected_language = st.selectbox(
            "Explanation language",
            list(LANGUAGES.keys()),
            index=0,
            key="medilens_language",
            help="The selected language is passed to the existing Gemini and Medical Q&A workflows.",
        )
        st.markdown(
            '<div class="ml-muted" style="font-size:10px;margin-top:-4px">Choose the language used by supported AI explanations and answers.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="ml-sidebar-promo">
              <div class="ml-promo-orb"><span>✦</span></div>
              <div class="ml-promo-title">Better Health<br>Through AI</div>
              <div class="ml-promo-copy">Trusted. Evidence-based.<br>Always responsible.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state[PAGE_STATE_KEY], LANGUAGES[selected_language]


# --------------------------------------------------
# EXISTING INTEGRATION HELPERS — contracts unchanged
# --------------------------------------------------
def load_function(module_name: str, function_name: str):
    """Import a project function only when its workflow is requested."""
    module = importlib.import_module(f"src.{module_name}")
    return getattr(module, function_name)


def is_module_error(result: Any) -> bool:
    return not isinstance(result, dict) or result.get("status") == "error"


def get_valid_results(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep numerical test results and exclude administrative PDF text."""
    valid_results: list[dict[str, Any]] = []

    for item in parsed.get("results", []) or []:
        if not isinstance(item, dict):
            continue

        test_name = str(item.get("test_name") or item.get("name") or "").strip()
        raw_value = item.get("value")
        unit = str(item.get("unit") or "").strip()

        try:
            numeric_value = float(str(raw_value).replace(",", "").strip())
            if not isfinite(numeric_value):
                continue
        except (ValueError, TypeError):
            continue

        if not test_name or not unit:
            continue

        cleaned = dict(item)
        cleaned["test_name"] = test_name
        cleaned["value"] = numeric_value
        cleaned["unit"] = unit
        valid_results.append(cleaned)

    return valid_results


# --------------------------------------------------
# SHARED PRESENTATION HELPERS
# --------------------------------------------------
def render_footer() -> None:
    st.markdown(
        """
        <div class="ml-footer">
          <span>© MediLens AI</span>
          <span>Clear information. Responsible AI.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _fmt_number(value: Any) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except (TypeError, ValueError):
        return "—"


def _format_reference(item: dict[str, Any]) -> str:
    low = item.get("reference_low")
    high = item.get("reference_high")
    if low is not None and high is not None:
        return f"{_fmt_number(low)} – {_fmt_number(high)}"
    if low is not None:
        return f"≥ {_fmt_number(low)}"
    if high is not None:
        return f"≤ {_fmt_number(high)}"
    return "Not provided"


def _status_pill(status: str) -> tuple[str, str]:
    mapping = {
        "within_range": ("Within range", "ml-status-normal"),
        "low": ("Low", "ml-status-low"),
        "high": ("High", "ml-status-high"),
        "unknown": ("Review", "ml-status-unknown"),
    }
    return mapping.get(status, ("Review", "ml-status-unknown"))


def render_process_rail() -> None:
    steps = [
        ("📄", "Upload", "Select report"),
        ("⌕", "Extract", "Read document text"),
        ("▦", "Parse", "Identify test results"),
        ("✦", "Explain", "Generate AI insights"),
    ]
    markup = '<div class="ml-progress-rail">'
    for icon, title, caption in steps:
        markup += (
            '<div class="ml-progress-step">'
            f'<div class="ml-progress-icon">{icon}</div>'
            f'<div><div class="ml-progress-title">{title}</div>'
            f'<div class="ml-progress-caption">{caption}</div></div></div>'
        )
    markup += '</div>'
    st.markdown(markup, unsafe_allow_html=True)


def render_results_table(results: list[dict[str, Any]], flags: dict[str, Any]) -> None:
    checked = flags.get("results", []) or []
    rows = []
    for index, item in enumerate(results):
        flag = checked[index] if index < len(checked) and isinstance(checked[index], dict) else {}
        status = str(flag.get("status", "unknown"))
        label, css_class = _status_pill(status)
        row_class = "row-low" if status in {"low", "high"} else ""
        name = html.escape(str(item.get("test_name", "Unknown test")))
        value = html.escape(_fmt_number(item.get("value")))
        unit = html.escape(str(item.get("unit", "—")))
        reference = html.escape(_format_reference(item))
        rows.append(
            f'<tr class="{row_class}"><td><strong>{name}</strong></td>'
            f'<td>{value}</td><td>{unit}</td><td>{reference}</td>'
            f'<td><span class="ml-status-pill {css_class}">{label}</span></td></tr>'
        )

    if not rows:
        rows_html = '<tr><td colspan="5">No numerical results are available.</td></tr>'
    else:
        rows_html = "".join(rows)

    st.markdown(
        """
        <div class="ml-table-wrap">
          <table class="ml-table">
            <thead><tr><th>Test name</th><th>Result</th><th>Unit</th><th>Reference range</th><th>Status</th></tr></thead>
            <tbody>
        """ + rows_html + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_explanation_panel(explanation: dict[str, Any]) -> None:
    ai_generated = explanation.get("ai_generated", True) is not False
    if ai_generated:
        panel_title = "AI Explanation"
        panel_subtitle = "Generated from the extracted report data"
    else:
        panel_title = "Reference Range Comparison"
        panel_subtitle = "Gemini is temporarily unavailable; showing report-based comparisons only"

    st.markdown(
        '<div class="ml-card-heading"><div class="ml-icon-tile">✦</div>'
        f'<div><div class="ml-panel-title">{panel_title}</div>'
        f'<div class="ml-panel-sub">{panel_subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )
    summary = explanation.get("summary") or "No summary is available."
    st.markdown(f'<div class="ml-ai-explanation">{html.escape(str(summary))}</div>', unsafe_allow_html=True)

    explanations = explanation.get("explanations", []) or []
    if explanations:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        for item in explanations:
            if not isinstance(item, dict):
                continue
            name = str(item.get("test_name") or "Test explanation")
            body = str(item.get("explanation") or "No explanation supplied.")
            with st.expander(name, expanded=False):
                st.write(body)
                limitations = item.get("limitations", []) or []
                if limitations:
                    st.caption("Limitations: " + "; ".join(str(x) for x in limitations))


def render_report_analysis(result: dict[str, Any], uploaded: Any) -> None:
    render_process_rail()
    st.markdown(
        '<div class="ml-analysis-heading"><div class="ml-success-mark">✓</div>'
        '<div><div class="ml-analysis-title">Report Analysis Complete</div>'
        '<div class="ml-panel-sub">Your medical report has been processed successfully.</div></div></div>',
        unsafe_allow_html=True,
    )

    results = result.get("results", []) or []
    flags = result.get("flags", {}) or {}
    counts = flags.get("flags", {}) or {}
    pages = (result.get("extracted", {}) or {}).get("pages_processed", 0)
    within = int(counts.get("within_range", 0) or 0)
    outside = int(counts.get("low", 0) or 0) + int(counts.get("high", 0) or 0)
    unknown = int(counts.get("unknown", 0) or 0)
    stats = [
        ("📄", pages, "Pages processed", ""),
        ("⚗️", len(results), "Test results found", ""),
        ("✓", within, "Within range", "good"),
        ("⚠", outside, "Outside range", "bad"),
    ]
    stat_html = '<div class="ml-stat-grid">'
    for icon, value, label, variant in stats:
        stat_html += (
            f'<div class="ml-stat {variant}"><div class="ml-stat-icon">{icon}</div>'
            f'<div><div class="ml-stat-value">{html.escape(str(value))}</div>'
            f'<div class="ml-stat-label">{label}</div></div></div>'
        )
    stat_html += '</div>'
    st.markdown(stat_html, unsafe_allow_html=True)

    if unknown:
        st.info(f"{unknown} result(s) need manual review because the supplied information was incomplete or could not be safely compared.")

    filename = result.get("filename") or getattr(uploaded, "name", "Medical report")
    size = result.get("file_size") or getattr(uploaded, "size", 0)
    download_bytes = uploaded.getvalue() if uploaded is not None else result.get("file_bytes")
    file_col, action_col = st.columns([4, 1.2])
    with file_col:
        st.markdown(
            '<div class="ml-filebar"><div style="display:flex;gap:12px;align-items:center">'
            '<div class="ml-icon-tile">📄</div><div>'
            f'<div class="ml-file-name">{html.escape(str(filename))}</div>'
            f'<div class="ml-file-meta">{int(size):,} bytes · Original uploaded document</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )
    with action_col:
        if download_bytes:
            st.download_button(
                "↓  Download report",
                data=download_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
                key="download_original_report",
            )

    left, right = st.columns([1.13, .97], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="ml-card-heading"><div class="ml-icon-tile">⚗</div>'
                '<div><div class="ml-panel-title">Test Results</div>'
                f'<div class="ml-panel-sub">{len(results)} numerical test(s) detected from the report</div></div></div>',
                unsafe_allow_html=True,
            )
            render_results_table(results, flags)
    with right:
        with st.container(border=True):
            render_explanation_panel(result.get("explanation", {}) or {})

    explanation = result.get("explanation", {}) or {}
    st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
    with st.container():
        summary = explanation.get("summary") or "No overall summary is available."
        st.markdown(
            '<div class="ml-summary-box"><div class="ml-card-heading" style="margin-bottom:7px">'
            '<div class="ml-icon-tile">📄</div><div class="ml-summary-title">Overall Summary</div></div>'
            f'<div class="ml-ai-explanation">{html.escape(str(summary))}</div></div>',
            unsafe_allow_html=True,
        )
    raw_text = result.get("raw_text", "")
    if raw_text:
        with st.expander("View raw extracted text"):
            st.code(raw_text, language="text")

    if st.button("↻  Analyze another report", key="analyze_another_report"):
        st.session_state.pop(ANALYSIS_STATE_KEY, None)
        st.session_state.pop(ANALYSIS_FILE_KEY, None)
        st.session_state["report_upload_mode"] = True
        st.rerun()


# --------------------------------------------------
# REPORT EXPLAINER PAGE
# --------------------------------------------------
def report_tab(language_code: str) -> None:
    render_topbar("Report Explainer", language_code)
    render_hero(
        "MediLens <span class='accent'>AI</span>",
        "Understand your medical report in plain language. Upload a report for an explanation or explore trusted health information with MediLens AI.",
        "hero-report.svg",
        "Report Explainer",
    )
    current_result = st.session_state.get(ANALYSIS_STATE_KEY)
    upload_mode = current_result is None or st.session_state.get("report_upload_mode", False)

    if upload_mode:
        with st.container(border=True):
            st.markdown(
                '<div class="ml-card-heading"><div class="ml-icon-tile">📄</div>'
                '<div><div class="ml-panel-title">Explain a medical report</div>'
                '<div class="ml-panel-sub">Upload a PDF laboratory report to extract test results and receive a patient-friendly explanation.</div></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="ml-upload-intro"><div class="ml-upload-symbol">⇧</div>'
                '<div><div class="ml-upload-title">Upload your medical report</div>'
                '<div class="ml-upload-caption">Drag a PDF into the drop area or choose a file from your computer.</div></div></div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Choose a medical report",
                type=["pdf"],
                key="report_upload",
                label_visibility="collapsed",
                help="The current extraction module accepts PDF files.",
            )
            st.markdown(
                '<div class="ml-format-row" style="justify-content:flex-start;margin-top:5px">'
                '<span class="ml-format">PDF ONLY</span>'
                '<span class="ml-format">DIGITAL OR SCANNED</span>'
                '<span class="ml-format">USE DE-IDENTIFIED DATA</span></div>',
                unsafe_allow_html=True,
            )
            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                current_hash = hashlib.sha256(file_bytes).hexdigest()
                st.caption(f"Selected file: {uploaded.name} · {len(file_bytes):,} bytes")
                if st.session_state.get(ANALYSIS_FILE_KEY) != current_hash:
                    st.session_state.pop(ANALYSIS_STATE_KEY, None)

                analyze = st.button(
                    "✦  Analyze report",
                    type="primary",
                    use_container_width=True,
                    key="analyze_report",
                )
                if analyze:
                    st.session_state[ANALYSIS_FILE_KEY] = current_hash
                    st.session_state.pop("report_upload_mode", None)
                    try:
                        # Existing module imports and processing calls are unchanged.
                        extract_report = load_function("report_extractor", "extract_report")
                        parse_report = load_function("report_parser", "parse_report")
                        check_abnormalities = load_function("abnormality_checker", "check_abnormalities")
                        explain_report = load_function("llm_service", "explain_report")

                        with st.spinner("Extracting your report and preparing the explanation..."):
                            extracted = extract_report(file_bytes, uploaded.name)
                            if is_module_error(extracted):
                                st.session_state[ANALYSIS_STATE_KEY] = {
                                    "status": "error",
                                    "message": "We couldn't read this PDF. Please check the file and try again.",
                                }
                            else:
                                extracted_text = extracted.get("text", "")
                                if not isinstance(extracted_text, str) or not extracted_text.strip():
                                    st.session_state[ANALYSIS_STATE_KEY] = {
                                        "status": "error",
                                        "message": "No readable text was found in this PDF.",
                                    }
                                else:
                                    parsed = parse_report(extracted_text)
                                    valid_results = get_valid_results(parsed)

                                    if not valid_results:
                                        st.session_state[ANALYSIS_STATE_KEY] = {
                                            "status": "error",
                                            "message": "No numerical laboratory results were identified. Please upload a supported laboratory report.",
                                        }
                                    else:
                                        clean_report = dict(parsed)
                                        clean_report["results"] = valid_results
                                        flags = check_abnormalities(clean_report)

                                        if is_module_error(flags):
                                            st.session_state[ANALYSIS_STATE_KEY] = {
                                                "status": "error",
                                                "message": "The report could not be prepared for explanation. Please verify the uploaded document.",
                                            }
                                        else:
                                            language = {
                                                "en": "English",
                                                "hi": "Hindi",
                                                "kn": "Kannada",
                                            }.get(language_code, "English")
                                            explanation = explain_report(clean_report, flags, language)

                                            if not isinstance(explanation, dict) or explanation.get("status") != "success":
                                                details = explanation.get("warnings", []) if isinstance(explanation, dict) else []
                                                print("MediLens explanation issue:", details, flush=True)
                                                st.session_state[ANALYSIS_STATE_KEY] = {
                                                    "status": "error",
                                                    "message": "Gemini couldn't complete the explanation just now. Please try again shortly.",
                                                }
                                            else:
                                                # Store fields already returned by the existing modules for presentation.
                                                st.session_state[ANALYSIS_STATE_KEY] = {
                                                    "status": "success",
                                                    "explanation": explanation,
                                                    "extracted": extracted,
                                                    "results": valid_results,
                                                    "flags": flags,
                                                    "raw_text": extracted_text,
                                                    "filename": uploaded.name,
                                                    "file_size": len(file_bytes),
                                                    "file_bytes": file_bytes,
                                                }
                                                st.session_state[ANALYSIS_FILE_KEY] = current_hash

                    except (ImportError, AttributeError) as exc:
                        print("Report module integration error:", type(exc).__name__, str(exc), flush=True)
                        st.session_state[ANALYSIS_STATE_KEY] = {
                            "status": "error",
                            "message": "A report-processing component is not available. Check the project modules.",
                        }
                    except Exception as exc:
                        print("MediLens report workflow error:", type(exc).__name__, str(exc), flush=True)
                        st.session_state[ANALYSIS_STATE_KEY] = {
                            "status": "error",
                            "message": "The report workflow couldn't finish. Please try again.",
                        }
                    st.rerun()

            st.markdown(
                '<div class="ml-feature-strip">'
                '<div class="ml-feature"><div class="ml-feature-icon">📄</div><div><div class="ml-feature-title">PDF extraction</div><div class="ml-feature-copy">Digital and scanned reports</div></div></div>'
                '<div class="ml-feature"><div class="ml-feature-icon">⚡</div><div><div class="ml-feature-title">AI explanation</div><div class="ml-feature-copy">Plain-language insights</div></div></div>'
                '<div class="ml-feature"><div class="ml-feature-icon">🌐</div><div><div class="ml-feature-title">Multiple languages</div><div class="ml-feature-copy">Choose your preferred language</div></div></div>'
                '<div class="ml-feature"><div class="ml-feature-icon">🛡️</div><div><div class="ml-feature-title">Privacy focused</div><div class="ml-feature-copy">Use sample or de-identified data</div></div></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile">✦</div><div><div class="ml-panel-title">How it works</div><div class="ml-panel-sub">Four clear stages from document to explanation</div></div></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ml-process">'
                '<div class="ml-process-step"><div class="ml-step-no">1</div><div><div class="ml-step-title">Upload report</div><div class="ml-step-copy">Select a PDF report</div></div></div>'
                '<div class="ml-process-step"><div class="ml-step-no">2</div><div><div class="ml-step-title">Extract text</div><div class="ml-step-copy">Read document content</div></div></div>'
                '<div class="ml-process-step"><div class="ml-step-no">3</div><div><div class="ml-step-title">Parse results</div><div class="ml-step-copy">Identify tests and ranges</div></div></div>'
                '<div class="ml-process-step"><div class="ml-step-no">4</div><div><div class="ml-step-title">AI explanation</div><div class="ml-step-copy">Understand the results</div></div></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        if current_result and current_result.get("status") == "error":
            st.warning(current_result.get("message", "The report could not be processed."))

    elif current_result and current_result.get("status") == "success":
        uploaded = st.session_state.get("report_upload")
        render_report_analysis(current_result, uploaded)
    elif current_result:
        st.warning(current_result.get("message", "The report could not be processed."))
        if st.button("Try another report", key="retry_report"):
            st.session_state.pop(ANALYSIS_STATE_KEY, None)
            st.session_state.pop(ANALYSIS_FILE_KEY, None)
            st.rerun()


# --------------------------------------------------
# MEDICAL Q&A PAGE
# --------------------------------------------------
QUICK_QUESTIONS = [
    "What is a normal glucose level?",
    "Explain high cholesterol",
    "What causes low Vitamin D?",
]


def _set_suggested_question(question: str) -> None:
    st.session_state["medical_question"] = question


def render_source_cards(sources: list[dict[str, Any]]) -> None:
    """Render only source metadata actually returned by the RAG module."""
    if not sources:
        st.markdown(
            '<div class="ml-source-empty"><div class="ml-source-empty-icon">⌕</div>'
            '<div class="ml-source-empty-title">Your references will appear here</div>'
            '<div class="ml-source-empty-copy">Ask a question to see the documents and page details returned by the medical knowledge base.</div></div>',
            unsafe_allow_html=True,
        )
        return

    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        title = source.get("title") or source.get("source") or source.get("source_id") or f"Source {index}"
        topic = source.get("topic") or "Retrieved medical reference"
        page = source.get("page")
        page_text = f"Page {html.escape(str(page))}" if page is not None else "Page not supplied"
        excerpt = source.get("excerpt") or "No preview text was returned by the source module."
        url = source.get("url")
        link_html = (
            f'<a class="ml-source-open" href="{html.escape(str(url), quote=True)}" target="_blank" rel="noopener noreferrer" aria-label="Open source">↗</a>'
            if url else '<span class="ml-source-open ml-source-open-muted" title="No direct URL supplied">↗</span>'
        )
        st.markdown(
            '<div class="ml-source-card">'
            '<div class="ml-source-icon">▤</div>'
            '<div class="ml-source-content">'
            '<div class="ml-source-topline">'
            f'<div class="ml-source-title">{html.escape(str(title))}</div>'
            f'<div class="ml-source-page">{page_text}</div>'
            '</div>'
            f'<div class="ml-source-meta">{html.escape(str(topic))}</div>'
            f'<div class="ml-source-snippet">{html.escape(str(excerpt))}</div>'
            '<div class="ml-source-bottom"><span class="ml-source-relevance">Retrieved reference</span>'
            f'{link_html}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )


def chatbot_tab(language_code: str) -> None:
    render_topbar("Medical Q&A", language_code)
    render_hero(
        "Medical Information <span class='accent'>Chatbot</span>",
        "Ask evidence-based medical questions powered by trusted health references.",
        "hero-chat.svg",
        "Medical Q&A",
    )

    left, right = st.columns([1.55, 1.0], gap="medium")
    chat_result = st.session_state.get(CHAT_STATE_KEY)

    # Main conversation surface: quick prompts, user question, AI response and composer.
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="ml-qna-quick-heading"><span class="ml-quick-icon">✣</span>'
                '<span>Quick Questions</span></div>',
                unsafe_allow_html=True,
            )
            prompt_columns = st.columns(3, gap="small")
            for index, question in enumerate(QUICK_QUESTIONS):
                with prompt_columns[index]:
                    if st.button(
                        question,
                        key=f"quick_question_{index}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["medical_question"] = question
                        st.rerun()

            st.markdown('<div class="ml-chat-divider"></div>', unsafe_allow_html=True)
            if chat_result:
                if chat_result.get("status") == "error":
                    st.warning(chat_result.get("message", "The answer is unavailable."))
                else:
                    question = chat_result.get("question", "")
                    response_time = chat_result.get("timestamp", "")
                    if question:
                        st.markdown(
                            '<div class="ml-user-row"><div class="ml-user-question">'
                            + html.escape(str(question))
                            + '</div><div class="ml-user-avatar">●</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="ml-question-time">{html.escape(str(response_time))}</div>',
                            unsafe_allow_html=True,
                        )

                    count = len(chat_result.get("sources", []) or [])
                    st.markdown(
                        '<div class="ml-answer-card">'
                        '<div class="ml-ai-head"><div class="ml-ai-avatar">✦</div>'
                        '<div class="ml-ai-identity"><div class="ml-ai-name">MediLens AI'
                        f'<span class="ml-source-badge">✦ {count} source(s)</span></div>'
                        '<div class="ml-ai-caption">Evidence-aware health information</div></div>'
                        f'<div class="ml-ai-time">{html.escape(str(response_time))}</div></div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    if chat_result.get("insufficient_evidence"):
                        st.warning("The available reference material may not contain enough evidence for a reliable answer.")
                    answer = chat_result.get("answer", "No answer was returned.")
                    st.markdown('<div class="ml-answer">', unsafe_allow_html=True)
                    st.markdown(answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="ml-answer-footnote">This information is based on the references returned by the current medical knowledge base. It is not a diagnosis.</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="ml-answer-actions"><span>◫&nbsp; Copy-ready answer</span>'
                        '<span>♡&nbsp; Health awareness</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="ml-qna-welcome">'
                    '<div class="ml-welcome-orb">✦</div>'
                    '<div class="ml-welcome-title">Ask. Understand. Take better care.</div>'
                    '<div class="ml-welcome-copy">Start with a suggested question or type your own. MediLens AI will look for relevant material in the connected medical knowledge base.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            with st.form("medilens_chat_form", clear_on_submit=False):
                composer_col, send_col = st.columns([5.1, 1.0], gap="small")
                with composer_col:
                    question = st.text_area(
                        "Ask a medical information question",
                        placeholder="Ask a medical question...",
                        height=62,
                        key="medical_question",
                        label_visibility="collapsed",
                    )
                with send_col:
                    submitted = st.form_submit_button("➤  Send", use_container_width=True)

            if submitted:
                if not question.strip():
                    st.warning("Enter a question before continuing.")
                else:
                    try:
                        # Existing RAG call and language argument are unchanged.
                        answer_question = load_function("medical_rag", "answer_question")
                        with st.spinner("Searching trusted medical references..."):
                            result = answer_question(question.strip(), language_code)

                        if is_module_error(result):
                            st.session_state[CHAT_STATE_KEY] = {
                                "status": "error",
                                "message": "The Medical Q&A service is temporarily unavailable.",
                            }
                        else:
                            st.session_state[CHAT_STATE_KEY] = {
                                "status": "success",
                                "question": question.strip(),
                                "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p"),
                                "answer": result.get("answer", ""),
                                "sources": result.get("sources", []) or [],
                                "insufficient_evidence": result.get("insufficient_evidence", False),
                            }
                    except Exception as exc:
                        print("Medical Q&A error:", type(exc).__name__, str(exc), flush=True)
                        st.session_state[CHAT_STATE_KEY] = {
                            "status": "error",
                            "message": "The Medical Q&A service couldn't complete the request.",
                        }
                    st.rerun()


    # Source references occupy their own panel, as in the approved visualization.
    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="ml-source-panel-heading"><div class="ml-source-panel-icon">▤</div>'
                '<div><div class="ml-panel-title">Source References</div>'
                '<div class="ml-panel-sub">Information from trusted medical sources</div></div></div>',
                unsafe_allow_html=True,
            )
            if chat_result and chat_result.get("status") == "success":
                render_source_cards(chat_result.get("sources", []) or [])
            else:
                render_source_cards([])



# --------------------------------------------------
# ABOUT PAGE — mission / process / stack / privacy / roadmap
# --------------------------------------------------
def about_tab(language_code: str) -> None:
    render_topbar("About", language_code)
    render_hero(
        "About MediLens <span class='accent'>AI</span>",
        "A student-built health-awareness prototype to make medical reports easier to understand using document processing and generative AI.",
        "hero-about.svg",
        "About MediLens AI",
    )

    top_left, top_right = st.columns([1.55, .92], gap="medium")
    with top_left:
        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile" style="color:#fcd34d;background:rgba(245,158,11,.13)">💡</div><div><div class="ml-panel-title">Our Goal</div><div class="ml-panel-sub">Make medical information easier to understand</div></div></div>', unsafe_allow_html=True)
            st.markdown('<div class="ml-ai-explanation">To help people understand their medical reports in simple language and provide health information based on the reference material available to the project.</div>', unsafe_allow_html=True)
    with top_right:
        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile">⭐</div><div><div class="ml-panel-title">Key Features</div></div></div>', unsafe_allow_html=True)
            feature_rows = [
                ("📋", "Medical report explanation", "Extract and explain supported laboratory results."),
                ("🧠", "AI-powered insights", "Patient-friendly explanations from Gemini."),
                ("📘", "Medical Q&A chatbot", "Answers grounded in available references."),
                ("🌐", "Multiple languages", "English, Hindi and Kannada workflows."),
                ("🛡️", "Privacy focused", "Use de-identified sample reports for testing."),
            ]
            markup = ""
            for icon, title, copy in feature_rows:
                markup += f'<div class="ml-feature-list-item"><div class="ml-feature-list-icon">{icon}</div><div><div class="ml-feature-list-title">{title}</div><div class="ml-feature-list-copy">{copy}</div></div></div>'
            st.markdown(markup, unsafe_allow_html=True)

    how_col, privacy_col = st.columns([1.55, .92], gap="medium")
    with how_col:
        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile">⚙</div><div><div class="ml-panel-title">How It Works</div><div class="ml-panel-sub">From uploaded document to a plain-language explanation</div></div></div>', unsafe_allow_html=True)
            flow = [
                ("1", "Upload report", "Select a PDF laboratory report."),
                ("2", "Extract text", "PyMuPDF reads digital PDF text; Tesseract OCR is used when configured for scanned pages."),
                ("3", "Parse results", "Identify test names, values, units and supplied reference ranges."),
                ("4", "AI explanation", "Gemini generates patient-friendly explanations from structured report information."),
            ]
            markup = '<div class="ml-flow-grid">'
            for no, title, copy in flow:
                markup += f'<div class="ml-flow-card"><div class="ml-flow-number"><div class="ml-step-no">{no}</div><div class="ml-flow-title">{title}</div></div><div class="ml-flow-copy">{copy}</div></div>'
            markup += '</div>'
            st.markdown(markup, unsafe_allow_html=True)
    with privacy_col:
        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile">🔒</div><div><div class="ml-panel-title">Data and Privacy</div></div></div>', unsafe_allow_html=True)
            privacy = [
                "Use only synthetic or de-identified sample reports for testing.",
                "Do not upload real patient information to a public demo.",
                "Verify extracted values against the original report.",
                "The prototype is for education and health awareness only.",
            ]
            markup = ""
            for item in privacy:
                markup += f'<div class="ml-privacy-row"><span style="color:#34d399">✓</span><span>{html.escape(item)}</span></div>'
            st.markdown(markup, unsafe_allow_html=True)

    tech_col, future_col = st.columns([1.55, .92], gap="medium")
    with tech_col:
        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile">&lt;/&gt;</div><div><div class="ml-panel-title">Technology Stack</div><div class="ml-panel-sub">Document processing, data extraction and generative AI</div></div></div>', unsafe_allow_html=True)
            technologies = [
                ("🐍", "Python", "Backend & processing"),
                ("📄", "PyMuPDF", "PDF text extraction"),
                ("🔎", "Tesseract OCR", "Scanned-page text"),
                ("✦", "Gemini API", "AI explanations"),
                ("⚡", "Streamlit", "Web interface"),
            ]
            markup = '<div class="ml-tech-grid">'
            for icon, name, copy in technologies:
                markup += f'<div class="ml-tech"><div class="ml-tech-icon">{icon}</div><div class="ml-tech-name">{name}</div><div class="ml-tech-copy">{copy}</div></div>'
            markup += '</div>'
            st.markdown(markup, unsafe_allow_html=True)
            st.caption("ChromaDB is used by the Medical Q&A module for persistent vector retrieval.")
    with future_col:
        with st.container(border=True):
            st.markdown('<div class="ml-card-heading"><div class="ml-icon-tile">↗</div><div><div class="ml-panel-title">Future Improvements</div></div></div>', unsafe_allow_html=True)
            future_items = [
                "Support more report layouts and document types.",
                "Expand the reviewed medical reference collection.",
                "Improve multilingual coverage and accessibility.",
                "Add clearer provenance for extracted measurements.",
            ]
            for item in future_items:
                st.markdown(f'<div class="ml-privacy-row"><span style="color:#8b9dff">•</span><span>{html.escape(item)}</span></div>', unsafe_allow_html=True)


# --------------------------------------------------
# APP ENTRY POINT
# --------------------------------------------------
def main() -> None:
    inject_styles()
    active_page, language_code = render_sidebar()

    if active_page == "report":
        report_tab(language_code)
    elif active_page == "qna":
        chatbot_tab(language_code)
    else:
        about_tab(language_code)

    render_footer()


if __name__ == "__main__":
    main()