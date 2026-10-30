"""MediLens AI - Member 1's Streamlit application and integration layer."""
from __future__ import annotations

import importlib
from typing import Any

import streamlit as st

st.set_page_config(page_title="MediLens AI", page_icon="🩺", layout="wide")

DISCLAIMER = (
    "MediLens AI is a student health-awareness prototype, not a diagnosis or a "
    "replacement for a qualified healthcare professional. Verify extracted values "
    "against the original report."
)
LANGUAGES = {"English": "en", "Hindi": "hi", "Kannada": "kn"}


def load_function(module_name: str, function_name: str):
    """Import a teammate's function only when a user tries that workflow."""
    module = importlib.import_module(f"src.{module_name}")
    return getattr(module, function_name)


def show_warnings(result: dict[str, Any], section: str) -> None:
    for warning in result.get("warnings", []) or []:
        st.warning(f"{section}: {warning}")


def show_status_error(result: dict[str, Any], section: str) -> bool:
    status = result.get("status", "error")
    if status == "error":
        messages = result.get("warnings", []) or [
            "The module returned an error without details."
        ]
        st.error(
            f"{section} failed: {'; '.join(str(message) for message in messages)}"
        )
        return True
    if status == "partial":
        st.warning(
            f"{section} returned partial results. Review all warnings and values."
        )
    return False


def render_report_results(
    parsed: dict[str, Any],
    flags: dict[str, Any],
    explanation: dict[str, Any],
) -> None:
    st.subheader("Plain-language summary")
    st.write(
        explanation.get("summary")
        or "No summary was returned by the explanation module."
    )

    st.subheader("Reference-range summary")
    counts = flags.get("flags", {}) or {}
    st.write(
        {
            "Low": counts.get("low", 0),
            "Within range": counts.get("within_range", 0),
            "High": counts.get("high", 0),
            "Unknown": counts.get("unknown", 0),
        }
    )

    st.subheader("Reference-range details")
    results = flags.get("results", []) or []
    if results:
        st.dataframe(results, width="stretch", hide_index=True)
    else:
        st.info(
            "No reference-range details were returned. "
            "This does not prove that all results are normal."
        )

    explanations = explanation.get("explanations", []) or []
    if explanations:
        st.subheader("Test explanations")
        for item in explanations:
            name = item.get("test_name", "Test")
            with st.expander(str(name), expanded=True):
                st.write(item.get("explanation", "No explanation supplied."))
                limitations = item.get("limitations", []) or []
                if limitations:
                    st.caption(
                        "Limitations: " + "; ".join(str(x) for x in limitations)
                    )

    safety_message = explanation.get("safety_message")
    if safety_message:
        st.info(safety_message)

    for title, result in (
        ("Parser", parsed),
        ("Range checker", flags),
        ("Explanation", explanation),
    ):
        show_warnings(result, title)


def report_tab(language_code: str) -> None:
    st.subheader("Explain a medical report")
    st.write(
        "Upload a PDF report. Use only synthetic or de-identified sample reports "
        "for project testing."
    )
    uploaded = st.file_uploader(
        "Choose a medical report", type=["pdf"], key="report_upload"
    )

    if uploaded is None:
        st.info(
            "Your report will be processed only after you select a file "
            "and press Analyze report."
        )
        return

    st.caption(f"Selected file: {uploaded.name} • {uploaded.size:,} bytes")
    if not st.button("Analyze report", type="primary", key="analyze_report"):
        return

    try:
        extract_report = load_function("report_extractor", "extract_report")
        parse_report = load_function("report_parser", "parse_report")
        check_abnormalities = load_function(
            "abnormality_checker", "check_abnormalities"
        )
        explain_report = load_function("llm_service", "explain_report")
    except (ImportError, AttributeError) as exc:
        st.warning(
            "The report workflow is not fully connected yet. Member 2 must add "
            "report_extractor.py, report_parser.py and llm_service.py, and Member 4 "
            "must add abnormality_checker.py under src/. Then try again."
        )
        st.caption(f"Integration detail: {type(exc).__name__}")
        return

    try:
        with st.spinner("Processing the report..."):
            extracted = extract_report(uploaded.getvalue(), uploaded.name)
            show_warnings(extracted, "Extraction")
            if show_status_error(extracted, "Extraction"):
                return

            parsed = parse_report(extracted.get("text", ""))
            if show_status_error(parsed, "Parsing"):
                show_warnings(parsed, "Parsing")
                return

            flags = check_abnormalities(parsed)
            if show_status_error(flags, "Reference-range checking"):
                show_warnings(flags, "Reference-range checking")
                return

            explanation_language = {
                "en": "English",
                "hi": "Hindi",
                "kn": "Kannada",
            }.get(language_code, "English")

            # A temporary Gemini/API failure should not hide successful extraction,
            # parsing, or reference-range results.
            try:
                explanation = explain_report(
                    parsed,
                    flags,
                    explanation_language,
                )
            except Exception as exc:
                explanation = {
                    "status": "error",
                    "summary": "The AI explanation is temporarily unavailable.",
                    "explanations": [],
                    "safety_message": (
                        "This prototype does not provide a diagnosis. Compare the "
                        "extracted values with the original report and consult a "
                        "qualified healthcare professional."
                    ),
                    "warnings": [f"Explanation service error: {type(exc).__name__}"],
                }

        if show_status_error(explanation, "Explanation"):
            show_warnings(explanation, "Explanation")
            explanation = {
                "status": "partial",
                "summary": (
                    "The report was extracted and checked successfully, but the AI "
                    "explanation is temporarily unavailable. Review the test results "
                    "and reference-range details below."
                ),
                "explanations": [],
                "safety_message": (
                    "This prototype does not provide a diagnosis. Compare all values "
                    "with the original report and consult a qualified healthcare "
                    "professional."
                ),
                "warnings": [],
            }

        render_report_results(parsed, flags, explanation)
        st.caption(
            "Please compare extracted values, units and reference ranges "
            "with the original report."
        )
    except Exception as exc:  # Do not expose API keys or raw tracebacks.
        st.error(
            "The report workflow could not complete. Check the module logs "
            "and function contracts."
        )
        st.caption(f"Error type: {type(exc).__name__}")


def chatbot_tab(language_code: str) -> None:
    st.subheader("Medical information chatbot")
    st.write(
        "Ask a general health-information question. Answers should be grounded "
        "in the team's approved reference documents."
    )
    question = st.text_area(
        "Your question",
        placeholder="Example: What does a reference range mean?",
        key="medical_question",
    )

    if st.button("Ask question", type="primary", key="ask_question"):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        try:
            answer_question = load_function("medical_rag", "answer_question")
            with st.spinner("Searching trusted references..."):
                result = answer_question(question.strip(), language_code)
            if show_status_error(result, "Medical Q&A"):
                show_warnings(result, "Medical Q&A")
                return

            answer = result.get("answer")
            if result.get("insufficient_evidence"):
                st.warning(
                    "The knowledge base did not provide enough evidence "
                    "for a reliable answer."
                )
            if answer:
                st.markdown("#### Answer")
                st.write(answer)

            sources = result.get("sources", []) or []
            if sources:
                st.markdown("#### Sources")
                for index, source in enumerate(sources, start=1):
                    title = (
                        source.get("title")
                        or source.get("source_id")
                        or f"Source {index}"
                    )
                    url = source.get("url")
                    page = source.get("page")
                    suffix = f" (page {page})" if page is not None else ""
                    if url:
                        st.markdown(f"{index}. [{title}]({url}){suffix}")
                    else:
                        st.write(f"{index}. {title}{suffix}")
                    if source.get("excerpt"):
                        st.caption(source["excerpt"])
            show_warnings(result, "Medical Q&A")
        except (ImportError, AttributeError):
            st.warning(
                "The chatbot is not connected yet. Member 3 must add "
                "src/medical_rag.py with answer_question(question, language)."
            )
        except Exception as exc:
            st.error(
                "The chatbot could not complete the request. Check the RAG "
                "module and its configuration."
            )
            st.caption(f"Error type: {type(exc).__name__}")


def main() -> None:
    st.title("🩺 MediLens AI")
    st.markdown("### Understand your medical report in plain language")
    st.write(
        "Upload a report for an explanation or ask a question using the "
        "medical knowledge base."
    )
    st.warning(DISCLAIMER)

    with st.sidebar:
        st.header("Settings")
        selected_language = st.selectbox(
            "Explanation language", list(LANGUAGES.keys()), index=0
        )
        st.caption("The selected language is passed to the LLM and RAG modules.")
        st.divider()
        st.markdown("**Project status**")
        st.write("Application UI: available")
        st.write("Processing modules: connect as teammates finish them")

    report_page, chatbot_page, about_page = st.tabs(
        ["📄 Report explainer", "💬 Medical Q&A", "ℹ️ About"]
    )
    with report_page:
        report_tab(LANGUAGES[selected_language])
    with chatbot_page:
        chatbot_tab(LANGUAGES[selected_language])
    with about_page:
        st.markdown("#### What the prototype does")
        st.write(
            "1. Extracts text from a PDF report.\n"
            "2. Parses test names, values, units and supplied reference ranges.\n"
            "3. Flags values using only the report's supplied ranges.\n"
            "4. Explains results and answers questions using retrieved medical references."
        )
        st.markdown("#### Important limitations")
        st.write(
            "A reference-range flag is not a diagnosis. OCR can make mistakes. "
            "The chatbot must say when retrieved evidence is insufficient. "
            "Do not use this student prototype to make treatment decisions."
        )


if __name__ == "__main__":
    main()
