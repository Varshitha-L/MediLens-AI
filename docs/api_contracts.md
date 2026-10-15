# MediLens AI shared API contracts

All modules return a dictionary with `status` (`success`, `partial`, or `error`) and `warnings` (a list). Missing or uncertain values remain `None`; never fabricate test values, units, reference ranges, model metrics or citations.

| Module | Function signature |
|---|---|
| `src/report_extractor.py` | `extract_report(file_bytes, filename)` |
| `src/report_parser.py` | `parse_report(extracted_text)` |
| `src/abnormality_checker.py` | `check_abnormalities(parsed_report)` |
| `src/llm_service.py` | `explain_report(parsed_report, flags, language)` |
| `src/medical_rag.py` | `answer_question(question, language)` |
| `src/risk_predictor.py` | `predict_risk(features)` (optional) |
| `src/translation.py` | `translate_text(text, language_code)` |
| `src/safety.py` | `validate_report(parsed_report)` |

The report extractor accepts file bytes and a filename, not a Streamlit upload object. Abnormality flags are `high`, `low`, `within_range`, or `unknown`; a flag is not a diagnosis. RAG citations must use real source metadata, and insufficient evidence must be stated clearly.
