"""Test MediLens PDF extraction and parsing without calling Gemini or using API quota."""
from pathlib import Path
import sys

from src import report_extractor, report_parser

PDF_PATH = Path("MediLens_Synthetic_Medical_Lab_Report.pdf")


def main() -> int:
    print("Parser module loaded from:", report_parser.__file__)
    print("Extractor module loaded from:", report_extractor.__file__)

    if not PDF_PATH.exists():
        print(f"ERROR: Could not find {PDF_PATH.resolve()}")
        print("Copy the synthetic PDF into the same folder as app.py and this script.")
        return 1

    try:
        extracted = report_extractor.extract_report(PDF_PATH.read_bytes(), PDF_PATH.name)
    except Exception as exc:
        print("EXTRACTION FAILED:", type(exc).__name__, str(exc))
        return 1

    print("\n=== EXTRACTION ===")
    print("Status:", extracted.get("status"))
    print("Source type:", extracted.get("source_type"))
    print("Pages processed:", extracted.get("pages_processed"))
    for warning in extracted.get("warnings", []) or []:
        print("Warning:", warning)

    extracted_text = extracted.get("text", "")
    print("\n=== EXTRACTED TEXT PREVIEW ===")
    print(extracted_text[:3000] if extracted_text else "[No text extracted]")

    if extracted.get("status") == "error" or not extracted_text.strip():
        print("\nSTOP: extraction did not return usable text. No Gemini request was made.")
        return 1

    try:
        parsed = report_parser.parse_report(extracted_text)
    except Exception as exc:
        print("PARSING FAILED:", type(exc).__name__, str(exc))
        return 1

    results = parsed.get("results", []) or []
    print("\n=== PARSED TEST RESULTS ===")
    print("Parse status:", parsed.get("status"))
    print("Number of tests found:", len(results))
    for item in results:
        print(
            f"- {item.get('test_name')}: value={item.get('value')!r}, "
            f"unit={item.get('unit')!r}, "
            f"reference_low={item.get('reference_low')!r}, "
            f"reference_high={item.get('reference_high')!r}"
        )
    for warning in parsed.get("warnings", []) or []:
        print("Parser warning:", warning)

    if len(results) == 5:
        print("\nPASS: all five sample tests were parsed. No Gemini request was made.")
        return 0

    print("\nCHECK NEEDED: expected five sample tests but found", len(results))
    print("Use the extracted-text preview above to diagnose the extractor/parser format mismatch.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
