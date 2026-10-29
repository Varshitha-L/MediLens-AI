
import time
from pathlib import Path

from report_extractor import extract_report
from report_parser import parse_report
from llm_service import explain_report


def process_medical_report(pdf_path, language="English"):
    total_start = time.perf_counter()
    pdf_path = Path(pdf_path)

    # Stage 1: Extract PDF text
    print("1. Starting PDF extraction...", flush=True)
    start = time.perf_counter()

    try:
        pdf_result = extract_report(
            pdf_path.read_bytes(),
            pdf_path.name,
        )
    except OSError as error:
        return {
            "status": "error",
            "message": "Unable to read the PDF file.",
            "details": [str(error)],
        }

    print(
        f"   Extraction finished in "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    if pdf_result["status"] == "error":
        return {
            "status": "error",
            "message": "PDF extraction failed.",
            "details": pdf_result["warnings"],
        }

    if not pdf_result["text"].strip():
        return {
            "status": "error",
            "message": "No extractable text was found in the PDF.",
            "details": pdf_result["warnings"],
        }

    # Stage 2: Parse extracted text
    print("2. Parsing report...", flush=True)
    start = time.perf_counter()

    parsed_report = parse_report(pdf_result["text"])

    print(
        f"   Parsing finished in {time.perf_counter() - start:.2f}s",
        flush=True,
    )

    if parsed_report["status"] == "error":
        return {
            "status": "error",
            "message": "Report parsing failed.",
            "details": parsed_report["warnings"],
        }

    results = parsed_report["results"]

    print(f"   Found {len(results)} test results.", flush=True)
    print("\nParsed report:", flush=True)
    print(parsed_report, flush=True)

    if not results:
        return {
            "status": "error",
            "message": "No test results could be parsed.",
            "details": (
                pdf_result["warnings"] + parsed_report["warnings"]
            ),
        }

    # Stage 3: Gemini explanation
    # The abnormality checker is a separate module in the team
    # architecture. Until it is integrated, pass an empty flags list.
    print("\n3. Sending results to Gemini...", flush=True)
    print("   Waiting for API response...", flush=True)

    llm_start = time.perf_counter()

    try:
        flags = {
            "status": "success",
            "flags": [],
            "warnings": [
                "Abnormality checker is not integrated yet."
            ],
        }

        llm_result = explain_report(
            parsed_report,
            flags=flags,
            language=language,
        )


    except Exception as error:
        return {
            "status": "error",
            "message": f"Explanation call failed: {type(error).__name__}",
            "details": [str(error)],
        }

    print(
        f"   Gemini stage finished in "
        f"{time.perf_counter() - llm_start:.2f}s",
        flush=True,
    )

    if llm_result["status"] != "success":
        return {
            "status": "error",
            "message": "Explanation generation failed.",
            "details": llm_result.get("warnings", []),
        }

    # Preserve warnings from all stages.
    warnings = (
        pdf_result["warnings"]
        + parsed_report["warnings"]
        + llm_result["warnings"]
    )

    return {
        "status": "success",
        "pages_processed": pdf_result["pages_processed"],
        "parsed_report": parsed_report,
        "explanation": {
            "summary": llm_result["summary"],
            "explanations": llm_result["explanations"],
            "safety_message": llm_result["safety_message"],
            "warnings": llm_result["warnings"],
        },
        "warnings": warnings,
        "total_seconds": round(
            time.perf_counter() - total_start, 2
        ),
    }


if __name__ == "__main__":
    result = process_medical_report("sample_report.pdf")

    print("\n========== PIPELINE RESULT ==========", flush=True)
    print("Status:", result["status"], flush=True)

    if result["status"] == "success":
        print("Pages processed:", result["pages_processed"])
        print("\nParsed report:")
        print(result["parsed_report"])

        print("\nPatient-friendly explanation:")
        print(result["explanation"]["summary"])

        for item in result["explanation"]["explanations"]:
            print(f"\n{item['test_name']}: {item['explanation']}")
            print("Limitations:", item["limitations"])

        print("\nSafety message:")
        print(result["explanation"]["safety_message"])

        print("\nWarnings:", result["warnings"])
        print("Total time:", result["total_seconds"], "seconds")

    else:
        print("Message:", result["message"])
        print("Details:", result.get("details", []))
