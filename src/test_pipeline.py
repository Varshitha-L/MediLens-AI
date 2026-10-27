
import time

from report_extractor import extract_text
from report_parser import parse_report
from llm_service import explain_results


def process_medical_report(pdf_path):
    total_start = time.perf_counter()

    # Stage 1: Extract PDF text
    print("1. Starting PDF extraction...", flush=True)
    start = time.perf_counter()

    pdf_result = extract_text(pdf_path)

    print(
        f"   Extraction finished in "
        f"{time.perf_counter() - start:.2f}s",
        flush=True
    )

    if pdf_result["status"] == "error":
        return {
            "status": "error",
            "message": "PDF extraction failed.",
            "details": pdf_result["warnings"]
        }

    if not pdf_result["text"].strip():
        return {
            "status": "error",
            "message": "No selectable text found in the PDF.",
            "details": pdf_result["warnings"]
        }

    # Stage 2: Parse results
    print("2. Parsing report...", flush=True)
    parsed_results = parse_report(pdf_result["text"])

    print(f"   Found {len(parsed_results)} test results.", flush=True)

    if not parsed_results:
        return {
            "status": "error",
            "message": "No test results could be parsed.",
            "details": pdf_result["warnings"]
        }

    # Show parsed results before calling Gemini
    print("\nParsed results:", flush=True)
    for result in parsed_results:
        print(result, flush=True)

    # Stage 3: Gemini explanation
    print("\n3. Sending results to Gemini...", flush=True)
    print("   Waiting for API response...", flush=True)

    llm_start = time.perf_counter()

    try:
        llm_result = explain_results(parsed_results)

    except Exception as error:
        return {
            "status": "error",
            "message": f"Gemini call failed: {type(error).__name__}",
            "details": [str(error)]
        }

    print(
        f"   Gemini stage finished in "
        f"{time.perf_counter() - llm_start:.2f}s",
        flush=True
    )

    if llm_result["status"] != "success":
        return {
            "status": "error",
            "message": "Explanation generation failed.",
            "details": llm_result.get("warnings", [])
        }

    return {
        "status": "success",
        "pages_processed": pdf_result["pages_processed"],
        "parsed_results": parsed_results,
        "explanation": llm_result["explanation"],
        "warnings": pdf_result["warnings"],
        "total_seconds": round(
            time.perf_counter() - total_start, 2
        )
    }


if __name__ == "__main__":
    pdf_path = "sample_report.pdf"

    result = process_medical_report(pdf_path)

    print("\n========== PIPELINE RESULT ==========", flush=True)
    print("Status:", result["status"], flush=True)

    if result["status"] == "success":
        print("Pages processed:", result["pages_processed"])
        print("\nPatient-friendly explanation:\n")
        print(result["explanation"])

        print("\nWarnings:", result["warnings"])
        print("Total time:", result["total_seconds"], "seconds")

    else:
        print("Message:", result["message"])
        print("Details:", result.get("details", []))
