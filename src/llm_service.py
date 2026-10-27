
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
REQUEST_TIMEOUT_MS = 30000


def explain_results(parsed_results, language="English"):
    """
    Generate a patient-friendly explanation of parsed medical results.

    Returns:
        {
            "status": "success" or "error",
            "explanation": str,
            "warnings": list
        }
    """

    # Validate input
    if not isinstance(parsed_results, list) or not parsed_results:
        return {
            "status": "error",
            "explanation": "",
            "warnings": ["No parsed medical results were provided."]
        }

    clean_results = []

    required_fields = {
        "test_name",
        "value",
        "unit",
        "reference_range"
    }

    for index, result in enumerate(parsed_results):
        if not isinstance(result, dict):
            return {
                "status": "error",
                "explanation": "",
                "warnings": [
                    f"Result {index + 1} must be a dictionary."
                ]
            }

        if not required_fields.issubset(result):
            return {
                "status": "error",
                "explanation": "",
                "warnings": [
                    f"Result {index + 1} is missing required fields."
                ]
            }

        if not str(result["test_name"]).strip():
            return {
                "status": "error",
                "explanation": "",
                "warnings": [
                    f"Result {index + 1} has no test name."
                ]
            }

        clean_results.append({
            "test_name": str(result["test_name"]),
            "value": result["value"],
            "unit": result["unit"],
            "reference_range": result["reference_range"]
        })

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "status": "error",
            "explanation": "",
            "warnings": [
                "GEMINI_API_KEY was not found. Check your .env file."
            ]
        }

    prompt = f"""
You are MediLens AI, a medical information assistant.

Explain the following reported test results in {language}
using concise, patient-friendly language.

Results:
{clean_results}

Instructions:
- Explain what each test measures.
- Preserve all supplied values and units exactly.
- Compare a result with a reference range only when supplied.
- If a range is missing, explicitly say it was not provided.
- Never invent medical information or diagnose a disease.
- Mention that results need interpretation in clinical context.
- Do not recommend medication or treatment.

Keep the response under 180 words.
Use headings: Overview, Test Explanation, Limitations, Next Step.
"""

    client = None
    start_time = time.perf_counter()

    try:
        print("   Connecting to Gemini...", flush=True)

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS
            )
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 800
            }
        )

        print("Finish reason:", response.candidates[0].finish_reason)
        print("Raw response:", repr(response.text))

        finish_reason = response.candidates[0].finish_reason
        explanation = response.text or ""

        print("Finish reason:", finish_reason, flush=True)

        if str(finish_reason).endswith("MAX_TOKENS"):
            return {
                "status": "error",
                "explanation": "",
                "warnings": [
                    "Gemini's response was truncated. Increase the token limit."
                ]
            }

        if not explanation.strip():
            return {
                "status": "error",
                "explanation": "",
                "warnings": ["Gemini returned an empty response."]
            }

        if not explanation or not explanation.strip():
            return {
                "status": "error",
                "explanation": "",
                "warnings": [
                    "Gemini returned an empty response."
                ]
            }

        print(
            f"   Gemini responded in "
            f"{time.perf_counter() - start_time:.2f}s",
            flush=True
        )

        return {
            "status": "success",
            "explanation": explanation.strip(),
            "warnings": []
        }

    except Exception as e:
        # Print the actual error for debugging.
        # Never print the API key.
        print(
            f"Gemini error details: {type(e).__name__}: {e}",
            flush=True
        )

        return {
            "status": "error",
            "explanation": "",
            "warnings": [
                f"Gemini request failed: {type(e).__name__}: {e}"
            ]
        }

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sample_results = [
        {
            "test_name": "Hemoglobin",
            "value": 13.5,
            "unit": "g/dL",
            "reference_range": None
        }
    ]

    result = explain_results(sample_results)

    print("\nStatus:", result["status"])

    if result["status"] == "success":
        print("\nExplanation:\n")
        print(result["explanation"])
    else:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print("-", warning)
