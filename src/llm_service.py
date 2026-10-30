
import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
REQUEST_TIMEOUT_MS = 30000
MAX_OUTPUT_TOKENS = 1500

SAFETY_MESSAGE = (
    "This tool supports health awareness and does not replace "
    "professional medical advice."
)


def _error_result(warnings):
    """Return a consistent error response."""
    return {
        "status": "error",
        "summary": "",
        "explanations": [],
        "safety_message": SAFETY_MESSAGE,
        "warnings": warnings,
    }


def explain_report(parsed_report, flags, language):
    """
    Explain parsed medical results using Gemini.

    Args:
        parsed_report: Parsed-report contract dictionary.
        flags: Abnormality-checker output dictionary.
        language: Desired explanation language.

    Returns:
        Dictionary following the MediLens AI LLM contract.
    """

    # Validate the parsed-report input.
    if not isinstance(parsed_report, dict):
        return _error_result(["parsed_report must be a dictionary."])

    if parsed_report.get("status") not in ("success", "partial"):
        return _error_result([
            "The parsed report is not in a usable state."
        ])

    results = parsed_report.get("results")

    if not isinstance(results, list) or not results:
        return _error_result([
            "The parsed report contains no medical results."
        ])

    if not isinstance(flags, dict):
        return _error_result([
            "flags must be an abnormality-checker dictionary."
        ])

    if not isinstance(language, str) or not language.strip():
        return _error_result(["A valid output language is required."])

    required_fields = {
        "test_name",
        "value",
        "unit",
        "reference_low",
        "reference_high",
        "extraction_confidence",
    }

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            return _error_result([
                f"Result {index + 1} must be a dictionary."
            ])

        missing = required_fields - result.keys()
        if missing:
            return _error_result([
                f"Result {index + 1} is missing fields: "
                + ", ".join(sorted(missing))
            ])

        if not isinstance(result["test_name"], str) or not result["test_name"].strip():
            return _error_result([
                f"Result {index + 1} has an invalid test name."
            ])

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _error_result([
            "GEMINI_API_KEY was not found. Check your .env file."
        ])

    prompt_data = {
        "parsed_report": parsed_report,
        "flags": flags,
        "language": language.strip(),
    }

    prompt = f"""
You are MediLens AI, a medical information assistant.

Use only the report data and flags supplied below.
Write a concise, patient-friendly explanation in the requested language.

INPUT:
{json.dumps(prompt_data, ensure_ascii=False, allow_nan=False)}

Return ONLY one valid JSON object with this exact structure:
{{
  "summary": "A concise overview",
  "explanations": [
    {{
      "test_name": "Exact supplied test name",
      "explanation": "Patient-friendly explanation",
      "limitations": ["Relevant limitation"]
    }}
  ],
  "safety_message": "{SAFETY_MESSAGE}"
}}

Rules:
- Include an explanation for each supplied test result.
- Preserve test names, values and units; never alter report data.
- Do not invent reference ranges or clinical facts.
- If reference ranges are missing, state that they were not provided.
- Do not diagnose diseases or recommend medication or treatment.
- Mention uncertainty and missing clinical context where relevant.
- Do not invent confidence scores or citations.
- Keep the summary and explanations concise.
- Do not add Markdown fences or extra JSON keys.
"""

    client = None
    start_time = time.perf_counter()

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS
            ),
        )

        print("Connecting to Gemini...", flush=True)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "response_mime_type": "application/json",
            },
        )

        candidates = response.candidates or []
        if not candidates:
            return _error_result([
                "Gemini returned no response candidates."
            ])

        finish_reason = candidates[0].finish_reason
        print("Finish reason:", finish_reason, flush=True)

        if str(finish_reason).endswith("MAX_TOKENS"):
            return _error_result([
                "Gemini's response was truncated. Retry with an "
                "appropriate output-token limit."
            ])

        raw_text = response.text or ""
        if not raw_text.strip():
            return _error_result([
                "Gemini returned an empty response."
            ])

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return _error_result([
                "Gemini returned invalid JSON."
            ])

        # Validate the required output schema.
        if not isinstance(data, dict):
            return _error_result([
                "Gemini's response must be a JSON object."
            ])

        required_output = {
            "summary",
            "explanations",
            "safety_message",
        }

        if not required_output.issubset(data):
            return _error_result([
                "Gemini's response is missing required fields."
            ])

        if not isinstance(data["summary"], str):
            return _error_result(["summary must be a string."])

        if not isinstance(data["explanations"], list):
            return _error_result(["explanations must be a list."])

        if not isinstance(data["safety_message"], str):
            return _error_result(["safety_message must be a string."])

        for item in data["explanations"]:
            if not isinstance(item, dict):
                return _error_result([
                    "Each explanation must be a dictionary."
                ])

            if not all(
                isinstance(item.get(key), str)
                for key in ("test_name", "explanation")
            ):
                return _error_result([
                    "Each explanation needs string test_name "
                    "and explanation fields."
                ])

            if not isinstance(item.get("limitations"), list) or not all(
                isinstance(value, str)
                for value in item["limitations"]
            ):
                return _error_result([
                    "Each explanation needs a list of string limitations."
                ])

        # Ensure the model has not omitted a supplied test.
        expected_names = [item["test_name"] for item in results]
        actual_names = [
            item["test_name"] for item in data["explanations"]
        ]

        if actual_names != expected_names:
            return _error_result([
                "Gemini's explanation test names do not match "
                "the parsed report."
            ])

        print(
            f"Gemini responded in "
            f"{time.perf_counter() - start_time:.2f}s",
            flush=True,
        )

        return {
            "status": "success",
            "summary": data["summary"],
            "explanations": data["explanations"],
            "safety_message": data["safety_message"],
            "warnings": [],
        }

    except Exception as exc:
        print(
            f"Gemini error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return _error_result([
            f"Gemini request failed: {type(exc).__name__}: {exc}"
        ])

    finally:
        if client is not None:
            client.close()
