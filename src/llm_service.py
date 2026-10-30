
import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
FALLBACK_MODEL_NAME = "gemini-3.5-flash-lite"
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


def _display_value(value):
    """Format a report value without changing its numeric meaning."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(value)
    return str(value)


def _range_based_fallback(parsed_report, flags, language, reason):
    """
    Provide a transparent, non-AI comparison if Gemini is unavailable.
    This uses only report values and the supplied reference limits/statuses.
    It does not infer diseases, causes, or treatment.
    """
    code = str(language).strip().lower()
    if code in {"hindi", "hi"}:
        code = "hi"
    elif code in {"kannada", "kn"}:
        code = "kn"
    else:
        code = "en"

    messages = {
        "en": {
            "summary": (
                "Gemini's AI explanation is temporarily unavailable. The notes below "
                "only compare extracted values with the reference limits printed in "
                "your report; they do not explain medical causes."
            ),
            "low": "{name}: {value} {unit} is below the report's lower reference limit ({low} {unit}).",
            "high": "{name}: {value} {unit} is above the report's upper reference limit ({high} {unit}).",
            "within_range": "{name}: {value} {unit} is within the reference range supplied in the report ({low} to {high} {unit}).",
            "unknown": "{name}: a safe comparison cannot be made because the report does not provide a complete or usable reference range.",
            "limit_1": "This is only a numerical comparison with the reference information extracted from the report.",
            "limit_2": "It is not a diagnosis. A healthcare professional must interpret results using the person's health history and clinical context.",
        },
        "hi": {
            "summary": (
                "Gemini की AI व्याख्या अभी उपलब्ध नहीं है। नीचे दिए गए नोट केवल रिपोर्ट "
                "में दी गई संदर्भ सीमाओं से संख्यात्मक तुलना करते हैं; ये चिकित्सकीय कारण नहीं बताते।"
            ),
            "low": "{name}: {value} {unit} रिपोर्ट में दी गई निचली संदर्भ सीमा ({low} {unit}) से कम है।",
            "high": "{name}: {value} {unit} रिपोर्ट में दी गई ऊपरी संदर्भ सीमा ({high} {unit}) से अधिक है।",
            "within_range": "{name}: {value} {unit} रिपोर्ट में दी गई संदर्भ सीमा ({low} से {high} {unit}) के भीतर है।",
            "unknown": "{name}: पूरी या उपयोगी संदर्भ सीमा उपलब्ध नहीं है, इसलिए सुरक्षित तुलना नहीं की जा सकती।",
            "limit_1": "यह केवल रिपोर्ट से निकाली गई संदर्भ जानकारी के साथ संख्यात्मक तुलना है।",
            "limit_2": "यह निदान नहीं है। परिणामों की व्याख्या स्वास्थ्य इतिहास और चिकित्सकीय जानकारी के आधार पर स्वास्थ्य विशेषज्ञ को करनी चाहिए।",
        },
        "kn": {
            "summary": (
                "Gemini AI ವಿವರಣೆ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ. ಕೆಳಗಿನ ಟಿಪ್ಪಣಿಗಳು ವರದಿಯಲ್ಲಿ "
                "ನೀಡಿರುವ ಉಲ್ಲೇಖ ಮಿತಿಗಳೊಂದಿಗೆ ಸಂಖ್ಯಾತ್ಮಕ ಹೋಲಿಕೆ ಮಾತ್ರ; ವೈದ್ಯಕೀಯ ಕಾರಣಗಳನ್ನು ವಿವರಿಸುವುದಿಲ್ಲ."
            ),
            "low": "{name}: {value} {unit} ವರದಿಯಲ್ಲಿ ನೀಡಿರುವ ಕೆಳಗಿನ ಉಲ್ಲೇಖ ಮಿತಿ ({low} {unit})ಗಿಂತ ಕಡಿಮೆಯಾಗಿದೆ.",
            "high": "{name}: {value} {unit} ವರದಿಯಲ್ಲಿ ನೀಡಿರುವ ಮೇಲಿನ ಉಲ್ಲೇಖ ಮಿತಿ ({high} {unit})ಗಿಂತ ಹೆಚ್ಚಾಗಿದೆ.",
            "within_range": "{name}: {value} {unit} ವರದಿಯಲ್ಲಿ ನೀಡಿರುವ ಉಲ್ಲೇಖ ಮಿತಿ ({low} ರಿಂದ {high} {unit}) ಒಳಗಿದೆ.",
            "unknown": "{name}: ಸಂಪೂರ್ಣ ಅಥವಾ ಉಪಯುಕ್ತ ಉಲ್ಲೇಖ ಮಿತಿ ಇಲ್ಲದ ಕಾರಣ ಸುರಕ್ಷಿತ ಹೋಲಿಕೆ ಮಾಡಲು ಸಾಧ್ಯವಿಲ್ಲ.",
            "limit_1": "ಇದು ವರದಿಯಿಂದ ಪಡೆದ ಉಲ್ಲೇಖ ಮಾಹಿತಿಯೊಂದಿಗೆ ಮಾಡಿದ ಸಂಖ್ಯಾತ್ಮಕ ಹೋಲಿಕೆ ಮಾತ್ರ.",
            "limit_2": "ಇದು ರೋಗನಿರ್ಣಯವಲ್ಲ. ವ್ಯಕ್ತಿಯ ಆರೋಗ್ಯ ಇತಿಹಾಸ ಮತ್ತು ವೈದ್ಯಕೀಯ ಸಂದರ್ಭವನ್ನು ಪರಿಗಣಿಸಿ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರು ಫಲಿತಾಂಶಗಳನ್ನು ಅರ್ಥೈಸಬೇಕು.",
        },
    }
    text = messages[code]
    raw_results = parsed_report.get("results", [])
    checked_results = flags.get("results", []) if isinstance(flags, dict) else []
    explanations = []

    for index, result in enumerate(raw_results):
        if not isinstance(result, dict):
            continue
        checked = (
            checked_results[index]
            if index < len(checked_results) and isinstance(checked_results[index], dict)
            else {}
        )
        status = checked.get("status", "unknown")
        if status not in {"low", "high", "within_range"}:
            status = "unknown"

        name = str(result.get("test_name", "Unknown test"))
        value = _display_value(result.get("value", "not supplied"))
        unit = str(result.get("unit") or "").strip()
        low_value = result.get("reference_low")
        high_value = result.get("reference_high")
        low = _display_value(low_value) if low_value is not None else "—"
        high = _display_value(high_value) if high_value is not None else "—"

        if status == "low":
            body = text["low"].format(name=name, value=value, unit=unit, low=low)
        elif status == "high":
            body = text["high"].format(name=name, value=value, unit=unit, high=high)
        elif status == "within_range":
            body = text["within_range"].format(name=name, value=value, unit=unit, low=low, high=high)
        else:
            body = text["unknown"].format(name=name, value=value, unit=unit)

        explanations.append({
            "test_name": name,
            "explanation": body,
            "limitations": [text["limit_1"], text["limit_2"]],
        })

    return {
        "status": "success",
        "summary": text["summary"],
        "explanations": explanations,
        "safety_message": SAFETY_MESSAGE,
        "warnings": ["AI explanation unavailable; showing range comparisons only."]
                    + ([str(reason)] if reason else []),
        "ai_generated": False,
        "explanation_mode": "range_comparison_fallback",
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

        response = None
        model_errors = []
        for model_name in (MODEL_NAME, FALLBACK_MODEL_NAME):
            try:
                print(f"Connecting to Gemini model: {model_name}", flush=True)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                        "response_mime_type": "application/json",
                    },
                )
                if response and response.text:
                    print(f"Gemini responded with {model_name}", flush=True)
                    break
                model_errors.append(f"{model_name}: empty response")
                response = None
            except Exception as model_exc:
                model_errors.append(f"{model_name}: {type(model_exc).__name__}")
                print(
                    f"Gemini model {model_name} failed: {type(model_exc).__name__}",
                    flush=True,
                )

        if response is None:
            return _range_based_fallback(
                parsed_report,
                flags,
                language,
                "; ".join(model_errors),
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
            return _range_based_fallback(
                parsed_report, flags, language, "Gemini returned an empty response."
            )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return _range_based_fallback(
                parsed_report, flags, language, "Gemini returned invalid JSON."
            )

        # Validate the required output schema.
        if not isinstance(data, dict):
            return _range_based_fallback(
                parsed_report, flags, language, "Gemini returned an invalid response format."
            )

        required_output = {
            "summary",
            "explanations",
            "safety_message",
        }

        if not required_output.issubset(data):
            return _range_based_fallback(
                parsed_report, flags, language, "Gemini's response was missing required fields."
            )

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
            "ai_generated": True,
            "explanation_mode": "gemini",
        }

    except Exception as exc:
        print(
            f"Gemini error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return _range_based_fallback(
            parsed_report,
            flags,
            language,
            f"Gemini request failed: {type(exc).__name__}",
        )

    finally:
        if client is not None:
            client.close()
