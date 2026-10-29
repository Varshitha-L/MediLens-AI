
import re
import uuid


def parse_report(extracted_text):
    """
    Convert extracted medical report text into the MediLens AI
    parsed-report contract.

    Returns a dictionary with status, report_id, results, warnings.
    """

    if not isinstance(extracted_text, str):
        return {
            "status": "error",
            "report_id": None,
            "results": [],
            "warnings": ["extracted_text must be a string."],
        }

    if not extracted_text.strip():
        return {
            "status": "partial",
            "report_id": None,
            "results": [],
            "warnings": ["No report text was provided."],
        }

    results = []
    warnings = []

    pattern = re.compile(
        r"^(.+?)\s*[:=]\s*(.+?)"
        r"(?:\s*\(\s*Reference:\s*([^)]+)\s*\))?$",
        re.IGNORECASE,
    )

    value_pattern = re.compile(
        r"^([+-]?\d+(?:\.\d+)?)\s*(.*?)$"
    )

    range_pattern = re.compile(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*$"
    )

    ignored_labels = {
        "patient name",
        "patient id",
        "patient age",
        "patient gender",
        "date",
        "report date",
        "test result",
        "laboratory",
        "lab name",
        "doctor name",
        "specimen",
        "sample type",
    }

    for line_number, line in enumerate(
        extracted_text.splitlines(), start=1
    ):
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("test result:"):
            line = line.split(":", 1)[1].strip()

        match = pattern.match(line)

        if not match:
            continue

        test_name = match.group(1).strip()
        raw_result = match.group(2).strip()
        raw_reference = match.group(3)

        if test_name.lower() in ignored_labels:
            continue

        if not test_name or not raw_result:
            warnings.append(
                f"Line {line_number}: incomplete test result skipped."
            )
            continue

        value_match = value_pattern.match(raw_result)

        if value_match:
            value = float(value_match.group(1))
            unit = value_match.group(2).strip() or None
        else:
            value = raw_result
            unit = None

        reference_low = None
        reference_high = None

        if raw_reference:
            range_match = range_pattern.match(raw_reference)

            if range_match:
                reference_low = float(range_match.group(1))
                reference_high = float(range_match.group(2))
            else:
                warnings.append(
                    f"Line {line_number}: reference range could "
                    "not be parsed; manual review may be needed."
                )

        results.append({
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "reference_low": reference_low,
            "reference_high": reference_high,
            "extraction_confidence": None,
        })

    if results and warnings:
        status = "partial"
    elif results:
        status = "success"
    else:
        status = "partial"
        warnings.append(
            "No recognizable medical test results were found."
        )

    return {
        "status": status,
        "report_id": f"report-{uuid.uuid4().hex[:12]}",
        "results": results,
        "warnings": warnings,
    }


if __name__ == "__main__":
    sample_text = """
    COMPLETE BLOOD COUNT
    Patient Name: Rahul
    Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)
    WBC: 7500 cells/uL
    Glucose = 95 mg/dL
    Blood Group: Positive
    HIV Screening: Negative
    Test result: Platelets = 250000 cells/uL
    """

    import json
    print(json.dumps(parse_report(sample_text), indent=2))
