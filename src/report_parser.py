"""Parse laboratory results from line-based text or simple extracted PDF tables.

The parser intentionally does not infer missing reference limits. One-sided ranges
(e.g. "less than 200") are retained as partial information and flagged for review.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

NUMBER = r"[+-]?\d+(?:\.\d+)?"
LINE_PATTERN = re.compile(
    rf"^(.+?)\s*[:=]\s*(.+?)"
    r"(?:\s*\(\s*Reference:\s*([^)]+)\s*\))?$",
    re.IGNORECASE,
)
VALUE_PATTERN = re.compile(rf"^({NUMBER})\s*(.*?)$")
RANGE_PATTERN = re.compile(rf"^\s*({NUMBER})\s*-\s*({NUMBER})\s*$")
UPPER_ONLY_PATTERN = re.compile(rf"(?:less than|<)\s*({NUMBER})", re.IGNORECASE)
LOWER_ONLY_PATTERN = re.compile(rf"(?:greater than|>)\s*({NUMBER})", re.IGNORECASE)

IGNORED_LABELS = {
    "patient name", "patient id", "patient age", "patient gender", "date",
    "report date", "test result", "laboratory", "lab name", "doctor name",
    "specimen", "sample type", "desirable", "test name", "result", "unit",
    "reference range", "flag",
}


def _parse_range(raw_reference: str | None) -> tuple[float | None, float | None, str | None]:
    if not raw_reference:
        return None, None, "reference range was not provided"
    m = RANGE_PATTERN.match(raw_reference)
    if m:
        return float(m.group(1)), float(m.group(2)), None
    m = UPPER_ONLY_PATTERN.search(raw_reference)
    if m:
        return None, float(m.group(1)), "one-sided upper reference limit; manual review needed"
    m = LOWER_ONLY_PATTERN.search(raw_reference)
    if m:
        return float(m.group(1)), None, "one-sided lower reference limit; manual review needed"
    return None, None, f"reference range could not be parsed: {raw_reference}"


def _build_result(test_name: str, raw_value: str, unit: str | None,
                  raw_reference: str | None, line_number: int,
                  warnings: list[str]) -> dict[str, Any] | None:
    test_name = test_name.strip()
    raw_value = raw_value.strip()
    if not test_name or test_name.lower() in IGNORED_LABELS or not raw_value:
        return None

    value_match = VALUE_PATTERN.match(raw_value)
    if value_match:
        value: float | str = float(value_match.group(1))
        parsed_unit = value_match.group(2).strip() or (unit.strip() if unit else None)
    else:
        value = raw_value
        parsed_unit = unit.strip() if unit and unit.strip() else None
        warnings.append(f"Line {line_number}: result value is not a numeric value; manual review needed.")

    low, high, range_warning = _parse_range(raw_reference)
    if range_warning:
        warnings.append(f"Line {line_number}: {range_warning}.")

    if not parsed_unit:
        warnings.append(f"Line {line_number}: unit was not provided; manual review needed.")

    return {
        "test_name": test_name,
        "value": value,
        "unit": parsed_unit,
        "reference_low": low,
        "reference_high": high,
        "extraction_confidence": None,
    }


def _parse_table(lines: list[str], warnings: list[str]) -> list[dict[str, Any]] | None:
    """Parse PyMuPDF's row-ordered text extracted from a simple five-column table."""
    header_index = None
    for i in range(max(0, len(lines) - 4)):
        normalized = [x.strip().lower() for x in lines[i:i + 5]]
        if normalized == ["test name", "result", "unit", "reference range", "flag"]:
            header_index = i
            break
    if header_index is None:
        return None

    results: list[dict[str, Any]] = []
    i = header_index + 5
    while i + 3 < len(lines):
        test_name = lines[i].strip()
        if not test_name or test_name.lower().startswith(("notes for", "generated solely", "medilens ai |", "page ")):
            break
        # Table rows have four data fields followed by a displayed flag.
        raw_value = lines[i + 1].strip()
        unit = lines[i + 2].strip()
        raw_reference = lines[i + 3].strip()
        if not VALUE_PATTERN.match(raw_value):
            break
        result = _build_result(test_name, raw_value, unit, raw_reference, i + 1, warnings)
        if result:
            results.append(result)
        i += 5 if i + 4 < len(lines) else 4
    return results


def parse_report(extracted_text: str) -> dict[str, Any]:
    """Return status, report_id, numeric test results, and review warnings."""
    if not isinstance(extracted_text, str):
        return {"status": "error", "report_id": None, "results": [], "warnings": ["extracted_text must be a string."]}
    if not extracted_text.strip():
        return {"status": "partial", "report_id": None, "results": [], "warnings": ["No report text was provided."]}

    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    warnings: list[str] = []

    # Prefer table parsing when the extractor supplies a recognizable table header;
    # this avoids treating table notes such as "Desirable: less than 200" as a test.
    table_results = _parse_table(lines, warnings)
    if table_results is not None:
        results = table_results
    else:
        results = []
        for line_number, line in enumerate(lines, start=1):
            if line.lower().startswith("test result:"):
                line = line.split(":", 1)[1].strip()
            match = LINE_PATTERN.match(line)
            if not match:
                continue
            test_name = match.group(1).strip()
            raw_result = match.group(2).strip()
            raw_reference = match.group(3)
            if test_name.lower() in IGNORED_LABELS:
                continue
            result = _build_result(test_name, raw_result, None, raw_reference, line_number, warnings)
            if result:
                results.append(result)

    if not results:
        warnings.append("No recognizable medical test results were found.")
        status = "partial"
    else:
        status = "partial" if warnings else "success"

    return {
        "status": status,
        "report_id": f"report-{uuid.uuid4().hex[:12]}",
        "results": results,
        "warnings": warnings,
    }


if __name__ == "__main__":
    example = """Fasting Blood Glucose: 135 mg/dL (Reference: 70-99)
Hemoglobin: 13.2 g/dL (Reference: 12.0-16.0)
"""
    import json
    print(json.dumps(parse_report(example), indent=2))
