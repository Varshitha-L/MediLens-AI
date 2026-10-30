"""Parse laboratory results from line-based text and extracted PDF tables.

Supports:
- Flattened four-column tables (Test Name, Result, Unit, Reference Range)
- Vertical four-column tables produced by PDF text extraction
- Vertical five-column tables with an additional Flag column
- Simple lines such as: Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)

Administrative text and non-numeric entries are not treated as laboratory results.
Missing reference limits are never inferred.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
UNIT = r"(?:10\^[0-9]+/)?[A-Za-zµμ]+(?:/[A-Za-zµμ]+)?|%"
VALUE_PATTERN = re.compile(rf"^\s*({NUMBER})\s*(.*?)\s*$")
RANGE_PATTERN = re.compile(rf"^\s*({NUMBER})\s*-\s*({NUMBER})\s*$")
UPPER_ONLY_PATTERN = re.compile(rf"(?:less than|<)\s*({NUMBER})", re.IGNORECASE)
LOWER_ONLY_PATTERN = re.compile(rf"(?:greater than|>)\s*({NUMBER})", re.IGNORECASE)

HEADER_PATTERN = re.compile(
    r"\btest\s+name\s+result\s+unit\s+reference\s+range(?:\s+flag)?\b",
    re.IGNORECASE,
)

# A row in a flattened table: name, numeric value, unit, reference interval.
FLAT_ROW_PATTERN = re.compile(
    rf"(?P<name>[A-Za-z][A-Za-z0-9 .()/&_-]*?)\s+"
    rf"(?P<value>{NUMBER})\s+"
    rf"(?P<unit>{UNIT})\s+"
    rf"(?P<reference>(?:(?:desirable|optimal|normal)\s*:\s*)?"
    rf"(?:{NUMBER}\s*-\s*{NUMBER}|(?:less than|greater than|<=|>=|<|>)\s*{NUMBER}))",
    re.IGNORECASE,
)

# Simple line format: Test name: 13.5 g/dL (Reference: 12.0-16.0)
LINE_PATTERN = re.compile(
    rf"^\s*(?P<name>[^:=]+?)\s*[:=]\s*"
    rf"(?P<value>{NUMBER})\s*(?P<unit>{UNIT})?\s*"
    rf"(?:\(\s*(?:reference|ref)\s*:\s*(?P<reference>[^)]+)\s*\))?\s*$",
    re.IGNORECASE,
)

IGNORED_LABELS = {
    "age", "sex", "age / sex", "patient", "patient name", "patient id",
    "sample id", "specimen", "report date", "collection date", "report type",
    "report status", "laboratory note", "lab note", "testing note",
    "report note", "document note", "disclaimer", "remarks",
    "notes for software testing", "generated solely for demonstration/testing",
}

ADMIN_PREFIXES = (
    "laboratory note", "lab note", "testing note", "report status",
    "report note", "document note", "disclaimer", "remarks",
    "notes for software testing", "generated solely", "sample details",
)

FLAGS = re.compile(r"^(?:(?:HIGH|LOW|NORMAL|ABNORMAL)\s+|WITHIN\s+RANGE\s+)+", re.IGNORECASE)
TRAILING_FLAGS = re.compile(r"\s+(?:HIGH|LOW|NORMAL|ABNORMAL|WITHIN\s+RANGE)$", re.IGNORECASE)


def _is_ignored_name(name: str) -> bool:
    normalized = re.sub(r"\s+", " ", name).strip().lower()
    return (
        normalized in IGNORED_LABELS
        or any(normalized.startswith(prefix) for prefix in ADMIN_PREFIXES)
    )


def _parse_range(raw_reference: str | None) -> tuple[float | None, float | None, str | None]:
    if not raw_reference or not raw_reference.strip():
        return None, None, "reference range was not provided"

    raw_reference = raw_reference.strip()
    match = RANGE_PATTERN.match(raw_reference)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        if low > high:
            return None, None, f"reference range has reversed limits: {raw_reference}"
        return low, high, None

    match = UPPER_ONLY_PATTERN.search(raw_reference)
    if match:
        return None, float(match.group(1)), "one-sided upper reference limit; manual review needed"

    match = LOWER_ONLY_PATTERN.search(raw_reference)
    if match:
        return float(match.group(1)), None, "one-sided lower reference limit; manual review needed"

    return None, None, f"reference range could not be parsed: {raw_reference}"


def _build_result(
    test_name: str,
    raw_value: str,
    unit: str | None,
    raw_reference: str | None,
    line_number: int,
    warnings: list[str],
) -> dict[str, Any] | None:
    test_name = re.sub(r"\s+", " ", test_name).strip()
    test_name = FLAGS.sub("", test_name)
    test_name = TRAILING_FLAGS.sub("", test_name).strip()

    if not test_name or _is_ignored_name(test_name):
        return None

    value_match = VALUE_PATTERN.match(raw_value)
    if not value_match:
        # Never turn a note or arbitrary text into a lab result.
        return None

    value = float(value_match.group(1))
    embedded_unit = value_match.group(2).strip()
    parsed_unit = embedded_unit or (unit.strip() if unit and unit.strip() else None)

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


def _parse_vertical_tables(lines: list[str], warnings: list[str]) -> list[dict[str, Any]] | None:
    """Parse row-ordered PyMuPDF text where each cell is on a separate line."""
    normalized = [line.strip().lower() for line in lines]
    headers: list[tuple[int, int]] = []

    for index in range(len(normalized)):
        if normalized[index:index + 5] == [
            "test name", "result", "unit", "reference range", "flag"
        ]:
            headers.append((index, 5))
        elif normalized[index:index + 4] == [
            "test name", "result", "unit", "reference range"
        ]:
            headers.append((index, 4))

    if not headers:
        return None

    results: list[dict[str, Any]] = []

    for header_number, (header_index, stride) in enumerate(headers):
        start = header_index + stride
        end = headers[header_number + 1][0] if header_number + 1 < len(headers) else len(lines)
        index = start

        while index + 3 < end:
            test_name = lines[index].strip()
            raw_value = lines[index + 1].strip()
            unit = lines[index + 2].strip()
            reference = lines[index + 3].strip()

            if not VALUE_PATTERN.match(raw_value):
                break

            result = _build_result(test_name, raw_value, unit, reference, index + 1, warnings)
            if result:
                results.append(result)

            index += stride

    return results


def _parse_flattened_tables(text: str, warnings: list[str]) -> list[dict[str, Any]] | None:
    """Parse table text flattened into a single line or paragraphs."""
    normalized_text = re.sub(r"\s+", " ", text).strip()
    headers = list(HEADER_PATTERN.finditer(normalized_text))
    if not headers:
        return None

    results: list[dict[str, Any]] = []

    for header_index, header in enumerate(headers):
        segment_end = headers[header_index + 1].start() if header_index + 1 < len(headers) else len(normalized_text)
        segment = normalized_text[header.end():segment_end]

        for match in FLAT_ROW_PATTERN.finditer(segment):
            name = match.group("name").strip()
            # Ignore any table flags that may precede the next test name.
            result = _build_result(
                name,
                match.group("value"),
                match.group("unit"),
                match.group("reference"),
                header_index + 1,
                warnings,
            )
            if result:
                results.append(result)

    return results


def _parse_simple_lines(lines: list[str], warnings: list[str]) -> list[dict[str, Any]]:
    """Parse supported name:value lines; ignore administrative prose."""
    results: list[dict[str, Any]] = []

    for line_number, line in enumerate(lines, start=1):
        if line.lower().startswith("test result:"):
            line = line.split(":", 1)[1].strip()

        match = LINE_PATTERN.match(line)
        if not match:
            continue

        test_name = match.group("name").strip()
        if _is_ignored_name(test_name):
            continue

        result = _build_result(
            test_name,
            match.group("value"),
            match.group("unit"),
            match.group("reference"),
            line_number,
            warnings,
        )
        if result:
            results.append(result)

    return results


def parse_report(extracted_text: str) -> dict[str, Any]:
    """Return status, report_id, numeric test results and review warnings."""
    if not isinstance(extracted_text, str) or not extracted_text.strip():
        return {
            "status": "error",
            "report_id": None,
            "results": [],
            "warnings": ["No readable report text was found."],
        }

    warnings: list[str] = []
    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]

    # Prefer line-per-cell tables. If no such layout exists, try flattened tables.
    results = _parse_vertical_tables(lines, warnings)
    if results is None:
        results = _parse_flattened_tables(extracted_text, warnings)
    if results is None:
        results = _parse_simple_lines(lines, warnings)

    # Remove duplicate records while preserving order.
    unique_results: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str | None]] = set()
    for item in results:
        key = (item["test_name"].casefold(), item["value"], item["unit"])
        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    if not unique_results:
        return {
            "status": "error",
            "report_id": None,
            "results": [],
            "warnings": ["No numerical laboratory test results were found."],
        }

    return {
        "status": "partial" if warnings else "success",
        "report_id": f"report-{uuid.uuid4().hex[:12]}",
        "results": unique_results,
        "warnings": warnings,
    }


if __name__ == "__main__":
    example = """Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)
Fasting Blood Glucose: 135 mg/dL (Reference: 70-99)
"""
    import json
    print(json.dumps(parse_report(example), indent=2))