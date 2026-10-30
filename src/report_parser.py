"""Parse numeric laboratory results from common extracted PDF layouts.

Supports line-based results, inline rows, flattened tables and vertical tables.
It preserves report-supplied values and reference limits; missing limits are
never guessed. Unknown or ambiguous rows are left for manual review.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
UNIT = r"(?:[x×]?\s*10(?:\^[0-9]+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)/(?:[A-Za-zµμ]+)|[A-Za-zµμ]+[A-Za-zµμ0-9.^²³⁴]*(?:/[A-Za-zµμ0-9.^²³⁴]+)?|%)"
RANGE_TEXT = (
    rf"(?:{NUMBER}\s*(?:-|–|—|\bto\b)\s*{NUMBER}"
    rf"|(?:less than|up to|not more than|below|<=|<|≤|greater than|more than|above|>=|>|≥)\s*{NUMBER})"
)
VALUE_PATTERN = re.compile(rf"^\s*({NUMBER})\s*(.*?)\s*$")
RANGE_PATTERN = re.compile(
    rf"^\s*({NUMBER})\s*(?:-|–|—|to)\s*({NUMBER})"
    rf"(?:\s*[A-Za-zµμ%][A-Za-zµμ0-9./%^²³⁴]*)?\s*$",
    re.IGNORECASE,
)
UPPER_ONLY_PATTERN = re.compile(
    rf"(?:less than|up to|not more than|below|<=|<|≤)\s*({NUMBER})",
    re.IGNORECASE,
)
LOWER_ONLY_PATTERN = re.compile(
    rf"(?:greater than|more than|above|>=|>|≥)\s*({NUMBER})",
    re.IGNORECASE,
)

HEADER_PATTERN = re.compile(
    r"\b(?:test\s+name|investigation|parameter|analyte|test)\s+"
    r"(?:observed\s+value|measured\s+value|patient\s+value|result|value)\s+"
    r"(?:unit|units)\s+"
    r"(?:biological\s+reference\s+interval|reference\s+interval|"
    r"reference\s+range|normal\s+range|normal\s+values|reference\s+values)"
    r"(?:\s+flag)?\b",
    re.IGNORECASE,
)

FLAT_ROW_PATTERN = re.compile(
    rf"(?P<name>[\w][\w\s()./&,+-]*?)\s+"
    rf"(?P<value>{NUMBER})\s+"
    rf"(?P<unit>{UNIT})\s+"
    rf"(?P<reference>(?:(?:desirable|optimal|normal)\s*:\s*)?{RANGE_TEXT}"
    rf"(?:\s*[A-Za-zµμ%][A-Za-zµμ0-9./%^²³⁴]*)?)",
    re.IGNORECASE,
)

LINE_PATTERN = re.compile(
    rf"^\s*(?P<name>[^:=]+?)\s*[:=]\s*"
    rf"(?P<value>{NUMBER})\s*(?P<unit>{UNIT})?\s*"
    rf"(?:[\[(]\s*(?:(?:reference|ref|normal\s+range|reference\s+range)"
    rf"\s*[:=]?\s*)?(?P<reference>{RANGE_TEXT})\s*[\])])?\s*$",
    re.IGNORECASE,
)

IGNORED_LABELS = {
    "age", "sex", "gender", "age / sex", "patient", "patient name", "patient id",
    "sample id", "specimen", "report date", "collection date", "date of birth",
    "report type", "report status", "laboratory", "lab name", "doctor name",
    "laboratory note", "lab note", "testing note", "report note", "document note",
    "disclaimer", "remarks", "notes", "result", "results", "unit", "units",
    "reference range", "reference interval", "flag", "status",
    "notes for software testing", "generated solely for demonstration/testing",
}

ADMIN_PREFIXES = (
    "laboratory note", "lab note", "testing note", "report status",
    "report note", "document note", "disclaimer", "remarks",
    "notes for software testing", "generated solely", "sample details",
)

FLAGS = re.compile(
    r"^(?:(?:HIGH|LOW|NORMAL|ABNORMAL|H|L)\s+|WITHIN\s+RANGE\s+)+",
    re.IGNORECASE,
)
TRAILING_FLAGS = re.compile(
    r"\s+(?:HIGH|LOW|NORMAL|ABNORMAL|WITHIN\s+RANGE|H|L)$",
    re.IGNORECASE,
)


def _to_float(value: str) -> float:
    return float(value.replace(",", "").strip())


def _is_ignored_name(name: str) -> bool:
    normalized = re.sub(r"\s+", " ", name).strip().lower()
    return normalized in IGNORED_LABELS or any(
        normalized.startswith(prefix) for prefix in ADMIN_PREFIXES
    )


def _parse_range(
    raw_reference: str | None,
) -> tuple[float | None, float | None, str | None]:
    if not raw_reference or not raw_reference.strip():
        return None, None, "reference range was not provided"

    raw_reference = raw_reference.strip()
    match = RANGE_PATTERN.match(raw_reference)
    if match:
        low, high = _to_float(match.group(1)), _to_float(match.group(2))
        if low > high:
            return None, None, f"reference range has reversed limits: {raw_reference}"
        return low, high, None

    match = UPPER_ONLY_PATTERN.search(raw_reference)
    if match:
        return None, _to_float(match.group(1)), "one-sided upper reference limit; manual review needed"

    match = LOWER_ONLY_PATTERN.search(raw_reference)
    if match:
        return _to_float(match.group(1)), None, "one-sided lower reference limit; manual review needed"

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
        return None

    value = _to_float(value_match.group(1))
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


def _header_kind(value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    aliases = {
        "name": {"test name", "test", "investigation", "parameter", "analyte", "examination"},
        "value": {"result", "value", "observed value", "measured value", "patient value"},
        "unit": {"unit", "units"},
        "reference": {
            "reference range", "reference interval", "biological reference interval",
            "biological reference range", "normal range", "normal values", "reference values",
        },
        "flag": {"flag", "status", "remarks"},
    }
    for kind, choices in aliases.items():
        if normalized in choices:
            return kind
    return None


def _parse_vertical_tables(
    lines: list[str], warnings: list[str]
) -> list[dict[str, Any]] | None:
    """Parse tables where each header and each cell is on a separate line."""
    headers: list[tuple[int, int, dict[str, int]]] = []
    required = {"name", "value", "unit", "reference"}

    for index in range(max(0, len(lines) - 3)):
        for stride in (5, 4):
            cells = lines[index:index + stride]
            if len(cells) != stride:
                continue
            kinds = [_header_kind(cell) for cell in cells]
            if required.issubset(set(kinds)) and len([k for k in kinds if k]) == stride:
                positions = {kind: offset for offset, kind in enumerate(kinds) if kind}
                headers.append((index, stride, positions))
                break

    if not headers:
        return None

    results: list[dict[str, Any]] = []
    for header_number, (header_index, stride, positions) in enumerate(headers):
        start = header_index + stride
        end = headers[header_number + 1][0] if header_number + 1 < len(headers) else len(lines)
        index = start
        while index + stride - 1 < end:
            cells = [cell.strip() for cell in lines[index:index + stride]]
            name = cells[positions["name"]]
            raw_value = cells[positions["value"]]
            unit = cells[positions["unit"]]
            reference = cells[positions["reference"]]
            if not VALUE_PATTERN.match(raw_value):
                break
            result = _build_result(name, raw_value, unit, reference, index + 1, warnings)
            if result:
                results.append(result)
            index += stride

    return results


def _parse_flattened_tables(
    text: str, warnings: list[str]
) -> list[dict[str, Any]] | None:
    """Parse tables flattened into one line or paragraphs with a header."""
    normalized_text = re.sub(r"\s+", " ", text).strip()
    headers = list(HEADER_PATTERN.finditer(normalized_text))
    if not headers:
        return None

    results: list[dict[str, Any]] = []
    for header_index, header in enumerate(headers):
        segment_end = (
            headers[header_index + 1].start()
            if header_index + 1 < len(headers) else len(normalized_text)
        )
        segment = normalized_text[header.end():segment_end]
        for match in FLAT_ROW_PATTERN.finditer(segment):
            result = _build_result(
                match.group("name"),
                match.group("value"),
                match.group("unit"),
                match.group("reference"),
                header_index + 1,
                warnings,
            )
            if result:
                results.append(result)
    return results


def _parse_simple_lines(
    lines: list[str], warnings: list[str]
) -> list[dict[str, Any]]:
    """Parse named values, including common colon and inline row layouts."""
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        candidate = line.strip()
        if candidate.lower().startswith("test result:"):
            candidate = candidate.split(":", 1)[1].strip()

        match = LINE_PATTERN.match(candidate)
        if match:
            result = _build_result(
                match.group("name"),
                match.group("value"),
                match.group("unit"),
                match.group("reference"),
                line_number,
                warnings,
            )
            if result:
                results.append(result)
            continue

        # Some laboratory PDFs place all four cells on one text line.
        for row in FLAT_ROW_PATTERN.finditer(candidate):
            result = _build_result(
                row.group("name"),
                row.group("value"),
                row.group("unit"),
                row.group("reference"),
                line_number,
                warnings,
            )
            if result:
                results.append(result)

    return results


def parse_report(extracted_text: str) -> dict[str, Any]:
    """Return numeric laboratory results without inventing missing data."""
    if not isinstance(extracted_text, str) or not extracted_text.strip():
        return {
            "status": "error",
            "report_id": None,
            "results": [],
            "warnings": ["No readable report text was found."],
        }

    warnings: list[str] = []
    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]

    results = _parse_vertical_tables(lines, warnings)
    if results is None:
        results = _parse_flattened_tables(extracted_text, warnings)
    if results is None:
        results = _parse_simple_lines(lines, warnings)

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
            "warnings": ["No numerical laboratory test results were found in a supported layout."],
        }

    return {
        "status": "partial" if warnings else "success",
        "report_id": f"report-{uuid.uuid4().hex[:12]}",
        "results": unique_results,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import json

    examples = [
        "Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)",
        "WBC Count 7.5 10^9/L 4.0–11.0",
        "Fasting Blood Glucose = 135 mg/dL (Ref: 70 to 99)",
    ]
    for example in examples:
        print(json.dumps(parse_report(example), indent=2))
