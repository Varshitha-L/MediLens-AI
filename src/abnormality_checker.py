"""Rule-based comparison of parsed laboratory results with supplied reference ranges.

This module is a decision-support utility, not a diagnostic system.
Expected input:
{
    "results": [
        {
            "test_name": "Hemoglobin",
            "value": 13.2,
            "unit": "g/dL",
            "reference_low": 12.0,
            "reference_high": 16.0
        }
    ]
}
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

VALID_STATUSES = {"low", "within_range", "high", "unknown"}


def _finite_number(value: Any) -> bool:
    """True for finite real numbers; booleans are deliberately rejected."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _check_one_result(result: Any) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    if not isinstance(result, dict):
        return {
            "test_name": None,
            "value": None,
            "unit": None,
            "reference_low": None,
            "reference_high": None,
            "status": "unknown",
            "message": "Result must be an object/dictionary.",
        }, ["invalid_result_type"]

    test_name = result.get("test_name")
    value = result.get("value")
    unit = result.get("unit")
    low = result.get("reference_low")
    high = result.get("reference_high")

    normalized_name = test_name.strip() if isinstance(test_name, str) else None
    normalized_unit = unit.strip() if isinstance(unit, str) else None

    output = {
        "test_name": normalized_name or "Unknown test",
        "value": value if _finite_number(value) else None,
        "unit": normalized_unit or None,
        "reference_low": low if _finite_number(low) else None,
        "reference_high": high if _finite_number(high) else None,
        "status": "unknown",
        "message": "",
    }

    if not normalized_name:
        warnings.append("missing_test_name")
    if not _finite_number(value):
        warnings.append("missing_or_invalid_value")
    if not normalized_unit:
        warnings.append("missing_or_invalid_unit")
    if not _finite_number(low) or not _finite_number(high):
        warnings.append("missing_or_invalid_reference_range")
    elif float(low) > float(high):
        warnings.append("reference_range_low_greater_than_high")

    # A result is only comparable if all essential fields are valid.
    if warnings:
        output["message"] = "Cannot safely compare this result; check the extracted fields and reference range."
        return output, warnings

    numeric_value = float(value)
    numeric_low = float(low)
    numeric_high = float(high)

    # Bounds are inclusive: a value equal to either bound is within range.
    if numeric_value < numeric_low:
        output["status"] = "low"
        output["message"] = "Below the supplied reference range."
    elif numeric_value > numeric_high:
        output["status"] = "high"
        output["message"] = "Above the supplied reference range."
    else:
        output["status"] = "within_range"
        output["message"] = "Within the supplied reference range."

    return output, warnings


def check_abnormalities(parsed_report: dict) -> dict:
    """Check parsed report values against the ranges supplied with each result.

    Returns a stable, JSON-serializable dictionary with overall status,
    normalized results, flags, and warnings. Never invents a reference range.
    """
    if not isinstance(parsed_report, dict):
        return {
            "status": "error",
            "results": [],
            "flags": {"low": 0, "within_range": 0, "high": 0, "unknown": 0},
            "warnings": ["parsed_report_must_be_a_dictionary"],
        }

    raw_results = parsed_report.get("results")
    if not isinstance(raw_results, list):
        return {
            "status": "error",
            "results": [],
            "flags": {"low": 0, "within_range": 0, "high": 0, "unknown": 0},
            "warnings": ["results_must_be_a_list"],
        }

    checked_results = []
    all_warnings = []
    flags = {"low": 0, "within_range": 0, "high": 0, "unknown": 0}

    for index, result in enumerate(raw_results):
        checked, warnings = _check_one_result(result)
        checked_results.append(checked)
        flags[checked["status"]] += 1
        all_warnings.extend(f"results[{index}]:{warning}" for warning in warnings)

    overall_status = "ok" if not all_warnings else "completed_with_warnings"
    return {
        "status": overall_status,
        "results": checked_results,
        "flags": flags,
        "warnings": all_warnings,
    }
