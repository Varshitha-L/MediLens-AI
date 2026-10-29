"""Rule-based comparison of parsed lab results with supplied reference limits.

This module is a decision-support utility, not a diagnostic system. It supports
full ranges (low and high), an upper limit only, or a lower limit only. When a
one-sided limit is supplied, it only flags values that cross that limit; values
on the other side remain unknown because the other bound is not available.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

VALID_STATUSES = {"low", "within_range", "high", "unknown"}


def _finite_number(value: Any) -> bool:
    """Return True for finite real numbers; reject booleans deliberately."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
    has_low = _finite_number(low)
    has_high = _finite_number(high)

    output = {
        "test_name": normalized_name or "Unknown test",
        "value": value if _finite_number(value) else None,
        "unit": normalized_unit or None,
        "reference_low": low if has_low else None,
        "reference_high": high if has_high else None,
        "status": "unknown",
        "message": "",
    }

    # Missing essential test fields prevent comparison.
    if not normalized_name:
        warnings.append("missing_test_name")
    if not _finite_number(value):
        warnings.append("missing_or_invalid_value")
    if not normalized_unit:
        warnings.append("missing_or_invalid_unit")

    # A supplied but nonnumeric bound is invalid. None means the report did
    # not provide that side of the reference range, which is handled below.
    if low is not None and not has_low:
        warnings.append("invalid_reference_low")
    if high is not None and not has_high:
        warnings.append("invalid_reference_high")
    if not has_low and not has_high and low is None and high is None:
        warnings.append("missing_or_invalid_reference_range")
    elif has_low != has_high:
        # Keep the established warning expected by existing tests and callers,
        # while also adding the more specific one-sided-limit review warning
        # during classification below.
        warnings.append("missing_or_invalid_reference_range")
    if has_low and has_high and float(low) > float(high):
        warnings.append("reference_range_low_greater_than_high")

    # Invalid fields mean we must not classify the result. A missing bound is
    # not itself a blocker when the supplied one-sided limit can still safely
    # identify a value above/below that limit; incomplete-range warnings are
    # retained for display and audit.
    invalid_warnings = [
        warning for warning in warnings
        if warning not in {
            "one_sided_upper_reference_limit_manual_review",
            "one_sided_lower_reference_limit_manual_review",
        }
        and not (warning == "missing_or_invalid_reference_range" and has_low != has_high)
    ]
    if invalid_warnings:
        output["message"] = (
            "Cannot safely compare this result; check the extracted fields "
            "and reference range."
        )
        return output, warnings

    numeric_value = float(value)

    # Full two-sided range: compare inclusively at the boundaries.
    if has_low and has_high:
        numeric_low = float(low)
        numeric_high = float(high)
        if numeric_value < numeric_low:
            output["status"] = "low"
            output["message"] = "Below the supplied reference range."
        elif numeric_value > numeric_high:
            output["status"] = "high"
            output["message"] = "Above the supplied reference range."
        else:
            output["status"] = "within_range"
            output["message"] = "Within the supplied reference range."

    # Upper-only limit, e.g. "Desirable: less than 200". A value above the
    # upper limit can be flagged high; otherwise the missing lower bound means
    # a full within-range conclusion would be unjustified.
    elif has_high:
        warnings.append("one_sided_upper_reference_limit_manual_review")
        if numeric_value > float(high):
            output["status"] = "high"
            output["message"] = "Above the supplied upper reference limit."
        else:
            output["status"] = "unknown"
            output["message"] = (
                "At or below the supplied upper limit, but the lower reference "
                "limit was not provided; manual review is needed."
            )

    # Lower-only limit. A value below it can be flagged low; otherwise the
    # missing upper bound prevents a full within-range conclusion.
    elif has_low:
        warnings.append("one_sided_lower_reference_limit_manual_review")
        if numeric_value < float(low):
            output["status"] = "low"
            output["message"] = "Below the supplied lower reference limit."
        else:
            output["status"] = "unknown"
            output["message"] = (
                "At or above the supplied lower limit, but the upper reference "
                "limit was not provided; manual review is needed."
            )

    return output, warnings


def check_abnormalities(parsed_report: dict) -> dict:
    """Compare parsed values with the report's supplied reference limits."""
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

    checked_results: list[dict[str, Any]] = []
    all_warnings: list[str] = []
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
