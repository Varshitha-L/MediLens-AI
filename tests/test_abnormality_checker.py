import pytest
from src.abnormality_checker import check_abnormalities


def test_low_normal_high_and_inclusive_bounds():
    report = {"results": [
        {"test_name": "A", "value": 9, "unit": "u", "reference_low": 10, "reference_high": 20},
        {"test_name": "B", "value": 10, "unit": "u", "reference_low": 10, "reference_high": 20},
        {"test_name": "C", "value": 20, "unit": "u", "reference_low": 10, "reference_high": 20},
        {"test_name": "D", "value": 21, "unit": "u", "reference_low": 10, "reference_high": 20},
    ]}
    result = check_abnormalities(report)
    assert [r["status"] for r in result["results"]] == [
        "low", "within_range", "within_range", "high"
    ]
    assert result["flags"] == {"low": 1, "within_range": 2, "high": 1, "unknown": 0}


@pytest.mark.parametrize("bad_value", [None, "abc", True, float("nan"), float("inf")])
def test_invalid_values_become_unknown(bad_value):
    result = check_abnormalities({"results": [{
        "test_name": "A", "value": bad_value, "unit": "u",
        "reference_low": 1, "reference_high": 2
    }]})
    assert result["results"][0]["status"] == "unknown"
    assert result["flags"]["unknown"] == 1


def test_missing_reference_range_is_unknown():
    result = check_abnormalities({"results": [{
        "test_name": "A", "value": 1.5, "unit": "u",
        "reference_low": None, "reference_high": 2
    }]})
    assert result["results"][0]["status"] == "unknown"
    assert "missing_or_invalid_reference_range" in result["warnings"][0]


def test_reversed_reference_range_is_unknown():
    result = check_abnormalities({"results": [{
        "test_name": "A", "value": 1.5, "unit": "u",
        "reference_low": 2, "reference_high": 1
    }]})
    assert result["results"][0]["status"] == "unknown"


def test_bad_report_shapes_fail_safely():
    assert check_abnormalities(None)["status"] == "error"
    assert check_abnormalities({})["warnings"] == ["results_must_be_a_list"]
    assert check_abnormalities({"results": "not-a-list"})["status"] == "error"


def test_non_dictionary_result_does_not_crash():
    result = check_abnormalities({"results": ["bad row"]})
    assert result["results"][0]["status"] == "unknown"
    assert result["flags"]["unknown"] == 1


def test_empty_results_are_valid_empty_result():
    result = check_abnormalities({"results": []})
    assert result["status"] == "ok"
    assert result["flags"] == {"low": 0, "within_range": 0, "high": 0, "unknown": 0}
