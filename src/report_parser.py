
import re


def parse_report(text):
    results = []

    pattern = (
        r"^(.+?)\s*[:=]\s*(.+?)"
        r"(?:\s*\(Reference:\s*([^)]+)\))?$"
    )

    value_pattern = r"^([+-]?\d+(?:\.\d+)?)\s*(.*)$"

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
        "sample type"
    }

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("test result:"):
            line = line.split(":", 1)[1].strip()

        match = re.search(pattern, line, re.IGNORECASE)

        if not match:
            continue

        test_name = match.group(1).strip()
        result = match.group(2).strip()

        reference_range = (
            match.group(3).strip()
            if match.group(3)
            else None
        )

        if test_name.lower() in ignored_labels:
            continue

        if not test_name or not result:
            continue

        value_match = re.match(value_pattern, result)

        if value_match:
            value = float(value_match.group(1))
            unit = value_match.group(2).strip() or None
        else:
            value = result
            unit = None

        results.append({
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "reference_range": reference_range
        })

    return results


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

    print(parse_report(sample_text))
