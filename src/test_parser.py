
import unittest

from report_parser import parse_report


class TestReportParser(unittest.TestCase):

    def test_numeric_result_and_reference_range(self):
        result = parse_report(
            "Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)"
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("report_id", result)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(len(result["results"]), 1)

        item = result["results"][0]
        self.assertEqual(item["test_name"], "Hemoglobin")
        self.assertEqual(item["value"], 13.5)
        self.assertIsInstance(item["value"], (int, float))
        self.assertEqual(item["unit"], "g/dL")
        self.assertEqual(item["reference_low"], 12.0)
        self.assertEqual(item["reference_high"], 16.0)
        self.assertIsNone(item["extraction_confidence"])

    def test_text_result(self):
        result = parse_report("HIV Screening: Negative")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["value"], "Negative")
        self.assertIsNone(result["results"][0]["unit"])

    def test_equals_separator(self):
        result = parse_report("Glucose = 95 mg/dL")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["test_name"], "Glucose")
        self.assertEqual(result["results"][0]["value"], 95.0)

    def test_multiple_results(self):
        text = """
        Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)
        WBC: 7500 cells/uL
        Glucose = 95 mg/dL
        """

        result = parse_report(text)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["results"]), 3)

    def test_empty_input(self):
        result = parse_report("")

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"], [])
        self.assertTrue(result["warnings"])

    def test_non_string_input(self):
        result = parse_report(None)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["results"], [])
        self.assertTrue(result["warnings"])

    def test_missing_reference_range(self):
        result = parse_report("Hemoglobin: 13.5 g/dL")

        item = result["results"][0]
        self.assertIsNone(item["reference_low"])
        self.assertIsNone(item["reference_high"])

    def test_patient_metadata_is_ignored(self):
        result = parse_report(
            "Patient Name: Rahul\nHemoglobin: 13.5 g/dL"
        )

        names = [item["test_name"] for item in result["results"]]
        self.assertEqual(names, ["Hemoglobin"])

    def test_unparseable_reference_range_is_not_invented(self):
        result = parse_report(
            "Hemoglobin: 13.5 g/dL (Reference: consult laboratory)"
        )

        item = result["results"][0]
        self.assertIsNone(item["reference_low"])
        self.assertIsNone(item["reference_high"])
        self.assertTrue(result["warnings"])

    def test_result_contract_fields(self):
        result = parse_report("Glucose: 95 mg/dL")

        self.assertEqual(
            set(result.keys()),
            {"status", "report_id", "results", "warnings"},
        )

        expected_fields = {
            "test_name",
            "value",
            "unit",
            "reference_low",
            "reference_high",
            "extraction_confidence",
        }

        self.assertEqual(set(result["results"][0].keys()), expected_fields)


if __name__ == "__main__":
    unittest.main()
