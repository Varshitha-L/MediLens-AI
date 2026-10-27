
import unittest

from report_parser import parse_report


class TestReportParser(unittest.TestCase):

    def test_numeric_result_with_reference_range(self):
        text = "Hemoglobin: 13.5 g/dL (Reference: 12.0-16.0)"

        result = parse_report(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["test_name"], "Hemoglobin")
        self.assertEqual(result[0]["value"], 13.5)
        self.assertEqual(result[0]["unit"], "g/dL")
        self.assertEqual(
            result[0]["reference_range"], "12.0-16.0"
        )

    def test_equals_separator(self):
        result = parse_report("Glucose = 95 mg/dL")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["test_name"], "Glucose")
        self.assertEqual(result[0]["value"], 95.0)
        self.assertEqual(result[0]["unit"], "mg/dL")

    def test_text_result(self):
        result = parse_report("HIV Screening: Negative")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], "Negative")
        self.assertIsNone(result[0]["unit"])

    def test_missing_reference_range(self):
        result = parse_report("Hemoglobin: 13.5 g/dL")

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["reference_range"])

    def test_empty_input(self):
        self.assertEqual(parse_report(""), [])

    def test_irrelevant_lines_are_ignored(self):
        text = """
        COMPLETE BLOOD COUNT
        Patient Name: Rahul
        Hemoglobin: 13.5 g/dL
        """

        result = parse_report(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["test_name"], "Hemoglobin")

    def test_multiple_results(self):
        text = """
        Hemoglobin: 13.5 g/dL
        Glucose: 95 mg/dL
        HIV Screening: Negative
        """

        result = parse_report(text)

        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
