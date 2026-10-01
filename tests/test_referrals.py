"""Local synthetic directory tests: no LLM or network operations."""
from pathlib import Path
import unittest
from unittest.mock import patch

from backend import tools


class ReferralTests(unittest.TestCase):
    def test_lawyer_city_specialization_and_languages(self):
        for language in ("en", "hi", "mr"):
            records = tools.find_lawyers("property/land", "Pune", language)
            self.assertEqual([record["id"] for record in records], ["DEMO-L001"])
        self.assertEqual([record["id"] for record in tools.find_lawyers("family", "Pune", "en")], ["DEMO-L004"])

    def test_case_and_whitespace_normalization(self):
        self.assertEqual(tools.find_lawyers(" PROPERTY/LAND ", " PUNE ", " MR "),
                         tools.find_lawyers("property/land", "Pune", "mr"))
        self.assertEqual(tools.find_legal_aid("  mUMBAI "), tools.find_legal_aid("Mumbai"))

    def test_unknown_or_missing_city(self):
        for city in (None, "", " ", 8, "Unknown"):
            self.assertEqual(tools.find_lawyers("property/land", city, "en"), [])
            self.assertEqual(tools.find_legal_aid(city), [])

    def test_unknown_or_missing_specialization(self):
        for specialization in (None, "", " ", 8, "land_dispute", "property"):
            self.assertEqual(tools.find_lawyers(specialization, "Pune", "en"), [])

    def test_only_explicit_languages_match(self):
        self.assertEqual(tools.find_lawyers("property/land", "Mumbai", "hi"), [])
        self.assertEqual(tools.find_lawyers("property/land", "Delhi", "mr"), [])
        for language in (None, "", "English", "fr"):
            self.assertEqual(tools.find_lawyers("property/land", "Pune", language), [])

    def test_multiple_city_results(self):
        for city in ("Pune", "Mumbai", "Delhi"):
            self.assertTrue(tools.find_lawyers("property/land", city, "en"))
            self.assertTrue(tools.find_legal_aid(city))

    def test_returns_deep_copies(self):
        lawyers = tools.find_lawyers("property/land", "Pune", "en")
        lawyers[0]["languages"].clear()
        lawyers[0]["name"] = "Changed"
        aid = tools.find_legal_aid("Pune")
        aid[0]["languages"].clear()
        self.assertEqual(tools.find_lawyers("property/land", "Pune", "en")[0]["name"], "Fictional Demo Lawyer Alpha")
        self.assertIn("en", tools.find_legal_aid("Pune")[0]["languages"])

    def test_all_records_synthetic_and_noncontactable(self):
        for filename in ("lawyers.json", "legal_aid.json"):
            for record in tools._load_referral_directory(filename):
                self.assertEqual(record["source"], "synthetic")
                self.assertTrue(record["name"].startswith("Fictional Demo"))
                self.assertTrue(record["email"].endswith("@example.invalid"))
                self.assertNotIn("phone", record)

    def test_aid_does_not_assess_eligibility(self):
        record = tools.find_legal_aid("Pune")[0]
        self.assertIn("No eligibility assessment", record["eligibility_note"])
        self.assertNotIn("eligible", record)

    def test_missing_empty_directories_safe(self):
        for effect in (FileNotFoundError(), None):
            with patch.object(Path, "read_text", side_effect=effect, return_value=""):
                self.assertEqual(tools.find_legal_aid("Pune"), [])
                self.assertEqual(tools.find_lawyers("property/land", "Pune", "en"), [])

    def test_invalid_data_errors_sanitized(self):
        for text in ("broken", "{}", '[{"id":"PRIVATE_SENTINEL"}]'):
            with patch.object(Path, "read_text", return_value=text):
                with self.assertRaises(tools.ReferralDataError) as caught:
                    tools.find_legal_aid("Pune")
                self.assertNotIn("PRIVATE_SENTINEL", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
