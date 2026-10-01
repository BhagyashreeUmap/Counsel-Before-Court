"""Run from the project root: python -m unittest discover -s tests -v."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.database import CaseDataError, load_cases
from backend import tools
from backend.tools import get_case, search_cases

INHERITANCE_QUERY = (
    "My father died and now my uncle is claiming our farmland. "
    "The land record is still in my father's name."
)


class RetrievalTests(unittest.TestCase):
    def test_loading(self):
        cases = load_cases()
        self.assertEqual(len(cases), 15)
        self.assertTrue(all(case["source"] == "synthetic" for case in cases))

    def test_loading_from_another_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(len(load_cases()), 15)
            finally:
                os.chdir(original)

    def test_bad_data_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            with self.assertRaisesRegex(CaseDataError, "not found"):
                load_cases(path)
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(CaseDataError, "Invalid JSON"):
                load_cases(path)
            for value in ({}, [1], [{}], [{"id": "X"}, {"id": "X"}]):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(CaseDataError):
                        load_cases(path)

    def test_get_case(self):
        case = get_case("C003")
        self.assertEqual(case["id"], "C003")
        self.assertIn("Synthetic", case["retrieval_notice"])
        self.assertIsNone(get_case("missing"))
        self.assertIsNone(get_case(None))

    def test_empty_and_unrelated_query(self):
        for query in ("", "  \n ", "quasar zeppelin"):
            self.assertEqual(search_cases(query), [])

    def test_k_and_scores(self):
        for k in (1, 3, 100):
            results = search_cases(INHERITANCE_QUERY, k=k)
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), min(k, 15))
            scores = [case["similarity_score"] for case in results]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(0 < score <= 1.0000001 for score in scores))
        self.assertEqual(len(search_cases(INHERITANCE_QUERY, k=3)), 3)
        self.assertEqual(search_cases(INHERITANCE_QUERY, k=0), [])
        self.assertEqual(search_cases(INHERITANCE_QUERY, k=-1), [])
        with self.assertRaises(TypeError):
            search_cases("land", k=1.5)
        with self.assertRaises(TypeError):
            search_cases(None)

    def test_inheritance_match(self):
        self.assertEqual(search_cases(INHERITANCE_QUERY)[0]["id"], "C003")

    def test_boundary_match(self):
        results = search_cases("My neighbour moved the boundary fence into my plot.")
        self.assertEqual(results[0]["id"], "C004")

    def test_results_are_independent(self):
        before = load_cases()
        case = get_case("C003")
        case["documents_citizen_had"].append("changed")
        result = search_cases(INHERITANCE_QUERY)[0]
        result["documents_citizen_had"].clear()
        self.assertEqual(get_case("C003")["documents_citizen_had"], before[2]["documents_citizen_had"])
        self.assertNotIn("similarity_score", get_case("C003"))
        self.assertEqual(load_cases(), before)

    def test_empty_index_and_field_normalization(self):
        for cases in ([], [{"id": "X", "title": None}], [
            {"id": "X", "title": "boundary", "documents_citizen_had": [None, 12, "fence"]}
        ]):
            with self.subTest(cases=cases), patch.object(tools, "_cases", return_value=cases):
                tools._search_index.cache_clear()
                try:
                    results = search_cases("boundary fence")
                    self.assertEqual(len(results), int(bool(cases and cases[0].get("title"))))
                finally:
                    tools._search_index.cache_clear()


if __name__ == "__main__":
    unittest.main()
