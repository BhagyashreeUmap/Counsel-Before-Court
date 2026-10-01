"""Private-history tests never open the default production database."""
from copy import deepcopy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from backend import agents, database, llm, tools
from backend.pipeline import CasePipeline
from backend.private_history import PrivateHistory, HistoryError, DEFAULT_PATH


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private.db"
        self.store = PrivateHistory(self.path)
        self.before = {p: p.read_bytes() if p.exists() else None for p in
                       (database.CASES_PATH, database.USER_MEMORY_PATH, DEFAULT_PATH)}
        self.snapshot = CasePipeline().state()
        self.a = self.store.get_or_create_profile("Demo Citizen", "citizen@example.invalid")["id"]
        self.b = self.store.get_or_create_profile("Other Citizen", "other@example.invalid")["id"]
        for target, name in ((llm, "ask_llm_json"), (llm.genai, "Client"),
                             (tools, "save_case"), (database, "save_user_memory")):
            guard = patch.object(target, name, side_effect=AssertionError("Forbidden side effect"))
            mock = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(mock.assert_not_called)

    def tearDown(self):
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes() if path.exists() else None, original)

    def create(self, owner=None):
        return self.store.create_private_case(owner or self.a, self.snapshot, "Land dispute")

    def test_create_profile(self):
        self.assertTrue(self.a.startswith("P"))
        self.assertNotEqual(self.a, self.b)

    def test_normalized_email_same_profile(self):
        profile = self.store.get_or_create_profile("New name", " CITIZEN@EXAMPLE.INVALID ")
        self.assertEqual(profile["id"], self.a)
        self.assertEqual(profile["display_name"], "Demo Citizen")

    def test_phone_normalization(self):
        p = self.store.get_or_create_profile("Demo", "+91 (00000) 00000")
        q = self.store.get_or_create_profile("Demo", "0091-00000-00000")
        self.assertEqual(p["id"], q["id"])

    def test_phone_country_prefix_not_inferred(self):
        p = self.store.get_or_create_profile("Demo", "0000000000")
        q = self.store.get_or_create_profile("Demo", "+910000000000")
        self.assertNotEqual(p["id"], q["id"])

    def test_invalid_input(self):
        for name, contact in (("", "a@example.invalid"), ("x" * 101, "a@example.invalid"),
                              ("Demo", "bad"), ("Demo", "a@"), ("Demo", "123"),
                              ("Demo", "a@example.invalid and b@example.invalid"),
                              ("Demo", None), ("Demo\nName", "a@example.invalid")):
            with self.subTest(contact=contact), self.assertRaises(HistoryError):
                self.store.get_or_create_profile(name, contact)

    def test_raw_contacts_not_stored(self):
        self.store.get_or_create_profile("Phone Demo", "+91 (00000) 00000")
        with closing(sqlite3.connect(self.path)) as db:
            dump = "\n".join(db.iterdump())
        for value in ("citizen@example.invalid", "+91 (00000) 00000", "+910000000000"):
            self.assertNotIn(value, dump)

    def test_create_case(self):
        record = self.create()
        self.assertTrue(record["id"].startswith("H"))
        self.assertEqual(record["profile_id"], self.a)

    def test_list_only_owner_and_newest_first(self):
        first = self.create()
        second = self.create()
        self.create(self.b)
        rows = self.store.get_profile_cases(self.a)
        self.assertEqual([r["id"] for r in rows], [second["id"], first["id"]])
        self.assertNotIn("pipeline_snapshot", rows[0])

    def test_snapshot_round_trip(self):
        self.snapshot["conversation"] = [{"role": "user", "content": "काका शेतजमिनीवर दावा करतात"}]
        self.assertEqual(self.create()["pipeline_snapshot"], self.snapshot)

    def test_restore_existing_pipeline(self):
        record = self.create()
        restored = self.store.reopen_private_case(self.a, record["id"])
        self.assertEqual(restored.state(), self.snapshot)
        self.assertEqual(restored.resume(), self.snapshot)

    def test_loaded_listener_snapshot_resumes_through_existing_api(self):
        self.snapshot["conversation"] = [{"role": "user", "content": "My uncle claims the land."},
                                         {"role": "assistant", "content": "What do you want?"}]
        self.snapshot["next_question"] = "What do you want?"
        self.snapshot["question_count"] = 1
        record = self.create()
        restored = self.store.reopen_private_case(self.a, record["id"])
        with patch.object(agents, "story_listener", return_value={
                "updated_case_file": self.snapshot["case_file"], "ready": False,
                "danger_flag": False, "next_question": "What happened next?"}) as listener:
            restored.listener_message("I want to prepare.")
        self.assertEqual(listener.call_args.args[2], 1)
        self.assertEqual(listener.call_args.args[0][-1]["content"], "I want to prepare.")
        self.assertEqual(restored.state()["current_stage"], "LISTENING")

    def test_save_status_title_and_snapshot(self):
        record = self.create()
        changed = deepcopy(self.snapshot)
        changed["current_stage"] = "COMPLETE"
        changed["case_file"]["legal_area"] = "land_dispute"
        changed["guidance_result"] = {"situation_summary": "Saved information"}
        saved = self.store.save_private_case_snapshot(self.a, record["id"], changed, "Preparation complete")
        self.assertEqual(saved["status"], "COMPLETE")
        self.assertEqual(saved["legal_area"], "land_dispute")
        self.assertEqual(saved["pipeline_snapshot"], changed)
        self.assertEqual(saved["created_at"], record["created_at"])
        self.assertGreaterEqual(saved["updated_at"], record["updated_at"])

    def test_other_profile_cannot_read(self):
        record = self.create()
        self.assertIsNone(self.store.get_private_case(self.b, record["id"]))
        self.assertIsNone(self.store.reopen_private_case(self.b, record["id"]))

    def test_other_profile_cannot_update(self):
        record = self.create()
        with self.assertRaises(HistoryError):
            self.store.save_private_case_snapshot(self.b, record["id"], self.snapshot)
        self.assertEqual(self.store.get_private_case(self.a, record["id"]), record)

    def test_other_profile_cannot_delete(self):
        record = self.create()
        self.assertFalse(self.store.delete_private_case(self.b, record["id"]))
        self.assertIsNotNone(self.store.get_private_case(self.a, record["id"]))

    def test_delete_private_only(self):
        record = self.create()
        self.assertTrue(self.store.delete_private_case(self.a, record["id"]))
        self.assertIsNone(self.store.get_private_case(self.a, record["id"]))
        self.assertFalse(self.store.delete_private_case(self.a, record["id"]))

    def test_private_history_absent_from_retrieval(self):
        self.snapshot["case_file"]["property_or_matter_details"] = "quasar zeppelin privateonly"
        record = self.create()
        self.assertIsNone(tools.get_case(record["id"]))
        self.assertEqual(tools.search_cases("quasar zeppelin privateonly"), [])

    def test_completion_never_contributes_shared_memory(self):
        self.snapshot["current_stage"] = "COMPLETE"
        record = self.create()
        self.assertIsNone(tools.get_case(record["id"]))
        self.assertEqual(database.USER_MEMORY_PATH.read_bytes(), self.before[database.USER_MEMORY_PATH])

    def test_invalid_snapshot_fails_without_case(self):
        for snapshot in ({}, {**self.snapshot, "api_key": "secret"},
                         {**self.snapshot, "current_stage": "INVALID"},
                         {**self.snapshot, "error": float("nan")},
                         {**self.snapshot, "error": object()}):
            with self.assertRaises(HistoryError):
                self.store.create_private_case(self.a, snapshot, "Title")
        self.assertEqual(self.store.get_profile_cases(self.a), [])

    def test_foreign_key_enforced(self):
        with self.assertRaises(HistoryError):
            self.store.create_private_case("missing", self.snapshot, "Title")
        with self.store._connection() as db:
            self.assertEqual(db.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_parameterized_sql(self):
        record = self.store.create_private_case(self.a, self.snapshot, "x'); DROP TABLE profiles; --")
        self.assertEqual(record["title"], "x'); DROP TABLE profiles; --")
        self.assertEqual(self.store.get_or_create_profile("Demo", "citizen@example.invalid")["id"], self.a)


if __name__ == "__main__":
    unittest.main()
