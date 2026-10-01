"""Offline Streamlit navigation using temporary history and mocked agents."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from backend import agents, database, llm, memory
from backend.pipeline import CasePipeline
from backend.private_history import PrivateHistory, DEFAULT_PATH
from ui_helpers import workflow_view, gap_responses, progress, current_activity, listener_recovery


def guidance():
    return {"situation_summary": "Family farmland dispute", "outcome_and_time": "Observed sample durations only.",
            "red_flags": ["Records not yet mentioned"], "similar_cases": ["A sample family dispute"],
            "document_checklist": [{"item": "death certificate", "status": "check_if_available"}],
            "past_cases_used": [{"id": "C003", "source": "synthetic", "why_similar": "An uncle claimed farmland"}],
            "options": [{"option": "Discuss preparation", "possible_benefit": "Clarity", "possible_tradeoff": "Time"}],
            "questions_for_lawyer": ["What records should I check?"],
            "referrals": {"legal_aid": [], "lawyers": []}, "lawyer_ready_summary": "A family farmland dispute.",
            "disclaimer": "General legal information, not legal advice."}


class UITests(unittest.TestCase):
    def setUp(self):
        self.before = {p: p.read_bytes() if p.exists() else None for p in (database.CASES_PATH, database.USER_MEMORY_PATH, DEFAULT_PATH)}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = PrivateHistory(Path(self.temp.name) / "history.db")
        p = patch("backend.private_history.PrivateHistory", return_value=self.store)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(llm, "ask_llm_json", side_effect=AssertionError("No Gemini"))
        self.ask = p.start()
        self.addCleanup(p.stop)
        p = patch.object(llm.genai, "Client", side_effect=AssertionError("No network"))
        p.start()
        self.addCleanup(p.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=15)

    def tearDown(self):
        self.ask.assert_not_called()
        for path, before in self.before.items():
            self.assertEqual(path.read_bytes() if path.exists() else None, before)

    def run_app(self):
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def profile(self):
        profile = self.store.get_or_create_profile("Demo", "ui@example.invalid")
        self.app.session_state["profile"] = profile
        self.app.session_state["view"] = "cases"
        return profile

    def case_view(self, stage):
        profile = self.profile()
        snapshot = CasePipeline().state()
        snapshot["current_stage"] = stage
        snapshot["case_file"]["legal_area"] = "land_dispute"
        snapshot["case_file"]["parties_and_relationship"] = "Citizen and uncle"
        snapshot["guidance_result"] = guidance() if stage == "COMPLETE" else None
        if stage == "WAITING_FOR_GAP_ANSWERS":
            snapshot["gap_items"] = [{"id": "G001", "evidence": "death certificate", "question": "Do you have it?", "field": "documents_mentioned"}]
        record = self.store.create_private_case(profile["id"], snapshot, "Demo case")
        self.app.session_state["case_id"] = record["id"]
        self.app.session_state["pipeline"] = CasePipeline(snapshot)
        self.app.session_state["view"] = workflow_view(snapshot)
        return record

    def test_welcome_acknowledgement_gate(self):
        self.run_app()
        self.assertTrue(self.app.button[0].disabled)
        self.app.checkbox[0].check()
        self.run_app()
        self.assertFalse(self.app.button[0].disabled)

    def test_profile_access_and_single_new_case_rerun(self):
        self.run_app()
        self.app.checkbox[0].check()
        self.run_app()
        self.app.text_input[0].set_value("Demo")
        self.app.text_input[1].set_value("ui@example.invalid")
        self.app.button[0].click()
        self.run_app()
        self.app.button[1].click()
        self.run_app()
        owner = self.app.session_state["profile"]["id"]
        self.assertEqual(len(self.store.get_profile_cases(owner)), 1)
        self.run_app()
        self.assertEqual(len(self.store.get_profile_cases(owner)), 1)

    def test_language_switch_preserves_pipeline_and_case(self):
        record = self.case_view("COMPLETE")
        self.run_app()
        before = self.app.session_state["pipeline"].state()
        self.app.segmented_control[0].set_value("मराठी")
        self.run_app()
        self.assertEqual(self.app.session_state["pipeline"].state(), before)
        self.assertEqual(self.app.session_state["case_id"], record["id"])
        self.assertEqual(len(self.store.get_profile_cases(record["profile_id"])), 1)

    def test_reopen_completed_case_no_agent_calls(self):
        record = self.case_view("COMPLETE")
        self.app.session_state["view"] = "cases"
        self.run_app()
        next(button for button in self.app.button if button.label == "Open / Resume").click()
        self.run_app()
        self.assertEqual(self.app.session_state["case_id"], record["id"])
        self.assertEqual(len(self.app.tabs), 6)

    def test_evidence_default_unknown_submission_contract(self):
        self.case_view("WAITING_FOR_GAP_ANSWERS")
        self.run_app()
        p = self.app.session_state["pipeline"]
        with patch.object(p, "gap_answers") as submit:
            next(button for button in self.app.button if button.label == "Submit answers").click()
            self.run_app()
            submit.assert_called_once_with([{"id": "G001", "evidence": "death certificate", "status": "unknown", "answer": ""}])

    def test_listener_submission_once_and_persisted(self):
        record = self.case_view("LISTENING")
        self.run_app()
        case = self.app.session_state["pipeline"].state()["case_file"]
        with patch.object(agents, "story_listener", return_value={"updated_case_file": case,
                "ready": False, "danger_flag": False, "next_question": "What do you want?"}) as listener:
            self.app.chat_input[0].set_value("My uncle claims farmland.")
            self.run_app()
            self.run_app()
            listener.assert_called_once()
        saved = self.store.get_private_case(record["profile_id"], record["id"])
        self.assertEqual(saved["pipeline_snapshot"]["question_count"], 1)

    def test_private_delete_requires_confirmation(self):
        record = self.case_view("COMPLETE")
        self.app.session_state["view"] = "cases"
        self.run_app()
        delete = next(button for button in self.app.button if button.label == "Delete saved private case")
        self.assertTrue(delete.disabled)
        self.app.checkbox[0].check()
        self.run_app()
        next(button for button in self.app.button if button.label == "Delete saved private case").click()
        self.run_app()
        self.assertIsNone(self.store.get_private_case(record["profile_id"], record["id"]))

    def test_danger_card_is_conditional(self):
        self.case_view("LISTENING")
        self.run_app()
        self.assertFalse(any("Need immediate help" in m.value for m in self.app.markdown))
        snapshot = self.app.session_state["pipeline"].state()
        snapshot["case_file"]["danger_flag"] = True
        snapshot["current_stage"] = "ERROR"
        self.app.session_state["pipeline"] = CasePipeline(snapshot)
        self.run_app()
        self.assertTrue(any("Need immediate help" in m.value for m in self.app.markdown))
        self.assertEqual(len(self.app.chat_input), 0)

    def test_completion_not_yet_and_keep_private_no_memory(self):
        self.case_view("COMPLETE")
        self.app.session_state["view"] = "completion"
        with patch.object(memory, "prepare_memory_record") as prepare, patch.object(memory, "confirm_save_memory") as save:
            self.run_app()
            self.app.segmented_control[1].set_value("resolved")
            self.run_app()
            self.assertTrue(any("private history" in m.value for m in self.app.caption))
            prepare.assert_not_called()
            save.assert_not_called()

    def test_exact_preview_two_gates_and_no_duplicate_save(self):
        self.case_view("COMPLETE")
        self.app.session_state["view"] = "completion"
        self.app.session_state["learn_resolved"] = "resolved"
        self.app.session_state["learn_sharing"] = "share"
        preview = {"id": "U000001", "lesson": "Keep record copies", "source": "user_contributed_anonymized"}
        with patch.object(memory, "prepare_memory_record", return_value=preview) as prepare, \
                patch.object(memory, "confirm_save_memory", return_value=preview) as save:
            self.run_app()
            self.app.text_area[2].set_value("Keep record copies")
            next(button for button in self.app.button if button.label == "Preview Anonymized Version").click()
            self.run_app()
            self.assertEqual(self.app.session_state["preview"], preview)
            prepare.assert_called_once()
            self.assertIs(prepare.call_args.kwargs["consent"], True)
            save.assert_not_called()
            self.run_app()
            prepare.assert_called_once()
            confirm = next(button for button in self.app.button if button.label == "Confirm & Share Anonymously")
            self.assertTrue(confirm.disabled)
            self.app.checkbox[0].check()
            self.run_app()
            next(button for button in self.app.button if button.label == "Confirm & Share Anonymously").click()
            self.run_app()
            save.assert_called_once_with(preview, confirmed=True)
            self.run_app()
            save.assert_called_once()

    def test_progress_has_no_fabricated_completed_final_pass(self):
        snapshot = CasePipeline().state()
        snapshot["current_stage"] = "COMPLETE"
        self.assertEqual(dict(progress(snapshot))["analyst_final"], "not_required")

    def test_status_mapping_preserves_canonical_values(self):
        items = [{"id": "G001", "evidence": "document"}]
        for status in ("available", "unavailable", "unknown"):
            response = gap_responses(items, [status], ["होय आहे पण खात्री नाही"])[0]
            self.assertEqual(response["status"], status)

    def test_native_evidence_selection_preserves_status_and_marathi_audit(self):
        self.case_view("WAITING_FOR_GAP_ANSWERS")
        self.run_app()
        pipeline = self.app.session_state["pipeline"]
        with patch.object(pipeline, "gap_answers") as submit:
            for status in ("available", "unavailable", "unknown"):
                with self.subTest(status=status):
                    self.app.segmented_control[1].set_value(status)
                    self.app.text_input[0].set_value("होय आहे पण खात्री नाही")
                    next(b for b in self.app.button if b.label == "Submit answers").click()
                    self.run_app()
                    submit.assert_called_once_with([{"id": "G001", "evidence": "death certificate", "status": status, "answer": "होय आहे पण खात्री नाही"}])
                    submit.reset_mock()

    def test_native_selection_defaults_and_existing_preselection(self):
        self.run_app()
        self.assertEqual(self.app.segmented_control[0].value, "English")
        self.case_view("WAITING_FOR_GAP_ANSWERS")
        self.run_app()
        self.assertEqual(self.app.segmented_control[1].value, "unknown")
        self.case_view("COMPLETE")
        self.app.session_state["view"] = "completion"
        self.run_app()
        self.assertEqual(self.app.segmented_control[1].value, "not_yet")
        self.app.segmented_control[1].set_value("resolved")
        self.run_app()
        self.assertEqual(self.app.segmented_control[2].value, "private")

    def test_listener_invalid_response_manual_recovery_and_valid_resubmit(self):
        record = self.case_view("LISTENING")
        self.run_app()
        # Real Listener validation, mocked transport: Gemini supplies no question
        # despite insufficient critical facts. This is the reproduced failure.
        with patch.object(llm, "ask_llm_json", return_value={"case_file_updates": {},
                "ready": True, "danger_flag": False, "next_question": ""}) as transport:
            self.app.chat_input[0].set_value("My uncle claims our farmland.").run()
            self.assertEqual(list(self.app.exception), [])
            self.assertEqual(self.app.session_state["pipeline"].state()["current_stage"], "ERROR")
            self.assertIn("No new facts were accepted", self.app.error[0].value)
            transport.assert_called_once()
        next(b for b in self.app.button if b.label == "Return to previous intake state").click()
        self.run_app()
        self.assertEqual(self.app.session_state["pipeline"].state()["current_stage"], "LISTENING")
        with patch.object(agents, "story_listener", return_value={"updated_case_file": self.app.session_state["pipeline"].state()["case_file"],
                "ready": False, "danger_flag": False, "next_question": "What do you want to prepare?"}) as listener:
            self.app.chat_input[0].set_value("I want to prepare before speaking to a lawyer.")
            self.run_app()
            listener.assert_called_once()
        saved = self.store.get_private_case(record["profile_id"], record["id"])
        self.assertEqual(saved["pipeline_snapshot"]["current_stage"], "LISTENING")

    def test_current_activity_only_waiting_after_finished_turn(self):
        snapshot = CasePipeline().state()
        snapshot["activity_events"] = [{"agent": "story_listener", "status": status, "message": "Safe event"}
                                       for status in ("started", "completed", "waiting")]
        self.assertEqual([e["status"] for e in current_activity(snapshot)], ["waiting"])
        self.assertEqual(len(snapshot["activity_events"]), 3)

    def test_recovery_never_bypasses_danger(self):
        snapshot = CasePipeline().state()
        snapshot.update(current_stage="ERROR", conversation=[{"role": "user", "content": "Danger"}])
        snapshot["case_file"]["danger_flag"] = True
        snapshot["activity_events"] = [{"agent": "story_listener", "status": status} for status in ("started", "error")]
        self.assertIsNone(listener_recovery(snapshot))


if __name__ == "__main__":
    unittest.main()
