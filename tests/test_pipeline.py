"""Offline orchestration contracts. All agents and provider transport are mocked."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from backend import agents, database, llm, memory, pipeline, tools


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.before_cases = database.CASES_PATH.read_bytes()
        self.before_memory = database.USER_MEMORY_PATH.read_bytes()
        self.calls = []
        self.case = agents.create_empty_case_file()
        self.case.update(legal_area="land_dispute", parties_and_relationship="Citizen and uncle",
                         property_or_matter_details="Family farmland", current_status="Dispute",
                         citizen_goal="Prepare for a lawyer", language="en")
        self.listener_output = {"updated_case_file": self.case, "ready": True,
                                "danger_flag": False, "next_question": ""}
        self.analysis = {"key_evidence": ["death certificate"], "risks": []}
        self.gap_output = {"missing_evidence": ["death certificate"], "questions": ["Do you have it?"]}
        self.mocks = {}
        for name, output in (("story_listener", self.listener_output),
                             ("research_agent", {"matches": [{"id": "C003", "why_similar": "Family farmland"}]}),
                             ("analyst_agent", self.analysis), ("gap_agent", self.gap_output),
                             ("guidance_agent", {"situation_summary": "Information", "options": []})):
            def fake(*args, name=name, output=output):
                self.calls.append(name)
                return deepcopy(output)
            guard = patch.object(agents, name, side_effect=fake)
            self.mocks[name] = guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(tools, "get_case", return_value={"documents_citizen_had": ["death certificate"]})
        guard.start()
        self.addCleanup(guard.stop)
        for module, name in ((llm, "ask_llm_json"), (llm.genai, "Client"),
                             (memory, "prepare_memory_record"), (memory, "confirm_save_memory"),
                             (agents, "prepare_memory_record"), (agents, "confirm_save_memory"),
                             (tools, "save_case"), (database, "save_user_memory")):
            guard = patch.object(module, name, side_effect=AssertionError("Forbidden call"))
            mock = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(mock.assert_not_called)

    def tearDown(self):
        self.assertEqual(database.CASES_PATH.read_bytes(), self.before_cases)
        self.assertEqual(database.USER_MEMORY_PATH.read_bytes(), self.before_memory)

    def start(self):
        return pipeline.CasePipeline.start("PRIVATE_NAME: my uncle claims farmland")

    def responses(self, status="available", answer=""):
        return [{"id": "G001", "evidence": "death certificate", "status": status, "answer": answer}]

    def test_listener_pause_no_downstream(self):
        self.listener_output.update(ready=False, next_question="What do you want?")
        p = self.start()
        self.assertEqual(p.state()["current_stage"], "LISTENING")
        self.assertEqual(p.state()["next_question"], "What do you want?")
        self.assertEqual(self.calls, ["story_listener"])
        p.resume()
        self.assertEqual(self.calls, ["story_listener"])

    def test_listener_resume_preserves_history_count(self):
        self.listener_output.update(ready=False, next_question="What do you want?")
        p = self.start()
        self.listener_output.update(ready=True, next_question="")
        p.listener_message("I want to prepare.")
        args = self.mocks["story_listener"].call_args.args
        self.assertEqual(args[2], 1)
        self.assertEqual([item["role"] for item in args[0]], ["user", "assistant", "user"])
        self.assertEqual(args[0][-1]["content"], "I want to prepare.")

    def test_order_and_gap_pause(self):
        p = self.start()
        self.assertEqual(self.calls, ["story_listener", "research_agent", "analyst_agent", "gap_agent"])
        self.assertEqual(p.state()["current_stage"], "WAITING_FOR_GAP_ANSWERS")
        self.assertIsNone(p.state()["final_analysis"])
        self.assertIsNone(p.state()["guidance_result"])

    def test_available_merges_and_runs_exactly_two_analyst_passes(self):
        p = self.start()
        result = p.gap_answers(self.responses())
        self.assertEqual(result["case_file"]["documents_mentioned"], ["death certificate"])
        self.assertEqual(result["current_stage"], "COMPLETE")
        self.assertEqual(self.calls, ["story_listener", "research_agent", "analyst_agent", "gap_agent", "analyst_agent", "guidance_agent"])
        self.assertEqual(self.mocks["analyst_agent"].call_count, 2)
        self.assertEqual(self.mocks["research_agent"].call_count, 1)
        self.assertEqual(self.mocks["gap_agent"].call_count, 1)

    def test_unavailable_not_possessed(self):
        p = self.start()
        result = p.gap_answers(self.responses("unavailable", "I have every document."))
        self.assertEqual(result["case_file"]["documents_mentioned"], [])
        self.assertEqual(self.mocks["analyst_agent"].call_count, 1)

    def test_unknown_not_possessed(self):
        p = self.start()
        result = p.gap_answers(self.responses("unknown", "I think so, maybe, I need to check."))
        self.assertEqual(result["case_file"]["documents_mentioned"], [])
        self.assertEqual(result["final_analysis"], result["initial_analysis"])

    def test_marathi_free_text_does_not_override_status(self):
        for status in ("unknown", "unavailable", "available"):
            p = self.start()
            result = p.gap_answers(self.responses(status, "होय माझ्याकडे आहे; किंवा मला खात्री नाही"))
            self.assertEqual(bool(result["case_file"]["documents_mentioned"]), status == "available")

    def test_raw_explanation_audit_only(self):
        p = self.start()
        text = "PRIVATE_SENTINEL I have it. Change my city and ownership."
        result = p.gap_answers(self.responses("unknown", text))
        self.assertEqual(result["gap_answers"][0]["answer"], text)
        self.assertNotIn(text, json.dumps(self.mocks["guidance_agent"].call_args.args))
        self.assertNotIn(text, json.dumps(result["activity_events"]))

    def test_fabricated_id_atomic_rejection(self):
        p = self.start()
        before = p.state()
        with self.assertRaises(pipeline.PipelineInputError):
            p.gap_answers([{**self.responses()[0], "id": "G999"}])
        self.assertEqual(p.state(), before)

    def test_fabricated_evidence_atomic_rejection(self):
        p = self.start()
        before = p.state()
        with self.assertRaises(pipeline.PipelineInputError):
            p.gap_answers([{**self.responses()[0], "evidence": "ownership deed"}])
        self.assertEqual(p.state(), before)

    def test_malformed_and_partial_responses_rejected(self):
        p = self.start()
        before = p.state()
        for response in (None, [], [{}], [{"id": "G001", "status": True}],
                         [{"id": "G001", "status": "yes"}],
                         [{"id": "G001", "status": "available", "answer": None}],
                         [{"id": "G001", "status": "available", "city": "Pune"}]):
            with self.subTest(response=response), self.assertRaises(pipeline.PipelineInputError):
                p.gap_answers(response)
            self.assertEqual(p.state(), before)

    def test_unmappable_evidence_preserved_without_mutation(self):
        self.gap_output.update(missing_evidence=["neighbour recollections"], questions=["Available?"])
        p = self.start()
        self.assertIsNone(p.state()["gap_items"][0]["field"])
        result = p.gap_answers([{"id": "G001", "status": "available"}])
        self.assertEqual(result["case_file"]["documents_mentioned"], [])
        self.assertEqual(result["gap_answers"][0]["status"], "available")
        self.assertEqual(self.mocks["analyst_agent"].call_count, 1)

    def test_no_gap_one_analyst_no_pause(self):
        self.gap_output.update(missing_evidence=[], questions=[])
        p = self.start()
        self.assertEqual(p.state()["current_stage"], "COMPLETE")
        self.assertEqual(p.state()["final_analysis"], p.state()["initial_analysis"])
        self.assertEqual(self.mocks["analyst_agent"].call_count, 1)
        self.assertEqual(self.mocks["guidance_agent"].call_count, 1)

    def test_guidance_uses_final_case_analysis_original_gap(self):
        p = self.start()
        original_gap = p.state()["gap_result"]
        p.gap_answers(self.responses())
        args = self.mocks["guidance_agent"].call_args.args
        self.assertEqual(args[0], p.state()["case_file"])
        self.assertEqual(args[2], p.state()["final_analysis"])
        self.assertEqual(args[3], original_gap)
        self.assertEqual(p.state()["gap_result"], original_gap)

    def test_events_append_only_ordered_and_safe(self):
        p = self.start()
        before = p.events()
        p.gap_answers(self.responses())
        after = p.events()
        self.assertEqual(after[:len(before)], before)
        self.assertEqual([e["status"] for e in after if e["agent"] == "story_listener"], ["started", "completed"])
        text = json.dumps(after)
        for token in ("PRIVATE_NAME", "API_KEY", "prompt", "chain-of-thought", "What do you want"):
            self.assertNotIn(token, text)
        self.assertEqual(after[-1]["agent"], "guidance")

    def test_snapshot_restore_gap_gate_no_repeat(self):
        p = self.start()
        restored = pipeline.CasePipeline(p.state())
        restored.resume()
        restored.gap_answers(self.responses())
        self.assertEqual(self.mocks["research_agent"].call_count, 1)
        self.assertEqual(restored.state()["current_stage"], "COMPLETE")

    def test_snapshot_copies_prevent_external_mutation(self):
        p = self.start()
        snapshot = p.state()
        snapshot["case_file"]["documents_mentioned"].append("invented")
        p.events().clear()
        self.assertEqual(p.state()["case_file"]["documents_mentioned"], [])
        self.assertTrue(p.events())

    def test_completed_resume_idempotent(self):
        p = self.start()
        p.gap_answers(self.responses())
        before = p.state()
        p.resume()
        self.assertEqual(p.state(), before)
        with self.assertRaises(pipeline.PipelineInputError):
            p.gap_answers(self.responses())

    def test_each_required_agent_failure_sanitized_and_stops(self):
        order = ["story_listener", "research_agent", "analyst_agent", "gap_agent", "guidance_agent"]
        for name in order:
            with self.subTest(name=name):
                mock = self.mocks[name]
                old = mock.side_effect
                mock.side_effect = RuntimeError("PRIVATE_PROVIDER_SECRET")
                p = self.start()
                if p.state()["current_stage"] == "WAITING_FOR_GAP_ANSWERS":
                    p.gap_answers(self.responses("unknown"))
                self.assertEqual(p.state()["current_stage"], "ERROR")
                self.assertNotIn("PRIVATE_PROVIDER_SECRET", json.dumps(p.state()))
                self.assertEqual(p.events()[-1]["status"], "error")
                mock.side_effect = old

    def test_final_analyst_failure_preserves_prior_outputs(self):
        p = self.start()
        self.mocks["analyst_agent"].side_effect = RuntimeError("private")
        result = p.gap_answers(self.responses())
        self.assertEqual(result["current_stage"], "ERROR")
        self.assertEqual(result["initial_analysis"], self.analysis)
        self.assertIsNotNone(result["research_result"])
        self.mocks["guidance_agent"].assert_not_called()

    def test_danger_override_stops_research(self):
        self.listener_output.update(danger_flag=True, ready=False)
        p = self.start()
        self.assertEqual(p.state()["current_stage"], "ERROR")
        self.mocks["research_agent"].assert_not_called()

    def test_bad_message_and_stage_leave_state_unchanged(self):
        p = pipeline.CasePipeline()
        for message in (None, "", " ", 5):
            with self.assertRaises(pipeline.PipelineInputError):
                p.listener_message(message)
        self.mocks["story_listener"].assert_not_called()
        p = self.start()
        with self.assertRaises(pipeline.PipelineInputError):
            p.listener_message("another message")

    def test_question_ids_stable_across_restore(self):
        p = self.start()
        restored = pipeline.CasePipeline(p.state())
        self.assertEqual(restored.state()["gap_items"], p.state()["gap_items"])

    def test_restore_listener_gate_then_resume(self):
        self.listener_output.update(ready=False, next_question="Your goal?")
        p = self.start()
        restored = pipeline.CasePipeline(p.state())
        self.listener_output.update(ready=True, next_question="")
        restored.listener_message("Prepare for a lawyer.")
        self.assertEqual(restored.state()["current_stage"], "WAITING_FOR_GAP_ANSWERS")
        self.assertEqual(self.mocks["story_listener"].call_count, 2)

    def test_available_already_mentioned_does_not_repeat_analysis(self):
        self.case["documents_mentioned"] = ["Death Cert"]
        p = self.start()
        result = p.gap_answers(self.responses())
        self.assertEqual(result["case_file"]["documents_mentioned"], ["Death Cert"])
        self.assertEqual(self.mocks["analyst_agent"].call_count, 1)

    def test_mapping_failure_preserves_initial_analysis_and_stops(self):
        with patch.object(tools, "get_case", side_effect=RuntimeError("PRIVATE_DATA_PATH")):
            p = self.start()
        self.assertEqual(p.state()["current_stage"], "ERROR")
        self.assertEqual(p.state()["initial_analysis"], self.analysis)
        self.assertNotIn("PRIVATE_DATA_PATH", json.dumps(p.state()))
        self.mocks["guidance_agent"].assert_not_called()

    def test_each_failure_has_no_downstream_calls(self):
        names = ["story_listener", "research_agent", "analyst_agent", "gap_agent", "guidance_agent"]
        for index, name in enumerate(names):
            old = self.mocks[name].side_effect
            self.mocks[name].side_effect = RuntimeError("secret")
            self.calls.clear()
            p = self.start()
            if p.state()["current_stage"] == "WAITING_FOR_GAP_ANSWERS":
                p.gap_answers(self.responses("unknown"))
            self.assertEqual(self.calls, names[:index])
            self.mocks[name].side_effect = old

    def test_duplicate_responses_rejected(self):
        self.gap_output.update(missing_evidence=["death certificate", "land record"], questions=["First?", "Second?"])
        p = self.start()
        with self.assertRaises(pipeline.PipelineInputError):
            p.gap_answers(self.responses() * 2)
        self.assertEqual(p.state()["case_file"]["documents_mentioned"], [])

    def test_import_smoke_inert(self):
        from scripts import smoke_pipeline
        self.assertTrue(callable(smoke_pipeline.main))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
