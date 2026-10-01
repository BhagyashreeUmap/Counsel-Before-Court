"""Manual real bilingual pipeline demo: python -m scripts.smoke_pipeline.

Never imported with side effects. Each semantic call has one provider attempt.
Simulated citizen responses are fixed fixture facts, never inferred from a
generated question. Persistent Memory is forbidden and storage is isolated.
"""
from functools import partial
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from backend import agents, database, llm, memory, tools
from backend.pipeline import CasePipeline
from scripts.smoke_guidance_agent import _check_marathi_prose


def _scenario(language):
    marathi = language == "mr"
    print("=== Marathi Pipeline Smoke ===" if marathi else "=== English Pipeline Smoke ===")
    first = ("माझ्या वडिलांचे गेल्या वर्षी निधन झाले. आता काका म्हणतात की आमची शेतजमीन त्यांची आहे. "
             "हा जमिनीबाबतचा वाद सध्या सुरू आहे. मला परिस्थिती समजून घ्यायची आहे आणि वकिलांशी बोलण्यापूर्वी तयारी करायची आहे."
             if marathi else
             "My father died last year and now my uncle says our farmland belongs to him. "
             "This is a current land dispute. I want to understand the situation and prepare before speaking to a lawyer.")
    reply = ("आमच्या कुटुंबाच्या शेतजमिनीवर काका दावा करत आहेत. वडिलांचे गेल्या वर्षी निधन झाले. "
             "वाद सध्या सुरू आहे. माझ्याकडे मृत्यू प्रमाणपत्र आहे. इतर कागदपत्रांची मला खात्री नाही. "
             "मला वकिलांशी बोलण्यापूर्वी परिस्थिती समजून तयारी करायची आहे. मराठीत बोला."
             if marathi else
             "My uncle claims our family's farmland after my father's death last year. The dispute is current. "
             "I have the death certificate. I am unsure about other documents. "
             "I want to understand the situation and prepare before speaking to a lawyer. Please use English.")
    p = CasePipeline.start(first)
    shown = 0

    def display_events():
        nonlocal shown
        events = p.events()
        for event in events[shown:]:
            print(f"[{event['agent']}] {event['status']}: {event['message']}")
            if "details" in event:
                print(json.dumps(event["details"], ensure_ascii=False))
        shown = len(events)

    display_events()
    for _ in range(agents.MAX_QUESTIONS + 1):
        if p.state()["current_stage"] != "LISTENING":
            break
        print("[Story Listener question]", p.state()["next_question"])
        p.listener_message(reply)
        display_events()
    if p.state()["current_stage"] == "WAITING_FOR_GAP_ANSWERS":
        responses = []
        for item in p.state()["gap_items"]:
            # Fixture confirms only this document; no generated evidence is
            # automatically treated as possessed.
            available = agents._gap_canonical(item["evidence"]) == "death certificate"
            responses.append({"id": item["id"], "evidence": item["evidence"],
                              "status": "available" if available else "unknown",
                              "answer": ("माझ्याकडे मृत्यू प्रमाणपत्र आहे." if available else "मला खात्री नाही.")
                              if marathi else ("I have the death certificate." if available else "I am not sure.")})
        print("[Simulated Citizen Gap Answers]", json.dumps(responses, ensure_ascii=False))
        p.gap_answers(responses)
        display_events()
    state = p.state()
    print("FINAL STAGE:", state["current_stage"])
    if state["current_stage"] != "COMPLETE":
        raise AssertionError("Pipeline did not complete; inspect safe stage events.")
    guidance = state["guidance_result"]
    assert guidance and state["research_result"]["matches"] and state["activity_events"]
    assert sum(e["agent"] == "research" and e["status"] == "started" for e in p.events()) == 1
    assert not any("memory" in e["agent"] for e in p.events())
    print(json.dumps(guidance, ensure_ascii=False, indent=2))
    assert guidance["situation_summary"].strip() and guidance["lawyer_ready_summary"].strip()
    assert guidance["options"] and guidance["questions_for_lawyer"]
    if marathi:
        for field in ("situation_summary", "lawyer_ready_summary", "disclaimer"):
            _check_marathi_prose(guidance[field], field)
        for field in ("similar_cases", "red_flags", "questions_for_lawyer"):
            for text in guidance[field]:
                _check_marathi_prose(text, field)
        for option in guidance["options"]:
            for field in ("option", "possible_benefit", "possible_tradeoff"):
                _check_marathi_prose(option[field], field)
        for citation in guidance["past_cases_used"]:
            _check_marathi_prose(citation["why_similar"], "case explanation")
    print("Review printed prose manually; canonical evidence and recorded time/cost may remain English.")


def main():
    production_path = database.USER_MEMORY_PATH
    before = production_path.read_bytes() if production_path.exists() else None
    cases_before = database.CASES_PATH.read_bytes()
    guards = []
    try:
        for module, name in ((memory, "prepare_memory_record"), (memory, "confirm_save_memory"),
                             (agents, "prepare_memory_record"), (agents, "confirm_save_memory"),
                             (tools, "save_case"), (database, "save_user_memory")):
            guard = patch.object(module, name, side_effect=AssertionError("Pipeline must never invoke Memory."))
            guard.start()
            guards.append(guard)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(database, "USER_MEMORY_PATH", Path(directory) / "memory.json"), \
                patch.object(llm, "ask_llm_json", partial(llm._ask_llm_json, max_attempts=1)):
            tools._cases.cache_clear()
            tools._search_index.cache_clear()
            _scenario("en")
            _scenario("mr")
            assert not database.USER_MEMORY_PATH.exists()
        return 0
    finally:
        for guard in reversed(guards):
            guard.stop()
        tools._cases.cache_clear()
        tools._search_index.cache_clear()
        assert (production_path.read_bytes() if production_path.exists() else None) == before
        assert database.CASES_PATH.read_bytes() == cases_before


if __name__ == "__main__":
    raise SystemExit(main())
