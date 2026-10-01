"""Manual isolated Guidance demo: python -m scripts.smoke_guidance_agent.

One provider attempt per scenario, no other agent calls. City is fixed smoke data.
"""
from functools import partial
from copy import deepcopy
import json
import re
from unittest.mock import patch

from backend import agents, llm


def _check_marathi_prose(text, field):
    # Script-based sanity check, not a Marathi/Hindi language classifier.
    # Human review of the printed prose remains part of the manual test.
    if not isinstance(text, str) or not text.strip():
        raise AssertionError(f"Marathi {field} must contain nonempty prose.")
    letters = [character for character in text if character.isalpha()]
    devanagari = sum("\u0900" <= character <= "\u097f" for character in letters)
    if not letters or devanagari / len(letters) < 0.5:
        raise AssertionError(f"Marathi {field} did not contain predominantly Devanagari prose.")


def _marathi_scenario(case, research, analysis, gap):
    marathi_case = deepcopy(case)
    marathi_case.update(language="mr", city="Pune",
        parties_and_relationship="नागरिक, दिवंगत वडील आणि काका",
        property_or_matter_details="कुटुंबाची शेतजमीन", timeline="वडिलांचे गेल्या वर्षी निधन झाले",
        other_side_claim="काका म्हणतात की शेतजमीन त्यांची आहे",
        current_status="कुटुंबाच्या शेतजमिनीबाबत सध्या वाद सुरू आहे",
        citizen_goal="परिस्थिती समजून घ्यायची आहे आणि वकिलांशी बोलण्यापूर्वी तयारी करायची आहे")
    marathi_gap = deepcopy(gap)
    marathi_gap["questions"] = ["तुमच्याकडे मृत्यू प्रमाणपत्र आहे का?", "तुमच्याकडे वारस प्रमाणपत्र आहे का?"]
    marathi_answers = [{"question": marathi_gap["questions"][0], "answer": "हो, माझ्याकडे मृत्यू प्रमाणपत्र आहे."},
        {"question": marathi_gap["questions"][1], "answer": "आमच्याकडे वारस प्रमाणपत्र आहे की नाही याची मला खात्री नाही."}]
    print("=== Marathi Guidance Smoke Test ===")
    with patch.object(llm, "ask_llm_json", partial(llm._ask_llm_json, max_attempts=1)):
        result = agents.guidance_agent(marathi_case, deepcopy(research), deepcopy(analysis), marathi_gap, marathi_answers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for field in ("situation_summary", "lawyer_ready_summary"):
        _check_marathi_prose(result[field], field)
    for field in ("similar_cases", "red_flags", "questions_for_lawyer"):
        if not result[field]:
            raise AssertionError(f"Marathi {field} was empty in this populated smoke scenario.")
        for text in result[field]:
            _check_marathi_prose(text, field)
    if not result["options"]:
        raise AssertionError("Marathi options were empty.")
    for option in result["options"]:
        for field in ("option", "possible_benefit", "possible_tradeoff"):
            _check_marathi_prose(option[field], field)
    if [(item["id"], item["source"]) for item in result["past_cases_used"]] != [(item["id"], "synthetic") for item in research["matches"]]:
        raise AssertionError("Marathi output changed grounded case identities or sources.")
    for item in result["past_cases_used"]:
        _check_marathi_prose(item["why_similar"], "past_cases_used explanation")
    expected_referrals = {"legal_aid": agents.tools.find_legal_aid("Pune"),
                          "lawyers": agents.tools.find_lawyers("property/land", "Pune", "mr")}
    if result["referrals"] != expected_referrals or not all(expected_referrals.values()):
        raise AssertionError("Marathi referral records do not match authoritative local tools.")
    if result["document_checklist"] != agents._guidance_checklist(marathi_case, analysis):
        raise AssertionError("Marathi checklist changed grounded evidence identities/statuses.")
    if result["disclaimer"] != agents.GUIDANCE_DISCLAIMERS["mr"]:
        raise AssertionError("Marathi output changed the guaranteed Marathi disclaimer.")
    _check_marathi_prose(result["disclaimer"], "disclaimer")
    if analysis["typical_time"] not in result["outcome_and_time"] or analysis["typical_cost_level"] not in result["outcome_and_time"]:
        raise AssertionError("Marathi presentation changed authoritative time/cost strings.")
    _check_marathi_prose(result["outcome_and_time"].split(analysis["typical_time"])[0], "time/cost wrapper")
    agents._guidance_check_text(json.dumps(result, ensure_ascii=False), {"C003", "C001", "C002"})
    prose = json.dumps({field: result[field] for field in agents.GUIDANCE_SCHEMA["properties"] if field != "localized_cases"}, ensure_ascii=False)
    if re.search(r"अधिनियम|कलम\s*\d|अनुच्छेद\s*\d|फॉर्म\s*[A-Z0-9]|\d+\s*(?:दिवसांत|दिवसांच्या आत|टक्के|रुपये)|(?:तुम्ही|तुमचा खटला)\s+(?:नक्की\s+)?(?:जिंकाल|हराल)|लगेच\s+(?:दावा|खटला)\s+दाखल", prose):
        raise AssertionError("Marathi prose contains an obvious unsupported law, procedure or prediction.")
    print("Marathi smoke checks passed. Original Analyst time/cost strings and canonical evidence labels remain English; review Marathi prose manually.")


def main():
    case = agents.create_empty_case_file()
    case.update(legal_area="land_dispute", parties_and_relationship="Citizen, deceased father and uncle",
                property_or_matter_details="Family farmland", timeline="Father died last year",
                other_side_claim="Uncle says the land belongs to him", current_status="Current family land dispute",
                citizen_goal="Understand the situation and prepare before speaking to a lawyer",
                documents_mentioned=["death certificate"], language="en")
    case["city"] = "Pune"  # Fixed direct smoke input, never inferred citizen data.
    research = {"matches": [
        {"id": "C003", "why_similar": "An uncle claimed family land after the father's death."},
        {"id": "C001", "why_similar": "A relative claimed farmland used by a family."},
        {"id": "C002", "why_similar": "A land-use dispute involved uncertainty about supporting records."}]}
    analysis = {"winning_patterns": ["C003 included family papers; C001 included possession records."],
                "losing_patterns": ["C002 lacked documentary proof and recorded delay."],
                "key_evidence": ["death certificate", "legal heir certificate", "family tree", "land record extract"],
                "risks": ["The citizen has not yet mentioned heir or land records."],
                "typical_time": "Across the 3 retrieved synthetic cases with usable time data, recorded durations ranged from 8 to 40 months.",
                "typical_cost_level": "The 3 retrieved synthetic cases with usable cost data recorded these cost levels: low, medium."}
    gap = {"missing_evidence": ["death certificate", "legal heir certificate"],
           "questions": ["Do you have the death certificate?", "Do you have a legal heir certificate?"]}
    answers = [{"question": gap["questions"][0], "answer": "Yes, I have the death certificate."},
               {"question": gap["questions"][1], "answer": "I am not sure whether we have one."}]
    print("=== Guidance Agent Smoke Test ===")
    try:
        # Same structured transport, with retries disabled for this manual demo.
        with patch.object(llm, "ask_llm_json", partial(llm._ask_llm_json, max_attempts=1)):
            result = agents.guidance_agent(case, research, analysis, gap, answers)
    except agents.GuidanceAgentError as exc:
        print(f"Guidance smoke test failed: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    expected = {"situation_summary", "similar_cases", "outcome_and_time", "options", "document_checklist",
                "red_flags", "lawyer_ready_summary", "questions_for_lawyer", "referrals", "past_cases_used", "disclaimer"}
    if set(result) != expected:
        raise AssertionError("Guidance did not return the exact public keys.")
    for field in ("similar_cases", "options", "document_checklist", "red_flags", "questions_for_lawyer", "past_cases_used"):
        if not isinstance(result[field], list):
            raise AssertionError("Guidance list field has an invalid type.")
    for field in ("situation_summary", "outcome_and_time", "lawyer_ready_summary", "disclaimer"):
        if not isinstance(result[field], str):
            raise AssertionError("Guidance string field has an invalid type.")
    expected_referrals = {"legal_aid": agents.tools.find_legal_aid(case["city"]),
                          "lawyers": agents.tools.find_lawyers("property/land", case["city"], case["language"])}
    if result["referrals"] != expected_referrals:
        raise AssertionError("Guidance referrals differ from authoritative local tool results.")
    if not result["referrals"]["legal_aid"] or not result["referrals"]["lawyers"]:
        raise AssertionError("Fixed Pune smoke input should demonstrate both fictional referral categories.")
    if any(item["status"] not in ("mentioned", "check_if_available") for item in result["document_checklist"]):
        raise AssertionError("Guidance checklist contains an unsupported status.")
    if {item["id"] for item in result["past_cases_used"]} != {"C003", "C001", "C002"}:
        raise AssertionError("Guidance citations differ from fixed Research IDs.")
    if result["disclaimer"] != agents.GUIDANCE_DISCLAIMER:
        raise AssertionError("Guidance mandatory disclaimer guarantees were changed.")
    agents._guidance_check_text(json.dumps(result, ensure_ascii=False), {"C003", "C001", "C002"})
    print("Guidance smoke checks passed. Referrals are fictional local records; nothing was contacted.")
    try:
        _marathi_scenario(case, research, analysis, gap)
    except agents.GuidanceAgentError as exc:
        print(f"Marathi Guidance smoke test failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
