"""Manual only: python -m scripts.smoke_research_agent. Consumes Gemini quota."""
import json

from backend.agents import ResearchAgentError, create_empty_case_file, research_agent
from backend.tools import get_case


def main():
    case_file = create_empty_case_file()
    case_file.update(
        legal_area="land_dispute", parties_and_relationship="Citizen, deceased father, and uncle",
        property_or_matter_details="Family farmland",
        timeline="Father died last year", other_side_claim="Uncle says the farmland belongs to him",
        current_status="Current dispute over the uncle's claim",
        citizen_goal="Understand the situation and prepare before speaking to a lawyer",
        language="en",
    )
    print("=== English Research Agent Smoke Test ===")
    try:
        result = research_agent(case_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["matches"]:
            print("Smoke test failed: no synthetic matches were returned.")
            return 1
        for match in result["matches"]:
            record = get_case(match["id"])
            if record is None:
                print("Smoke test failed: a selected case could not be found in stored data.")
                return 1
            print(f"{match['id']}: {record['title']}")
    except ResearchAgentError as exc:
        print(f"Smoke test failed: {exc}")
        return 1
    except Exception:
        print("Smoke test failed: authoritative case titles could not be loaded.")
        return 1
    if not any(match["id"] == "C003" for match in result["matches"]):
        print("Warning: the ideally relevant synthetic case C003 was not selected.")

    marathi_case_file = create_empty_case_file()
    marathi_case_file.update(
        legal_area="land_dispute",
        parties_and_relationship="नागरिक, दिवंगत वडील आणि काका",
        property_or_matter_details="कुटुंबाची शेतजमीन",
        timeline="वडिलांचे गेल्या वर्षी निधन झाले",
        other_side_claim="काका म्हणतात की कुटुंबाची शेतजमीन त्यांची आहे",
        current_status="काकांच्या दाव्यामुळे शेतजमिनीबाबत सध्या वाद सुरू आहे",
        citizen_goal="मला परिस्थिती समजून घ्यायची आहे आणि वकिलांशी बोलण्यापूर्वी तयारी करायची आहे",
        language="mr",
        danger_flag=False,
    )
    print("=== Marathi Research Agent Smoke Test ===")
    try:
        marathi_result = research_agent(marathi_case_file)
        print(json.dumps(marathi_result, ensure_ascii=False, indent=2))
        if not marathi_result["matches"]:
            print("Marathi smoke test failed: no synthetic matches were returned.")
            return 1
        for match in marathi_result["matches"]:
            record = get_case(match["id"])
            if record is None or record.get("id") != match["id"]:
                print("Marathi smoke test failed: a selected ID does not exist in stored case memory.")
                return 1
            print(f"ID: {match['id']}")
            print(match["why_similar"])
            print(f"Stored title: {record['title']}")
    except ResearchAgentError as exc:
        print(f"Marathi smoke test failed: {exc}")
        return 1
    except Exception:
        print("Marathi smoke test failed: results or authoritative titles could not be read.")
        return 1
    if not any(match["id"] == "C003" for match in marathi_result["matches"]):
        print("Warning: Marathi retrieval did not select the ideally relevant synthetic case C003.")
    print("Research Agent smoke test completed; matches are synthetic examples, not precedent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
