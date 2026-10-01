"""Manual isolated English Gap demo: python -m scripts.smoke_gap_agent."""
import json
import re

from backend.agents import GapAgentError, _gap_canonical, create_empty_case_file, gap_agent


def main():
    case_file = create_empty_case_file()
    case_file.update(
        legal_area="land_dispute", parties_and_relationship="Citizen, deceased father and uncle",
        property_or_matter_details="Family farmland", timeline="Father died last year",
        other_side_claim="Uncle claims the farmland", current_status="Current family farmland dispute",
        citizen_goal="Understand the situation and prepare before speaking to a lawyer",
        documents_mentioned=["Death Cert"], language="en",
    )
    # Actual C003/C001 document labels; no real Research or Analyst invocation.
    analyst_result = {"key_evidence": ["death certificate", "legal heir certificate", "family tree",
        "old land record in father's name", "land tax receipts for over 20 years",
        "statements from two independent neighbours"]}
    print("=== Gap Agent Smoke Test ===")
    print("Citizen documents:", json.dumps(case_file["documents_mentioned"]))
    print("Analyst key_evidence:", json.dumps(analyst_result["key_evidence"]))
    try:
        result = gap_agent(case_file, analyst_result)
    except GapAgentError as exc:
        print(f"Gap smoke test failed: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if set(result) != {"missing_evidence", "questions"}:
        raise AssertionError("Gap result must contain exactly two public keys.")
    evidence, questions = result["missing_evidence"], result["questions"]
    if not isinstance(evidence, list) or not isinstance(questions, list):
        raise AssertionError("Gap evidence and questions must be lists.")
    if not 0 < len(evidence) == len(questions) <= 3:
        raise AssertionError("Gap smoke scenario must return one to three aligned evidence/question pairs.")
    mentioned = {_gap_canonical(item) for item in case_file["documents_mentioned"]}
    allowed = {_gap_canonical(item) for item in analyst_result["key_evidence"]}
    keys = [_gap_canonical(item) for item in evidence]
    if len(set(keys)) != len(keys) or any(key not in allowed or key in mentioned for key in keys):
        raise AssertionError("Gap returned duplicate, already-mentioned or ungrounded evidence.")
    if not all(isinstance(question, str) and question.strip() for question in questions):
        raise AssertionError("Gap questions must be nonempty strings.")
    text = " ".join(questions).casefold()
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|percent)|(?:you|citizen) (?:will|are likely to|is likely to) (?:win|lose)|guarantee(?:s|d)? (?:success|victory|a win)|(?:chance|probability) of (?:winning|losing)", text):
        raise AssertionError("Gap questions contain obvious outcome prediction or guarantee wording.")
    print("Gap smoke checks passed; selected evidence is not yet mentioned, not confirmed absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
