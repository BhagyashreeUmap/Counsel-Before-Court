"""Manual English Analyst-only demo: python -m scripts.smoke_analyst_agent."""
import json
import re

from backend.agents import AnalystAgentError, analyst_agent, create_empty_case_file
from backend.tools import get_case


def main():
    case_file = create_empty_case_file()
    case_file.update(
        legal_area="land_dispute", parties_and_relationship="Citizen, deceased father and uncle",
        property_or_matter_details="Family farmland", timeline="Father died last year",
        other_side_claim="Uncle says the farmland belongs to him",
        current_status="Current dispute about the uncle's farmland claim",
        citizen_goal="Understand the situation and prepare before speaking to a lawyer",
        language="en", danger_flag=False,
    )
    # Actual memory: family claim, documented possession, undocumented use.
    research_result = {"matches": [{"id": case_id, "why_similar": "Fixed synthetic demo selection"}
                                   for case_id in ("C003", "C001", "C002")]}
    print("=== Analyst Agent Smoke Test ===")
    try:
        records = []
        for match in research_result["matches"]:
            record = get_case(match["id"])
            if record is None:
                raise AssertionError("Smoke case ID was not found in authoritative memory.")
            records.append(record)
            print(f"{record['id']}: {record['title']}")
        result = analyst_agent(case_file, research_result)
    except AnalystAgentError as exc:
        print(f"Analyst smoke test failed: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    arrays = ("winning_patterns", "losing_patterns", "key_evidence", "risks")
    if set(result) != set(arrays) | {"typical_time", "typical_cost_level"}:
        raise AssertionError("Analyst result does not contain exactly the six required keys.")
    if not all(isinstance(result[field], list) for field in arrays):
        raise AssertionError("Analyst qualitative fields must be lists.")
    if not all(isinstance(result[field], str) for field in ("typical_time", "typical_cost_level")):
        raise AssertionError("Analyst time/cost observations must be strings.")
    text = json.dumps(result, ensure_ascii=False).casefold()
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|percent)|probability of (?:winning|losing)|chance of (?:winning|losing)", text):
        raise AssertionError("Analyst output contains an obvious probability prediction.")
    if re.search(r"(?:you|citizen) (?:will|are likely to|is likely to) (?:win|lose)|guarantee(?:s|d)? (?:success|a win|victory)", text):
        raise AssertionError("Analyst output contains obvious win/loss prediction or guarantee wording.")
    times = [record["time_taken_months"] for record in records]
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", result["typical_time"])]
    if numbers != [len(times), min(times), max(times)] or "months" not in result["typical_time"]:
        raise AssertionError("Analyst time observation does not match authoritative recorded durations.")
    observed_costs = {record["cost_level"] for record in records}
    reported_costs = set(re.findall(r"\b(?:low|medium|high)\b", result["typical_cost_level"]))
    if reported_costs != observed_costs or re.search(r"₹|rupees|\brs\.?\b", result["typical_cost_level"], re.I):
        raise AssertionError("Analyst cost observation does not match authoritative cost categories.")
    print("Analyst smoke checks passed; observations are synthetic and do not predict citizen outcomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
