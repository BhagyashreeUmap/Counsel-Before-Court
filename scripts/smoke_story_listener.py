"""Manual synthetic demos: python -m scripts.smoke_story_listener.

Consumes API quota. Two independent one-turn calls; the transport may retry
transient failures. Importing this script never runs the demo.
"""
import json
import re

from backend.agents import StoryListenerError, create_empty_case_file, story_listener


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_marathi(result):
    """Lightweight smoke expectations, not proof of factual/legal correctness."""
    case_file = result["updated_case_file"]
    _require(case_file["language"] == "mr", "Marathi language was not normalized to 'mr'.")
    _require(result["danger_flag"] is False and case_file["danger_flag"] is False,
             "Marathi synthetic dispute was incorrectly flagged as immediate danger.")
    _require(result["ready"] is False,
             "Marathi intake was marked ready before the citizen supplied a goal.")
    question = result["next_question"]
    _require(isinstance(question, str) and bool(question.strip()),
             "Marathi Story Listener did not produce a follow-up question.")
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    list_marker = r"(?:[-*•]|\d+[.)]|[a-zA-Z][.)])\s+"
    obvious_sequence = re.search(r"[?？]\s*\S", question)
    _require(len(lines) == 1 and not re.match(list_marker, lines[0])
             and not obvious_sequence and len(question) <= 350,
             "Marathi Story Listener produced an obvious question list or lengthy interview dump.")
    for field, description in (
        ("parties_and_relationship", "party/relationship"),
        ("property_or_matter_details", "property"),
    ):
        value = case_file[field]
        _require(isinstance(value, str) and any(character.isalnum() for character in value),
                 f"Marathi Story Listener did not extract useful {description} information.")
    _require(any(isinstance(case_file[field], str)
                 and any(character.isalnum() for character in case_file[field])
                 for field in ("timeline", "other_side_claim", "current_status")),
             "Marathi Story Listener did not extract timeline, other-side claim, or current-status information.")


def _print_result(label, result):
    print(label)
    print(json.dumps({key: result[key] for key in (
        "updated_case_file", "next_question", "ready", "danger_flag"
    )}, ensure_ascii=False, indent=2))


def main():
    try:
        result = story_listener([
            {"role": "user", "content": "My father died last year and now my uncle says our farmland belongs to him."}
        ], create_empty_case_file())
        _print_result("=== English Story Listener Smoke Test ===", result)
        marathi_result = story_listener([
            {"role": "user", "content": "माझ्या वडिलांचे गेल्या वर्षी निधन झाले. आता माझे काका म्हणतात की आमची शेतजमीन त्यांची आहे."}
        ], create_empty_case_file(), question_count=0)
        _print_result("=== Marathi Story Listener Smoke Test ===", marathi_result)
        _check_marathi(marathi_result)
    except StoryListenerError as exc:
        print(f"Smoke test failed: {exc}")
        return 1
    print("Story Listener smoke tests passed: English and Marathi one-turn demos completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
