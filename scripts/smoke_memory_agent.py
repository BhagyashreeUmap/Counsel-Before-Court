"""Manual isolated Memory demos: python -m scripts.smoke_memory_agent. Two LLM calls."""
import json
from functools import partial
from pathlib import Path
import tempfile
from unittest.mock import patch

from backend import database, memory, tools, llm


def _meaningful_marathi(text, field):
    if not isinstance(text, str) or sum("\u0900" <= character <= "\u097f" for character in text) < 8:
        raise AssertionError(f"Marathi {field} did not preserve meaningful Marathi prose.")


def _marathi_smoke():
    print("=== Marathi Memory Agent Smoke Test ===")
    case = {"legal_area": "land_dispute", "name": "कल्पित उदाहरणकर", "phone": "8765432109",
        "email": "marathi-demo@example.invalid", "address": "४५६ काल्पनिक गल्ली, नमुना नगर, कल्पितवाडी",
        "village": "कल्पितवाडी", "government_id": "223456789012", "account_number": "887766554433",
        "parties_and_relationship": "कल्पित उदाहरणकर यांच्या काकांनी कुटुंबाच्या शेतजमिनीवर दावा केला",
        "property_or_matter_details": "४५६ काल्पनिक गल्ली, नमुना नगर, कल्पितवाडी येथील शेतजमिनीच्या सिंचन कालव्याबाबत वाद",
        "documents_mentioned": ["सिंचनाच्या पावत्या", "जमिनीची नोंद"]}
    details = {"outcome": "settled", "time_taken_months": 7, "cost_level": "low",
        "options_tried": ["काकांशी चर्चा केली", "कुटुंबातील बैठक घेतली"],
        "key_factors": ["मला वाटते सिंचनाच्या जुन्या पावत्यांमुळे परिस्थिती स्पष्ट झाली"],
        "lesson": "जमिनीच्या आणि सिंचनाच्या नोंदींच्या प्रती जपून ठेवण्याचे महत्त्व मला समजले"}
    production_path = database.USER_MEMORY_PATH
    production_before = production_path.read_bytes() if production_path.exists() else None
    corpus_before = database.CASES_PATH.read_bytes()
    query = "सिंचन कालवा शेतजमीन काका"
    with tempfile.TemporaryDirectory() as directory, patch.object(database, "USER_MEMORY_PATH", Path(directory) / "memory.json"):
        tools._cases.cache_clear()
        tools._search_index.cache_clear()
        try:
            assert memory.prepare_memory_record(case, details, False) is None
            assert not database.USER_MEMORY_PATH.exists()
            with patch.object(llm, "ask_llm_json", partial(llm._ask_llm_json, max_attempts=1)):
                preview = memory.prepare_memory_record(case, details, True)
            print("=== Exact Marathi Memory Preview ===")
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            serialized = json.dumps(preview, ensure_ascii=False)
            for value in (case["name"], case["phone"], case["email"], case["address"], case["village"],
                          "४५६ काल्पनिक गल्ली", "नमुना नगर", case["government_id"], case["account_number"]):
                if value in serialized:
                    raise AssertionError("Raw Marathi fixture identity/contact/place data survived anonymization.")
            _meaningful_marathi(preview["story_summary"], "story")
            _meaningful_marathi(preview["lesson"], "lesson")
            for field in ("options_tried", "key_factors"):
                if not preview[field]:
                    raise AssertionError(f"Marathi {field} learning was discarded.")
                for text in preview[field]:
                    _meaningful_marathi(text, field)
            if "काका" not in preview["story_summary"] and "काकां" not in preview["story_summary"]:
                raise AssertionError("Marathi uncle relationship was lost.")
            if not any(word in serialized for word in ("सिंचन", "शेतजमीन", "जमिनी")):
                raise AssertionError("Marathi land/irrigation meaning was lost.")
            if not any(word in preview["story_summary"] for word in ("सिंचन", "कालव")):
                raise AssertionError("Marathi irrigation dispute meaning was lost.")
            if not any(word in " ".join(preview["key_factors"]) for word in ("पावत्य", "पावती", "नोंद")):
                raise AssertionError("Marathi evidence-related factors were lost.")
            if not any(word in preview["lesson"] for word in ("प्रती", "जप", "नोंद")):
                raise AssertionError("Marathi record-preservation lesson was lost.")
            assert tools.get_case(preview["id"]) is None
            assert preview["id"] not in [item["id"] for item in tools.search_cases(query, 20)]
            assert not database.USER_MEMORY_PATH.exists()
            saved = memory.confirm_save_memory(preview, True)
            assert saved == preview
            print("=== Marathi Search After Save ===")
            results = tools.search_cases(query, 20)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            assert preview["id"] in [item["id"] for item in results]
            retrieved = tools.get_case(preview["id"])
            retrieved.pop("retrieval_notice")
            assert retrieved == preview and retrieved["source"] == memory.SOURCE
            assert database.CASES_PATH.read_bytes() == corpus_before
        finally:
            tools._cases.cache_clear()
            tools._search_index.cache_clear()
    assert (production_path.read_bytes() if production_path.exists() else None) == production_before


def main():
    print("=== Memory Agent Smoke Test ===")
    case = {"legal_area": "land_dispute", "name": "Rohan Exampleperson", "phone": "9876543210",
        "email": "rohan@example.com", "address": "123 Demo Street, Example Nagar, Pune", "city": "Pune",
        "government_id": "123456789012", "account_number": "998877665544",
        "parties_and_relationship": "Rohan Exampleperson's uncle disputed the orchard irrigation sluice",
        "property_or_matter_details": "an orchard irrigation sluice at 123 Demo Street, Example Nagar, Pune",
        "documents_mentioned": ["land record", "irrigation receipts"]}
    details = {"outcome": "settled", "time_taken_months": 9, "cost_level": "low",
        "options_tried": ["family discussion"], "key_factors": ["i think irrigation records helped"],
        "lesson": "i learned to keep copies of records"}
    original = database.CASES_PATH.read_bytes()
    with tempfile.TemporaryDirectory() as directory, patch.object(database, "USER_MEMORY_PATH", Path(directory) / "memory.json"):
        tools._cases.cache_clear()
        tools._search_index.cache_clear()
        try:
            assert not any(item["id"].startswith("U") for item in tools.search_cases("orchard irrigation sluice", 20))
            assert memory.prepare_memory_record(case, details, False) is None
            assert not database.USER_MEMORY_PATH.exists()
            with patch.object(llm, "ask_llm_json", partial(llm._ask_llm_json, max_attempts=1)):
                preview = memory.prepare_memory_record(case, details, True)
            print("=== Exact Memory Preview ===")
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            for value in ("Rohan Exampleperson", "9876543210", "rohan@example.com", "123 Demo Street", "123456789012", "998877665544"):
                assert value not in json.dumps(preview)
            assert "uncle" in preview["story_summary"].lower()
            assert tools.get_case(preview["id"]) is None
            saved = memory.confirm_save_memory(preview, True)
            assert saved == preview
            print("Saved exact confirmed preview.")
            print("=== Search After Save ===")
            results = tools.search_cases("orchard irrigation sluice", 20)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            assert preview["id"] in [item["id"] for item in results]
            record = tools.get_case(preview["id"])
            record.pop("retrieval_notice")
            assert record == preview and record["source"] == memory.SOURCE
            assert database.CASES_PATH.read_bytes() == original
        finally:
            tools._cases.cache_clear()
            tools._search_index.cache_clear()
    _marathi_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
