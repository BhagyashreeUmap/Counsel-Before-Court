"""Manual only: python -m scripts.smoke_llm (one real Gemini request).

Uses synthetic text only, prints no response content, and consumes API quota.
This file is deliberately outside unittest discovery and named without test_.
"""

from backend.llm import LLMError, _ask_llm_json


def main() -> int:
    schema = {
        "type": "object",
        "properties": {
            "legal_area": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["legal_area", "summary"],
        "additionalProperties": False,
    }
    try:
        # Shared wrapper path with retries disabled: at most one actual request.
        _ask_llm_json(
            "You extract structured information from synthetic legal-demo text. "
            "This is demonstration data, not legal advice or a substitute for a lawyer.",
            "My uncle is claiming land that is still registered in my late father's name.",
            schema,
            max_attempts=1,
        )
    except LLMError as exc:
        print(f"Smoke test failed: {exc}")
        return 1
    print("Smoke test passed: Gemini returned a validated JSON object.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
